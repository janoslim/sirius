/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// CUDA managed memory for the hash-table state of large hash joins:
// `sirius.memory.gpu.large_hash_join_in_managed_memory` (FlowCSD fork, setting (라)).
//
// With the option on, a hash join whose build side exceeds `max_build_hash_table_bytes` (the size
// above which Sirius partitions the join) and that one hash table can serve runs unpartitioned, and
// only that join's hash-table state lives in CUDA managed memory: the build batches that arrive
// after the first `max_build_hash_table_bytes`, the concatenated build table, the build-key cast
// columns and the cuDF hash-table objects. Every other allocation stays in the GPU memory space
// (device memory with its reservations, accounting and downgrade), so the managed state is outside
// that accounting and is never downgraded. A build longer than one cuDF table (`cudf::size_type`
// rows) stays partitioned, and its build-side PARTITION stops copying batches into managed memory
// once it sees that (sirius_physical_partition).
//
// - managed_hash_state_pool: CUDA managed memory plus a registry of its live allocations, so that a
//   freed pointer finds its way back and a batch can be recognized as managed without a driver call.
//   The build batches copied on arrival always live in host-resident managed memory: an RMM pool
//   over cudaMallocManaged chunks advised to stay in host memory and be mapped for the GPU over
//   PCIe. The concatenated build table and the hash table follow `managed_hash_join_placement`:
//   `migrate_on_access` uses the device's default managed pool (the pool that
//   `sirius.memory.gpu.managed_memory` uses), `host_resident` the host-resident pool.
// - managed_hash_join_router: cuDF's current device resource while the option is on. cuDF's hash-join
//   objects take no memory resource (cudf::filtered_join, cudf::distinct_hash_join and the
//   preprocessed build table of cudf::hash_join allocate from the current resource), so placing them
//   is only possible by routing per thread: inside a scoped_managed_hash_join_allocation an
//   allocation goes to the pool's placement memory, otherwise to the GPU memory space's
//   allocator. A deallocation goes to the pool exactly when its pointer is a live pool allocation.
// - is_managed_hash_state(): whether a batch's table lives in the pool, so that downgrade and the
//   task reservation estimates leave it out.
//
// Header-only like managed_gpu_memory.hpp, so the option adds no source file to the build. One
// process runs one Sirius context, so the pool is process-wide and never destroyed: a buffer freed
// during static destruction must still find it.

#include "flowcsd/flowcsd_stats.hpp"
#include "log/logging.hpp"
#include "memory/managed_gpu_memory.hpp"
#include "sirius_config.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/common.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>

namespace sirius::memory {

/**
 * @brief Chunks of managed memory that stay in host memory: cudaMallocManaged plus cudaMemAdvise
 * preferred location = host memory and accessed by = the GPU, so the GPU maps the pages over PCIe
 * instead of migrating them. Freed with cudaFree.
 *
 * Allocations of a managed memory pool reject cudaMemAdvise ("operation not supported", Slurm job
 * 1175), and a pool located in host memory let the GPU write its first pages at 1.0 GB/s against
 * 5.8 GB/s for advised cudaMallocManaged (job 1176), hence this upstream. Synchronous: cudaFree
 * synchronizes the device, so it only feeds rmm::mr::pool_memory_resource (managed_hash_state_pool),
 * which keeps its chunks for the life of the process.
 */
class advised_managed_memory_resource {
 public:
  explicit advised_managed_memory_resource(int device_id) noexcept : _device_id{device_id} {}

  /// @throws std::runtime_error when cudaMallocManaged or a hint fails (the chunk is freed first).
  void* allocate_sync(std::size_t bytes, std::size_t /*alignment*/)
  {
    void* ptr = nullptr;
    auto error = cudaMallocManaged(&ptr, bytes);
    if (error == cudaSuccess) {
      cudaMemLocation host{};
      host.type = cudaMemLocationTypeHost;
      cudaMemLocation gpu{};
      gpu.type = cudaMemLocationTypeDevice;
      gpu.id   = _device_id;
      error    = cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, host);
      if (error == cudaSuccess) {
        error = cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, gpu);
      }
      if (error != cudaSuccess) { static_cast<void>(cudaFree(ptr)); }
    }
    if (error != cudaSuccess) {
      // Reported by the exception; do not leave it for later cudaGetLastError checks.
      static_cast<void>(cudaGetLastError());
      throw std::runtime_error(
        std::string{"managed hash-join state (host_resident): cudaMallocManaged/cudaMemAdvise of "} +
        std::to_string(bytes) + " B failed: " + cudaGetErrorString(error));
    }
    return ptr;
  }

  void deallocate_sync(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) noexcept
  {
    static_cast<void>(cudaFree(ptr));
  }

  void* allocate(::cuda::stream_ref /*stream*/, std::size_t bytes, std::size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(::cuda::stream_ref /*stream*/,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  friend bool operator==(advised_managed_memory_resource const& lhs,
                         advised_managed_memory_resource const& rhs) noexcept
  {
    return lhs._device_id == rhs._device_id;
  }

  friend void get_property(advised_managed_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  int _device_id;
};

/// Which managed memory of managed_hash_state_pool an allocation comes from.
enum class managed_memory_kind : std::uint8_t {
  /// The memory `managed_hash_join_placement` names: the folded build table, the build-key cast
  /// columns and the cuDF hash table.
  placement,
  /// Host-resident managed memory whatever the placement: the build batches that the build-side
  /// PARTITION copies on arrival, before they are folded.
  host_resident,
};

/**
 * @brief Managed memory for the hash-table state of large hash joins, with one registry of the live
 * allocations (see the file comment).
 *
 * Holds two managed memories: an RMM pool over advised_managed_memory_resource chunks
 * (host-resident) and, with migrate_on_access placement, the device's default managed memory pool.
 * The build batches that the build-side PARTITION copies on arrival always go to the host-resident
 * memory: the fold reads each of them once, in order, so keeping them on the GPU gains nothing, and
 * under migrate_on_access they filled the GPU and competed with the GPU memory space's own pool (Q3
 * at SF=1000 then failed with an illegal memory access while partitioning them, Slurm job 1183;
 * that this caused it is suspected, not proven). The registry covers both memories, so a freed
 * pointer returns to the memory it came from, is_managed_hash_state() recognizes both, and the
 * query's peak statistic counts both.
 *
 * cuDF calls that take a memory resource (cudf::concatenate, the cudf::table copy constructor)
 * allocate here through resource_ref(). Thread-safe.
 */
class managed_hash_state_pool {
 public:
  /**
   * @brief Create the host-resident memory on @p device_id (an RMM pool that starts empty and grows
   * on demand) and, for migrate_on_access, also the default managed memory pool of @p device_id
   * (the pool of `managed_memory`). host_resident placement uses the host-resident memory for all.
   */
  managed_hash_state_pool(int device_id, managed_hash_join_placement placement)
    : _host_resident{rmm::mr::pool_memory_resource{
        any_device_resource{advised_managed_memory_resource{device_id}}, 0}},
      _migrate_on_access{
        placement == managed_hash_join_placement::migrate_on_access
          ? std::optional<any_device_resource>{make_default_managed_pool(device_id)}
          : std::nullopt},
      _device_id{device_id},
      _placement{placement}
  {
  }

  managed_hash_state_pool(managed_hash_state_pool const&)            = delete;
  managed_hash_state_pool& operator=(managed_hash_state_pool const&) = delete;
  managed_hash_state_pool(managed_hash_state_pool&&)                 = delete;
  managed_hash_state_pool& operator=(managed_hash_state_pool&&)      = delete;

  [[nodiscard]] int device_id() const noexcept { return _device_id; }
  [[nodiscard]] managed_hash_join_placement placement() const noexcept { return _placement; }

  /**
   * @brief Allocate @p bytes of @p kind managed memory ordered on @p stream and record it as live.
   * @throws rmm::out_of_memory or rmm::bad_alloc when the memory cannot be allocated.
   */
  void* allocate(managed_memory_kind kind,
                 ::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ptr = memory(kind).allocate(stream, bytes, alignment);
    track_new_allocation(ptr, bytes, kind);
    return ptr;
  }

  /// @brief Synchronous form of allocate().
  void* allocate_sync(managed_memory_kind kind,
                      std::size_t bytes,
                      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ptr = memory(kind).allocate_sync(bytes, alignment);
    track_new_allocation(ptr, bytes, kind);
    return ptr;
  }

  /**
   * @brief Free @p ptr on @p stream, into the memory it came from, when it is a live allocation of
   * this pool.
   * @return false, having done nothing, when @p ptr is not one (it came from another resource).
   */
  bool deallocate_if_owned(::cuda::stream_ref stream,
                           void* ptr,
                           std::size_t bytes,
                           std::size_t alignment) noexcept
  {
    auto const kind = erase_live_allocation(ptr);
    if (!kind.has_value()) { return false; }
    memory(*kind).deallocate(stream, ptr, bytes, alignment);
    return true;
  }

  /// @brief Synchronous form of deallocate_if_owned().
  bool deallocate_sync_if_owned(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    auto const kind = erase_live_allocation(ptr);
    if (!kind.has_value()) { return false; }
    memory(*kind).deallocate_sync(ptr, bytes, alignment);
    return true;
  }

  /// @brief Whether @p address lies inside a live allocation of this pool, in either memory.
  [[nodiscard]] bool contains(void const* address) const
  {
    // Without live allocations nothing can match; the common case skips the lock.
    if (address == nullptr || _live_count.load(std::memory_order_acquire) == 0) { return false; }
    auto const position = reinterpret_cast<std::uintptr_t>(address);
    std::shared_lock const lock(_mutex);
    auto const after = _live_allocations.upper_bound(position);
    if (after == _live_allocations.begin()) { return false; }
    auto const& [start, allocation] = *std::prev(after);
    return position < start + allocation.bytes;
  }

  /// @brief Copyable resource over one memory of a pool: a cuda::mr resource must be copyable,
  /// which the pool (mutex, registry) is not.
  class resource_handle {
   public:
    resource_handle(managed_hash_state_pool& pool, managed_memory_kind kind) noexcept
      : _pool{&pool}, _kind{kind}
    {
    }

    void* allocate(::cuda::stream_ref stream,
                   std::size_t bytes,
                   std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
    {
      return _pool->allocate(_kind, stream, bytes, alignment);
    }
    void deallocate(::cuda::stream_ref stream,
                    void* ptr,
                    std::size_t bytes,
                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
    {
      _pool->deallocate(_kind, stream, ptr, bytes, alignment);
    }
    void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
    {
      return _pool->allocate_sync(_kind, bytes, alignment);
    }
    void deallocate_sync(void* ptr,
                         std::size_t bytes,
                         std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
    {
      _pool->deallocate_sync(_kind, ptr, bytes, alignment);
    }
    friend bool operator==(resource_handle const& lhs, resource_handle const& rhs) noexcept
    {
      return lhs._pool == rhs._pool && lhs._kind == rhs._kind;
    }
    friend void get_property(resource_handle const&, ::cuda::mr::device_accessible) noexcept {}

   private:
    managed_hash_state_pool* _pool;
    managed_memory_kind _kind;
  };

  /// @brief Memory resource reference to the @p kind memory of this pool, valid as long as the pool
  /// (the process).
  [[nodiscard]] rmm::device_async_resource_ref resource_ref(managed_memory_kind kind) noexcept
  {
    if (kind == managed_memory_kind::host_resident) {
      return rmm::device_async_resource_ref{_host_resident_handle};
    }
    return rmm::device_async_resource_ref{_placement_handle};
  }

 private:
  /// Spelled `::cuda` because Sirius has its own sirius::cuda namespace.
  using any_device_resource = ::cuda::mr::any_resource<::cuda::mr::device_accessible>;

  /// A live allocation: its size and the memory it came from.
  struct live_allocation {
    std::size_t bytes;
    managed_memory_kind kind;
  };

  /// @brief The memory that @p kind allocations come from.
  [[nodiscard]] any_device_resource& memory(managed_memory_kind kind) noexcept
  {
    // With host_resident placement the placement memory is the host-resident memory.
    if (kind == managed_memory_kind::placement && _migrate_on_access.has_value()) {
      return *_migrate_on_access;
    }
    return _host_resident;
  }

  /// @brief Free @p ptr, which a handle over the @p kind memory gives back, on @p stream.
  void deallocate(managed_memory_kind kind,
                  ::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    if (deallocate_if_owned(stream, ptr, bytes, alignment)) { return; }
    report_untracked_free(ptr);
    memory(kind).deallocate(stream, ptr, bytes, alignment);
  }

  /// @brief Synchronous form of deallocate().
  void deallocate_sync(managed_memory_kind kind,
                       void* ptr,
                       std::size_t bytes,
                       std::size_t alignment) noexcept
  {
    if (deallocate_sync_if_owned(ptr, bytes, alignment)) { return; }
    report_untracked_free(ptr);
    memory(kind).deallocate_sync(ptr, bytes, alignment);
  }

  /// @brief Record a new allocation as live and raise the query's peak statistic.
  void track_new_allocation(void* ptr, std::size_t bytes, managed_memory_kind kind)
  {
    // A zero-byte request returns no memory: nothing to track.
    if (ptr == nullptr) { return; }
    std::size_t live_bytes = 0;
    {
      std::unique_lock const lock(_mutex);
      _live_allocations.insert_or_assign(reinterpret_cast<std::uintptr_t>(ptr),
                                         live_allocation{bytes, kind});
      _live_bytes += bytes;
      live_bytes = _live_bytes;
      _live_count.store(_live_allocations.size(), std::memory_order_release);
    }
    flowcsd::max_bytes(flowcsd::byte_counter::managed_hash_state_peak, live_bytes);
  }

  /// @brief Remove @p ptr from the live allocations.
  /// @return The memory it came from, or no value when it is not a live allocation.
  std::optional<managed_memory_kind> erase_live_allocation(void* ptr) noexcept
  {
    if (ptr == nullptr || _live_count.load(std::memory_order_acquire) == 0) { return std::nullopt; }
    auto const position = reinterpret_cast<std::uintptr_t>(ptr);
    {
      // Most frees routed through managed_hash_join_router are not managed: look up shared first.
      std::shared_lock const lookup(_mutex);
      if (!_live_allocations.contains(position)) { return std::nullopt; }
    }
    std::unique_lock const lock(_mutex);
    auto const found = _live_allocations.find(position);
    if (found == _live_allocations.end()) { return std::nullopt; }
    auto const kind = found->second.kind;
    _live_bytes -= found->second.bytes;
    _live_allocations.erase(found);
    _live_count.store(_live_allocations.size(), std::memory_order_release);
    return kind;
  }

  /// @brief Report a pointer freed through a handle that is not a live allocation.
  static void report_untracked_free(void* ptr) noexcept
  {
    // Every non-null pointer of this pool is registered at allocation, so this is a bookkeeping
    // bug; the caller still returns the memory to the handle's memory.
    if (ptr == nullptr) { return; }
    SIRIUS_LOG_ERROR("[managed_hash_state_pool] freed {} which is not a live managed allocation",
                     ptr);
  }

  /// Host-resident managed memory (constructor): the PARTITION's build copies, and with
  /// host_resident placement everything.
  any_device_resource _host_resident;
  /// The device's default managed memory pool with migrate_on_access placement; empty otherwise.
  std::optional<any_device_resource> _migrate_on_access;
  int _device_id;
  managed_hash_join_placement _placement;
  mutable std::shared_mutex _mutex;
  std::map<std::uintptr_t, live_allocation> _live_allocations;  ///< by start; guarded by _mutex
  std::size_t _live_bytes{0};               ///< both memories; guarded by _mutex
  std::atomic<std::size_t> _live_count{0};  ///< _live_allocations.size(), read without the lock
  /// What resource_ref() refers to.
  resource_handle _placement_handle{*this, managed_memory_kind::placement};
  resource_handle _host_resident_handle{*this, managed_memory_kind::host_resident};
};

namespace managed_hash_join_detail {

/// Open scoped_managed_hash_join_allocation scopes on the calling thread.
inline thread_local int open_allocation_scopes = 0;

/// The process-wide pool (install_managed_hash_state_pool), or nullptr while the option is off.
inline std::atomic<managed_hash_state_pool*> installed_pool{nullptr};

}  // namespace managed_hash_join_detail

/// @brief The process-wide managed hash-state pool, or nullptr when the option is off.
[[nodiscard]] inline managed_hash_state_pool* managed_hash_state() noexcept
{
  return managed_hash_join_detail::installed_pool.load(std::memory_order_acquire);
}

/**
 * @brief Create the process-wide pool on @p device_id; called once, when the GPU memory spaces are
 * created (sirius_memory_reservation_manager).
 * @throws std::logic_error when a pool for another device or placement already exists.
 */
inline managed_hash_state_pool& install_managed_hash_state_pool(int device_id,
                                                                managed_hash_join_placement placement)
{
  static std::mutex install_mutex;
  std::lock_guard const lock(install_mutex);
  if (auto* installed = managed_hash_state(); installed != nullptr) {
    if (installed->device_id() != device_id || installed->placement() != placement) {
      throw std::logic_error(
        "managed hash-join state: one process supports one GPU and one page placement");
    }
    return *installed;
  }
  // Never destroyed (file comment).
  auto* pool = new managed_hash_state_pool(device_id, placement);
  managed_hash_join_detail::installed_pool.store(pool, std::memory_order_release);
  return *pool;
}

/**
 * @brief While alive, the calling thread's allocations through cuDF's current device resource go to
 * the placement memory of the managed hash-state pool (managed_hash_join_router). Nests.
 */
class scoped_managed_hash_join_allocation {
 public:
  scoped_managed_hash_join_allocation() noexcept
  {
    ++managed_hash_join_detail::open_allocation_scopes;
  }
  ~scoped_managed_hash_join_allocation() { --managed_hash_join_detail::open_allocation_scopes; }

  scoped_managed_hash_join_allocation(scoped_managed_hash_join_allocation const&) = delete;
  scoped_managed_hash_join_allocation& operator=(scoped_managed_hash_join_allocation const&) = delete;
};

/**
 * @brief cuDF's current device resource while the option is on: the calling thread's allocations go
 * to the placement memory of the managed hash-state pool inside a
 * scoped_managed_hash_join_allocation and to the GPU memory space's allocator otherwise; a
 * deallocation goes to the pool when its pointer is a live pool allocation and to the GPU memory
 * space's allocator otherwise.
 *
 * Copyable view: both targets outlive it (the GPU memory space owns its allocator, and the pool is
 * never destroyed).
 */
class managed_hash_join_router {
 public:
  managed_hash_join_router(rmm::device_async_resource_ref space_allocator,
                           managed_hash_state_pool& managed_pool) noexcept
    : _space_allocator{space_allocator}, _managed_pool{&managed_pool}
  {
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (managed_hash_join_detail::open_allocation_scopes > 0) {
      return _managed_pool->allocate(managed_memory_kind::placement, stream, bytes, alignment);
    }
    return _space_allocator.allocate(stream, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (_managed_pool->deallocate_if_owned(stream, ptr, bytes, alignment)) { return; }
    _space_allocator.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (managed_hash_join_detail::open_allocation_scopes > 0) {
      return _managed_pool->allocate_sync(managed_memory_kind::placement, bytes, alignment);
    }
    return _space_allocator.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (_managed_pool->deallocate_sync_if_owned(ptr, bytes, alignment)) { return; }
    _space_allocator.deallocate_sync(ptr, bytes, alignment);
  }

  friend bool operator==(managed_hash_join_router const& lhs,
                         managed_hash_join_router const& rhs) noexcept
  {
    return lhs._managed_pool == rhs._managed_pool && lhs._space_allocator == rhs._space_allocator;
  }

  friend void get_property(managed_hash_join_router const&, ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  rmm::device_async_resource_ref _space_allocator;
  managed_hash_state_pool* _managed_pool;
};

/**
 * @brief Deep copy of @p view into the @p kind memory of the managed hash-state pool, ordered on
 * @p stream.
 * @throws std::logic_error when the option is off.
 */
[[nodiscard]] inline std::unique_ptr<cudf::table> copy_to_managed_hash_state(
  cudf::table_view const& view, managed_memory_kind kind, ::cuda::stream_ref stream)
{
  auto* pool = managed_hash_state();
  if (pool == nullptr) {
    throw std::logic_error(
      "copy_to_managed_hash_state: sirius.memory.gpu.large_hash_join_in_managed_memory is off");
  }
  return std::make_unique<cudf::table>(view, stream, pool->resource_ref(kind));
}

namespace managed_hash_join_detail {

/// First device address among @p column's data, null mask and children, or nullptr if it has none.
inline void const* first_buffer_address(cudf::column_view const& column)
{
  if (column.head() != nullptr) { return column.head(); }
  if (column.null_mask() != nullptr) { return column.null_mask(); }
  for (auto child = column.child_begin(); child != column.child_end(); ++child) {
    if (auto const* address = first_buffer_address(*child); address != nullptr) { return address; }
  }
  return nullptr;
}

}  // namespace managed_hash_join_detail

/**
 * @brief Whether @p data is a GPU table held in the managed hash-state pool (either memory).
 *
 * A managed table is allocated as a whole in the pool (copy_to_managed_hash_state or a concatenation
 * into the pool), so its first buffer decides. Always false while the option is off; a registry
 * lookup instead of a driver call (cudaPointerGetAttributes) otherwise.
 */
[[nodiscard]] inline bool is_managed_hash_state(cucascade::idata_representation const* data)
{
  auto const* pool = managed_hash_state();
  if (pool == nullptr || data == nullptr) { return false; }
  auto const* gpu = dynamic_cast<cucascade::gpu_table_representation const*>(data);
  if (gpu == nullptr) { return false; }
  for (auto const& column : gpu->get_table_view()) {
    if (auto const* address = managed_hash_join_detail::first_buffer_address(column);
        address != nullptr) {
      return pool->contains(address);
    }
  }
  return false;
}

}  // namespace sirius::memory
