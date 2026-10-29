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
// that accounting and is never downgraded.
//
// - managed_hash_state_pool: CUDA managed memory plus a registry of its live allocations, so that a
//   freed pointer finds its way back and a batch can be recognized as managed without a driver call.
//   `managed_hash_join_placement: migrate_on_access` uses the device's default managed pool (the pool
//   that `sirius.memory.gpu.managed_memory` uses); `host_resident` uses an RMM pool over
//   cudaMallocManaged chunks advised to stay in host memory and be mapped for the GPU over PCIe.
// - managed_hash_join_router: cuDF's current device resource while the option is on. cuDF's hash-join
//   objects take no memory resource (cudf::filtered_join, cudf::distinct_hash_join and the
//   preprocessed build table of cudf::hash_join allocate from the current resource), so placing them
//   is only possible by routing per thread: inside a scoped_managed_hash_join_allocation an
//   allocation goes to the pool, otherwise to the GPU memory space's allocator. A deallocation goes
//   to the pool exactly when its pointer is a live pool allocation.
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

/**
 * @brief Managed memory for the hash-table state of large hash joins, with a registry of the live
 * allocations (see the file comment).
 *
 * cuDF calls that take a memory resource (cudf::concatenate, the cudf::table copy constructor)
 * allocate here through resource_ref(). Thread-safe.
 */
class managed_hash_state_pool {
 public:
  /**
   * @brief Allocate from the default managed memory pool of @p device_id (migrate_on_access, the
   * pool of `managed_memory`) or from an RMM pool over advised_managed_memory_resource chunks
   * (host_resident; the pool starts empty and grows on demand).
   */
  managed_hash_state_pool(int device_id, managed_hash_join_placement placement)
    : _pool{placement == managed_hash_join_placement::host_resident
              ? ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::pool_memory_resource{
                  ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
                    advised_managed_memory_resource{device_id}},
                  0}}
              : ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
                  make_default_managed_pool(device_id)}},
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
   * @brief Allocate @p bytes of managed memory ordered on @p stream and record it as live.
   * @throws rmm::out_of_memory or rmm::bad_alloc when the pool cannot allocate.
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ptr = _pool.allocate(stream, bytes, alignment);
    track_new_allocation(ptr, bytes);
    return ptr;
  }

  /// @brief Free an allocation of this pool on @p stream.
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    forget_freed_allocation(ptr);
    _pool.deallocate(stream, ptr, bytes, alignment);
  }

  /// @brief Synchronous form of allocate().
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ptr = _pool.allocate_sync(bytes, alignment);
    track_new_allocation(ptr, bytes);
    return ptr;
  }

  /// @brief Synchronous form of deallocate().
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    forget_freed_allocation(ptr);
    _pool.deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief Free @p ptr on @p stream when it is a live allocation of this pool.
   * @return false, having done nothing, when @p ptr is not one (it came from another resource).
   */
  bool deallocate_if_owned(::cuda::stream_ref stream,
                           void* ptr,
                           std::size_t bytes,
                           std::size_t alignment) noexcept
  {
    if (!erase_live_allocation(ptr)) { return false; }
    _pool.deallocate(stream, ptr, bytes, alignment);
    return true;
  }

  /// @brief Synchronous form of deallocate_if_owned().
  bool deallocate_sync_if_owned(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    if (!erase_live_allocation(ptr)) { return false; }
    _pool.deallocate_sync(ptr, bytes, alignment);
    return true;
  }

  /// @brief Whether @p address lies inside a live allocation of this pool.
  [[nodiscard]] bool contains(void const* address) const
  {
    // Without live allocations nothing can match; the common case skips the lock.
    if (address == nullptr || _live_count.load(std::memory_order_acquire) == 0) { return false; }
    auto const position = reinterpret_cast<std::uintptr_t>(address);
    std::shared_lock const lock(_mutex);
    auto const after = _live_allocations.upper_bound(position);
    if (after == _live_allocations.begin()) { return false; }
    auto const& [start, bytes] = *std::prev(after);
    return position < start + bytes;
  }

  /// @brief Copyable resource that forwards to a pool: a cuda::mr resource must be copyable, which
  /// the pool (mutex, registry) is not.
  class resource_handle {
   public:
    explicit resource_handle(managed_hash_state_pool& pool) noexcept : _pool{&pool} {}

    void* allocate(::cuda::stream_ref stream,
                   std::size_t bytes,
                   std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
    {
      return _pool->allocate(stream, bytes, alignment);
    }
    void deallocate(::cuda::stream_ref stream,
                    void* ptr,
                    std::size_t bytes,
                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
    {
      _pool->deallocate(stream, ptr, bytes, alignment);
    }
    void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
    {
      return _pool->allocate_sync(bytes, alignment);
    }
    void deallocate_sync(void* ptr,
                         std::size_t bytes,
                         std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
    {
      _pool->deallocate_sync(ptr, bytes, alignment);
    }
    friend bool operator==(resource_handle const& lhs, resource_handle const& rhs) noexcept
    {
      return lhs._pool == rhs._pool;
    }
    friend void get_property(resource_handle const&, ::cuda::mr::device_accessible) noexcept {}

   private:
    managed_hash_state_pool* _pool;
  };

  /// @brief Memory resource reference to this pool, valid as long as the pool (the process).
  [[nodiscard]] rmm::device_async_resource_ref resource_ref() noexcept
  {
    return rmm::device_async_resource_ref{_handle};
  }

 private:
  /// @brief Record a new allocation as live and raise the query's peak statistic.
  void track_new_allocation(void* ptr, std::size_t bytes)
  {
    // A zero-byte request returns no memory: nothing to track.
    if (ptr == nullptr) { return; }
    std::size_t live_bytes = 0;
    {
      std::unique_lock const lock(_mutex);
      _live_allocations.insert_or_assign(reinterpret_cast<std::uintptr_t>(ptr), bytes);
      _live_bytes += bytes;
      live_bytes = _live_bytes;
      _live_count.store(_live_allocations.size(), std::memory_order_release);
    }
    flowcsd::max_bytes(flowcsd::byte_counter::managed_hash_state_peak, live_bytes);
  }

  /// @brief Remove @p ptr from the live allocations. @return false when it is not one.
  bool erase_live_allocation(void* ptr) noexcept
  {
    if (ptr == nullptr || _live_count.load(std::memory_order_acquire) == 0) { return false; }
    auto const position = reinterpret_cast<std::uintptr_t>(ptr);
    {
      // Most frees routed through managed_hash_join_router are not managed: look up shared first.
      std::shared_lock const lookup(_mutex);
      if (!_live_allocations.contains(position)) { return false; }
    }
    std::unique_lock const lock(_mutex);
    auto const found = _live_allocations.find(position);
    if (found == _live_allocations.end()) { return false; }
    _live_bytes -= found->second;
    _live_allocations.erase(found);
    _live_count.store(_live_allocations.size(), std::memory_order_release);
    return true;
  }

  /// @brief Remove a pointer freed directly through this pool from the live allocations.
  void forget_freed_allocation(void* ptr) noexcept
  {
    if (ptr == nullptr || erase_live_allocation(ptr)) { return; }
    // Every non-null pointer of this pool is registered at allocation, so this is a bookkeeping bug;
    // the memory is still returned to the pool.
    SIRIUS_LOG_ERROR("[managed_hash_state_pool] freed {} which is not a live managed allocation",
                     ptr);
  }

  /// The managed memory this pool hands out (constructor).
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> _pool;
  int _device_id;
  managed_hash_join_placement _placement;
  mutable std::shared_mutex _mutex;
  std::map<std::uintptr_t, std::size_t> _live_allocations;  ///< start -> bytes; guarded by _mutex
  std::size_t _live_bytes{0};                               ///< guarded by _mutex
  std::atomic<std::size_t> _live_count{0};  ///< _live_allocations.size(), read without the lock
  resource_handle _handle{*this};           ///< what resource_ref() refers to
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
 * the managed hash-state pool (managed_hash_join_router). Nests.
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
 * to the managed hash-state pool inside a scoped_managed_hash_join_allocation and to the GPU memory
 * space's allocator otherwise; a deallocation goes to the pool when its pointer is a live pool
 * allocation and to the GPU memory space's allocator otherwise.
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
      return _managed_pool->allocate(stream, bytes, alignment);
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
      return _managed_pool->allocate_sync(bytes, alignment);
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
 * @brief Deep copy of @p view into the managed hash-state pool, ordered on @p stream.
 * @throws std::logic_error when the option is off.
 */
[[nodiscard]] inline std::unique_ptr<cudf::table> copy_to_managed_hash_state(
  cudf::table_view const& view, ::cuda::stream_ref stream)
{
  auto* pool = managed_hash_state();
  if (pool == nullptr) {
    throw std::logic_error(
      "copy_to_managed_hash_state: sirius.memory.gpu.large_hash_join_in_managed_memory is off");
  }
  return std::make_unique<cudf::table>(view, stream, pool->resource_ref());
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
 * @brief Whether @p data is a GPU table held in the managed hash-state pool.
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
