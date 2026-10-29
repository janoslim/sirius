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

// CUDA managed memory for the GPU memory space: `sirius.memory.gpu.managed_memory` (FlowCSD fork).
//
// With the option on, the GPU memory space allocates from the device's default managed memory
// pool instead of a device memory pool.  The space can then be sized past the GPU
// (`usage_limit_bytes` up to host memory), and the CUDA driver pages data between GPU and host
// memory where Sirius would otherwise spill it.  cuFile, which KvikIO uses for GPUDirect Storage
// reads, cannot DMA into managed memory, so the buffers the Parquet scan reads into come from a
// plain device memory pool per device instead (gds_read_buffer_resource).
//
// Header-only so the option adds no source file to the build.  One process runs one Sirius
// context (as for the process-wide flowcsd knobs), so those device pools are process-wide: created
// with the device's managed GPU memory space and kept until the process exits.

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace sirius::memory {

namespace managed_gpu_memory_detail {

/// Plain device memory pools for GPUDirect Storage read buffers, one per device whose GPU memory
/// space is managed memory.
struct gds_read_pools {
  std::mutex mutex;
  std::map<int, rmm::mr::cuda_async_memory_resource> by_device;  ///< guarded by mutex
  std::atomic<bool> any_device{false};  ///< lets the default (device memory) path skip the lock
};

/// The process-wide pools.  Never destroyed on purpose: a chunk buffer freed during static
/// destruction must still find its pool.
inline gds_read_pools& process_gds_read_pools()
{
  static auto* const pools = new gds_read_pools{};
  return *pools;
}

}  // namespace managed_gpu_memory_detail

/**
 * @brief GPU memory space allocator for `sirius.memory.gpu.managed_memory: true`.
 *
 * Allocates CUDA managed memory from @p device_id's default managed pool
 * (`rmm::mr::cuda_async_managed_memory_resource`, i.e. cudaMallocFromPoolAsync), and creates the
 * device's plain device memory pool for GPUDirect Storage read buffers.  @p capacity is unused:
 * managed memory needs no pool sized up front, and the GPU memory space's reservation adaptor
 * enforces the configured capacity.  Matches cucascade::memory::DeviceMemoryResourceFactoryFn
 * (spelled `::cuda` because Sirius has its own sirius::cuda namespace).
 */
inline ::cuda::mr::any_resource<::cuda::mr::device_accessible> make_managed_gpu_memory_resource(
  int device_id, [[maybe_unused]] std::size_t capacity)
{
  rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{device_id}};
  rmm::mr::cuda_async_managed_memory_resource managed_pool;
  // Keep freed memory in the pool across synchronizations, as the device memory pool of the
  // default path does (rmm::mr::cuda_async_memory_resource sets its release threshold to
  // UINT64_MAX).  The default managed pool's threshold is 0, which would give freed memory back at
  // every synchronization and fault it in again on the next allocation.
  std::uint64_t keep_freed_bytes = std::numeric_limits<std::uint64_t>::max();
  RMM_CUDA_TRY(cudaMemPoolSetAttribute(
    managed_pool.pool_handle(), cudaMemPoolAttrReleaseThreshold, &keep_freed_bytes));

  // The read-buffer pool is a default rmm::mr::cuda_async_memory_resource on this device: a
  // private device memory pool that grows on demand and keeps freed memory.
  auto& pools = managed_gpu_memory_detail::process_gds_read_pools();
  {
    std::lock_guard const lock(pools.mutex);
    pools.by_device.try_emplace(device_id);
  }
  pools.any_device.store(true, std::memory_order_release);
  return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{std::move(managed_pool)};
}

/**
 * @brief Allocator for buffers that an I/O backend fills directly, such as KvikIO through
 * GPUDirect Storage, on the current device.
 *
 * cuFile cannot DMA into CUDA managed memory, so when the current device's GPU memory space is
 * managed memory this is the device's plain device memory pool.  Otherwise it is
 * @p space_allocator, the GPU memory space's own allocator, as without the option.
 */
inline rmm::device_async_resource_ref gds_read_buffer_resource(
  rmm::device_async_resource_ref space_allocator)
{
  auto& pools = managed_gpu_memory_detail::process_gds_read_pools();
  if (!pools.any_device.load(std::memory_order_acquire)) { return space_allocator; }
  std::lock_guard const lock(pools.mutex);
  auto const found = pools.by_device.find(rmm::get_current_cuda_device().value());
  if (found == pools.by_device.end()) { return space_allocator; }
  return rmm::device_async_resource_ref{found->second};
}

}  // namespace sirius::memory
