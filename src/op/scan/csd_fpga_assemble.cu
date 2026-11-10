/*
 * One kernel for all output regions of a Key Idea 1 split (csd_fpga_assemble.hpp): blockIdx.y picks the region,
 * blockIdx.x and threadIdx.x stride over its rows. Region starts in the staged streams are 64-byte aligned, so 4- and
 * 8-byte loads are aligned.
 */
#include "op/scan/csd_fpga_assemble.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {
namespace {

constexpr unsigned THREADS_PER_BLOCK = 256;
constexpr unsigned BLOCKS_PER_REGION = 64;
constexpr std::size_t MAX_REGIONS_PER_LAUNCH = 65535;  // gridDim.y limit
constexpr std::uint32_t KIND_FIXED = 1, KIND_DICT = 2, KIND_STRING = 3;

template <typename T>
__device__ void copy_values(T const* src, T* dst, std::uint64_t count, std::uint64_t first, std::uint64_t step)
{
  for (std::uint64_t k = first; k < count; k += step) { dst[k] = src[k]; }
}

template <typename T>
__device__ void gather_values(T const* entries, std::uint32_t const* codes, T* dst, std::uint64_t count,
                              std::uint64_t first, std::uint64_t step)
{
  for (std::uint64_t k = first; k < count; k += step) { dst[k] = entries[codes[k]]; }
}

__global__ void region_copy_kernel(csd_region_copy const* regions)
{
  csd_region_copy const r   = regions[blockIdx.y];
  std::uint64_t const first = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint64_t const step  = std::uint64_t(gridDim.x) * blockDim.x;
  if (r.kind == KIND_FIXED) {
    if (r.width == 8) {
      copy_values(reinterpret_cast<std::uint64_t const*>(r.src_main),
                  static_cast<std::uint64_t*>(r.dst) + r.dst_row, r.elements, first, step);
    } else {
      copy_values(reinterpret_cast<std::uint32_t const*>(r.src_main),
                  static_cast<std::uint32_t*>(r.dst) + r.dst_row, r.elements, first, step);
    }
  } else if (r.kind == KIND_DICT) {
    auto const* codes = reinterpret_cast<std::uint32_t const*>(r.src_main + r.dict_entries * r.width);
    if (r.width == 8) {
      gather_values(reinterpret_cast<std::uint64_t const*>(r.src_main), codes,
                    static_cast<std::uint64_t*>(r.dst) + r.dst_row, r.elements, first, step);
    } else {
      gather_values(reinterpret_cast<std::uint32_t const*>(r.src_main), codes,
                    static_cast<std::uint32_t*>(r.dst) + r.dst_row, r.elements, first, step);
    }
  } else if (r.kind == KIND_STRING) {
    auto const* ends = reinterpret_cast<std::uint32_t const*>(r.src_main);
    auto* offsets    = static_cast<std::int32_t*>(r.dst) + r.dst_row + 1;
    for (std::uint64_t k = first; k < r.elements; k += step) {
      offsets[k] = static_cast<std::int32_t>(r.dst_chars_offset + std::int64_t(ends[k]));
    }
  }
}

}  // namespace

void launch_csd_region_copies(csd_region_copy const* regions, std::size_t count, cudaStream_t stream)
{
  for (std::size_t done = 0; done < count; done += MAX_REGIONS_PER_LAUNCH) {
    auto const batch = std::min(count - done, MAX_REGIONS_PER_LAUNCH);
    dim3 const grid(BLOCKS_PER_REGION, static_cast<unsigned>(batch));
    region_copy_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(regions + done);
    auto const status = cudaGetLastError();
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("[csd_fpga] region copy kernel launch: ") + cudaGetErrorString(status));
    }
  }
}

}  // namespace sirius::op::scan
