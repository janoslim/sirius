/*
 * GPU side of the Key Idea 1 scan: turn the SmartSSD scan engine's output regions (already copied to device memory)
 * into cuDF column buffers in one kernel launch per split. Region layout: fsc_engine_kernel.sv (file head).
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sirius::op::scan {

/// One output region and where its rows land in the split's output column.
struct csd_region_copy {
  std::uint8_t const* src_main = nullptr;  ///< device: the region in the staged main stream
  std::uint32_t kind           = 0;        ///< 1 fixed-width values, 2 dictionary entries + codes, 3 string offsets
  std::uint32_t width          = 0;        ///< value width (fixed) or dictionary entry width (4 or 8)
  std::uint64_t elements       = 0;        ///< surviving rows
  std::uint64_t dict_entries   = 0;
  void* dst                    = nullptr;  ///< fixed / dictionary: column data; string: int32 offsets column
  std::uint64_t dst_row        = 0;        ///< first output row of the region
  std::int64_t dst_chars_offset = 0;       ///< string: first chars byte of the region in the output column
};

/// Fixed-width regions copy values, dictionary regions look codes up in their entries, string regions write
/// `dst_chars_offset + end offset` into offsets[dst_row + k + 1]. @p regions is in device memory.
void launch_csd_region_copies(csd_region_copy const* regions, std::size_t count, cudaStream_t stream);

}  // namespace sirius::op::scan
