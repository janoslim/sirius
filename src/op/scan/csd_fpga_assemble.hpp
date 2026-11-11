/*
 * GPU side of the SmartSSD scan: turn the scan engine's output regions (already copied to device memory) into cuDF
 * column buffers. Region and tile layout: fsc_engine_kernel.sv (file head) in gpu-smartssd-olap.
 *
 *   Key Idea 1 (plain regions): one kernel launch per split copies values, looks dictionary codes up, and writes
 *     string offsets (`launch_csd_region_copies`).
 *   Key Idea 2 (compact regions): a region is a run of compact tiles of up to 1,024 rows (DICT_BP, VALUE_FOR,
 *     DELTA_FOR, LEN_FSST). Tiles have different sizes, so `csd_tile_decoder::locate_tiles` first walks each region's
 *     tile headers (one thread per region); then one block per tile unpacks it straight into the output column
 *     (`decode_value_tiles`, `decode_string_tiles`). The host never reads a tile.
 */
#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

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

constexpr std::uint64_t CSD_TILE_ROWS = 1024;  ///< rows of every compact tile except a region's last

/// One compact region (Key Idea 2) and where its rows land in the split's output column.
struct csd_tile_region {
  std::uint8_t const* tiles  = nullptr;  ///< device: the region's first tile (after its dictionary entries)
  std::uint64_t tile_bytes   = 0;        ///< bytes of all tiles of the region
  std::uint8_t const* dict   = nullptr;  ///< device: dictionary entries (dictionary regions)
  std::uint64_t dict_entries = 0;
  std::uint32_t kind         = 0;        ///< 1 fixed-width values, 2 dictionary codes, 3 strings
  std::uint32_t width        = 0;        ///< fixed / dictionary: value width, 4 or 8 bytes
  std::uint64_t elements     = 0;        ///< rows of all tiles
  std::uint64_t first_tile   = 0;        ///< the region's first tile in the split's tile list
  void* dst                  = nullptr;  ///< fixed / dictionary: column data; string: int32 offsets column
  std::uint64_t dst_row      = 0;        ///< first output row of the region
  std::uint32_t string_column = 0;       ///< string regions: which `csd_string_column` the tiles write to
};

/// One tile of the split's tile list, as `locate_tiles` finds it. `src` is null if its region's tiles are malformed.
struct csd_tile_ref {
  std::uint8_t const* src      = nullptr;  ///< device: the tile's first byte
  std::uint32_t region         = 0;
  std::uint32_t tile_in_region = 0;
};

/// Where the string tiles of one output column write their bytes.
struct csd_string_column {
  std::uint8_t* chars            = nullptr;  ///< device: the column's chars buffer
  std::uint64_t first_tile_start = 0;        ///< `string_tile_starts` value of the column's first tile
};

/// Decodes the compact tiles of one split. All tiles of the split form one list: first the tiles of the fixed-width
/// and dictionary regions, then the string tiles, column by column (`csd_tile_region::first_tile` in that order).
class csd_tile_decoder {
 public:
  /// Copy the static FSST symbol table to the device once per process (start-up work).
  static void initialize();

  /// @p regions lists the value regions first, then the string regions grouped by output column.
  csd_tile_decoder(csd_tile_region const* regions,
                   std::size_t region_count,
                   std::size_t value_tiles,
                   std::size_t string_tiles,
                   ::cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr);

  /// Walk every region's tile headers and record where each tile starts; malformed tiles raise the error code.
  void locate_tiles();
  /// Unpack every fixed-width and dictionary tile into its output column.
  void decode_value_tiles();
  /// Sum the decoded string bytes of every string tile and return, for tile indices @p boundaries (positions in the
  /// string part of the tile list, each in [0, string tiles]), the decoded bytes before that tile. Synchronizes.
  [[nodiscard]] std::vector<std::uint64_t> string_bytes_before(std::vector<std::size_t> const& boundaries);
  /// Unpack every string tile: offsets into the region's offsets column, bytes into @p columns[string_column].
  void decode_string_tiles(csd_string_column const* columns, std::size_t column_count);
  /// Synchronize and throw if any kernel met a malformed tile.
  void finish();

 private:
  ::cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  std::size_t region_count_, value_tiles_, string_tiles_;
  rmm::device_uvector<csd_tile_region> regions_;
  rmm::device_uvector<csd_tile_ref> tiles_;
  rmm::device_uvector<std::uint64_t> string_tile_starts_;  ///< decoded bytes before each string tile (+ the total)
  rmm::device_uvector<std::uint32_t> error_;               ///< one word: 0, or the first error code a kernel met
};

}  // namespace sirius::op::scan
