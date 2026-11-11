/*
 * GPU kernels that turn the SmartSSD scan engine's output into cuDF column buffers (csd_fpga_assemble.hpp).
 *
 * Key Idea 1: one kernel for all output regions of a split: blockIdx.y picks the region, blockIdx.x and threadIdx.x
 * stride over its rows. Region starts in the staged streams are 64-byte aligned, so 4- and 8-byte loads are aligned.
 *
 * Key Idea 2 (compact tiles; layout: `CompactTileTranscoder::build_tile` in gpu-smartssd-olap
 * fpga-offloading/five-stage-page-core/model/compact_tile_transcoder.cpp):
 *   word0: [1:0] mode (0 DICT_BP, 1 VALUE_FOR, 2 DELTA_FOR, 3 LEN_FSST), [2] 8-byte values (LEN_FSST: raw string
 *          bytes), [9:3] bit width, [20:10] rows, [31:21] LEN_FSST minimum length bits [10:0]
 *   DICT_BP  : word0, packed codes
 *   VALUE_FOR: word0, base (4 or 8 B), packed (value - base)
 *   DELTA_FOR: word0, first value, minimum delta (both 4 or 8 B; no minimum delta for a 1-row tile), packed
 *              (delta - minimum delta), row 0 packed as 0
 *   LEN_FSST : word0, word1 ([23:0] string payload bytes, [31:24] minimum length bits [18:11]), packed
 *              (length - minimum length), payload (FSST codes or raw bytes) padded to 4 B
 *   "packed": value i sits at bit i * width of a little-endian bit stream, padded to 4 B.
 * Every tile starts on a 4-byte boundary but not on an 8-byte one, so tiles are read as 32-bit words.
 * One block of 256 threads decodes one tile; each thread handles 4 of its 1,024 rows.
 */
#include "op/scan/csd_fpga_assemble.hpp"

#include "flowcsd/csd_fsst_static_table.hpp"

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <algorithm>
#include <array>
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

// ── Key Idea 2: compact tiles ────────────────────────────────────────────────────────────────────
constexpr unsigned TILE_THREADS         = 256;
constexpr unsigned ROWS_PER_THREAD      = CSD_TILE_ROWS / TILE_THREADS;
constexpr std::uint32_t MODE_DICT_BP    = 0, MODE_VALUE_FOR = 1, MODE_DELTA_FOR = 2, MODE_LEN_FSST = 3;
constexpr std::uint32_t FSST_SYMBOL_CODES = 256;  // table size on the device; code 255 is the escape
// First error a kernel met (0 = none), reported by `csd_tile_decoder::finish`.
constexpr std::uint32_t ERROR_TILE_HEADER = 1;   // invalid header, wrong mode or row count, or tile past its region
constexpr std::uint32_t ERROR_REGION_BYTES = 2;  // the tiles do not fill their region exactly
constexpr std::uint32_t ERROR_DICT_CODE    = 3;  // dictionary code beyond the dictionary
constexpr std::uint32_t ERROR_STRING_BYTES = 4;  // string payload decodes to a different size than the lengths say

struct tile_header {
  std::uint32_t mode;
  std::uint32_t wide;  ///< header bit 2: 8-byte values (VALUE_FOR, DELTA_FOR) or raw string bytes (LEN_FSST)
  std::uint32_t bit_width, rows;
};

__device__ tile_header read_tile_header(std::uint32_t word0)
{
  return {word0 & 0x3u, (word0 >> 2) & 0x1u, (word0 >> 3) & 0x7Fu, (word0 >> 10) & 0x7FFu};
}

/// Bytes of `rows` packed values of `bit_width` bits.
__device__ std::uint64_t packed_bytes(std::uint32_t rows, std::uint32_t bit_width)
{
  return (std::uint64_t(rows) * bit_width + 31) / 32 * 4;
}

/// Value `index` of a packed stream that starts at `words`. A value of up to 64 bits spans at most three words.
__device__ std::uint64_t unpack_bits(std::uint32_t const* words, std::uint32_t index, std::uint32_t bit_width)
{
  if (bit_width == 0) { return 0; }
  std::uint64_t const bit   = std::uint64_t(index) * bit_width;
  std::uint32_t const word  = std::uint32_t(bit >> 5);
  std::uint32_t const shift = std::uint32_t(bit & 31);
  std::uint64_t value       = std::uint64_t(words[word]) >> shift;
  std::uint32_t have        = 32 - shift;
  if (have < bit_width) {
    value |= std::uint64_t(words[word + 1]) << have;
    have += 32;
  }
  if (have < bit_width) { value |= std::uint64_t(words[word + 2]) << have; }
  return bit_width == 64 ? value : value & ((std::uint64_t(1) << bit_width) - 1);
}

/// A tile's base value: 8 bytes, or 4 bytes sign-extended (the transcoder's arithmetic is 64-bit).
__device__ std::uint64_t load_base(std::uint32_t const* words, bool wide)
{
  if (wide) { return std::uint64_t(words[0]) | (std::uint64_t(words[1]) << 32); }
  return std::uint64_t(std::int64_t(std::int32_t(words[0])));
}

__device__ void store_value(csd_tile_region const& region, std::uint64_t row, std::uint64_t value)
{
  if (region.width == 8) {
    static_cast<std::uint64_t*>(region.dst)[row] = value;
  } else {
    static_cast<std::uint32_t*>(region.dst)[row] = std::uint32_t(value);
  }
}

/// Inclusive prefix sums of the 1,024 values of a tile in shared memory (Hillis-Steele; needs TILE_THREADS threads).
template <typename T>
__device__ void tile_prefix_sum(T* values)
{
  for (unsigned distance = 1; distance < CSD_TILE_ROWS; distance <<= 1) {
    T add[ROWS_PER_THREAD];
    for (unsigned k = 0; k < ROWS_PER_THREAD; ++k) {
      unsigned const i = threadIdx.x + k * TILE_THREADS;
      add[k]           = i >= distance ? values[i - distance] : T(0);
    }
    __syncthreads();
    for (unsigned k = 0; k < ROWS_PER_THREAD; ++k) { values[threadIdx.x + k * TILE_THREADS] += add[k]; }
    __syncthreads();
  }
}

/// One thread per region walks its tile headers and records every tile's address. A tile's size follows from its
/// header alone. Tiles after a malformed one get a null address, so the decode kernels skip them.
__global__ void locate_tiles_kernel(csd_tile_region const* regions,
                                    std::size_t region_count,
                                    csd_tile_ref* tiles,
                                    std::uint32_t* error)
{
  std::size_t const index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= region_count) { return; }
  csd_tile_region const region  = regions[index];
  std::uint64_t const tile_count = (region.elements + CSD_TILE_ROWS - 1) / CSD_TILE_ROWS;
  std::uint64_t at               = 0;
  bool ok                        = true;
  for (std::uint64_t t = 0; t < tile_count; ++t) {
    csd_tile_ref ref{nullptr, std::uint32_t(index), std::uint32_t(t)};
    if (ok && at + 4 > region.tile_bytes) { ok = false; }
    if (ok) {
      auto const* words        = reinterpret_cast<std::uint32_t const*>(region.tiles + at);
      tile_header const header = read_tile_header(words[0]);
      std::uint64_t const rows = t + 1 < tile_count ? CSD_TILE_ROWS : region.elements - t * CSD_TILE_ROWS;
      bool const mode_fits =
        region.kind == KIND_DICT
          ? header.mode == MODE_DICT_BP
          : (region.kind == KIND_STRING
               ? header.mode == MODE_LEN_FSST
               : ((header.mode == MODE_VALUE_FOR || header.mode == MODE_DELTA_FOR) &&
                  header.wide == (region.width == 8 ? 1u : 0u)));
      std::uint64_t size             = 0;
      std::uint64_t const value_bytes = header.wide ? 8 : 4;
      std::uint64_t const packed      = packed_bytes(header.rows, header.bit_width);
      if (header.mode == MODE_DICT_BP) {
        size = 4 + packed;
      } else if (header.mode == MODE_VALUE_FOR) {
        size = 4 + value_bytes + packed;
      } else if (header.mode == MODE_DELTA_FOR) {
        size = 4 + (header.rows == 1 ? 1 : 2) * value_bytes + packed;
      } else if (at + 8 <= region.tile_bytes) {
        size = 8 + packed + (std::uint64_t(words[1] & 0xFFFFFFu) + 3) / 4 * 4;
      }
      ok = mode_fits && header.rows == rows && header.bit_width <= 64 && size != 0 &&
           at + size <= region.tile_bytes;
      if (ok) {
        ref.src = region.tiles + at;
        at += size;
      }
    }
    tiles[region.first_tile + t] = ref;
  }
  if (!ok) {
    atomicCAS(error, 0u, ERROR_TILE_HEADER);
  } else if (at != region.tile_bytes) {
    atomicCAS(error, 0u, ERROR_REGION_BYTES);
  }
}

/// One block per fixed-width or dictionary tile: unpack its rows into the output column.
__global__ void decode_value_tiles_kernel(csd_tile_ref const* tiles,
                                          csd_tile_region const* regions,
                                          std::uint32_t* error)
{
  __shared__ std::uint64_t s_sum[CSD_TILE_ROWS];  // DELTA_FOR: running sums of (delta - minimum delta)
  csd_tile_ref const ref = tiles[blockIdx.x];
  if (ref.src == nullptr) { return; }
  csd_tile_region const region = regions[ref.region];
  auto const* words            = reinterpret_cast<std::uint32_t const*>(ref.src);
  tile_header const header     = read_tile_header(words[0]);
  std::uint64_t const row0     = region.dst_row + std::uint64_t(ref.tile_in_region) * CSD_TILE_ROWS;
  std::uint32_t const value_words = header.wide ? 2 : 1;

  if (header.mode == MODE_DICT_BP) {
    for (std::uint32_t i = threadIdx.x; i < header.rows; i += TILE_THREADS) {
      std::uint64_t const code = unpack_bits(words + 1, i, header.bit_width);
      if (code >= region.dict_entries) {
        atomicCAS(error, 0u, ERROR_DICT_CODE);
        continue;
      }
      // Dictionary entries start at the region's 64-byte-aligned start, so 8-byte loads are aligned.
      std::uint64_t const value = region.width == 8
                                    ? reinterpret_cast<std::uint64_t const*>(region.dict)[code]
                                    : reinterpret_cast<std::uint32_t const*>(region.dict)[code];
      store_value(region, row0 + i, value);
    }
  } else if (header.mode == MODE_VALUE_FOR) {
    std::uint64_t const base  = load_base(words + 1, header.wide);
    std::uint32_t const* packed = words + 1 + value_words;
    for (std::uint32_t i = threadIdx.x; i < header.rows; i += TILE_THREADS) {
      store_value(region, row0 + i, base + unpack_bits(packed, i, header.bit_width));
    }
  } else {
    // value[i] = first + i * minimum delta + sum of the packed residuals of rows 1..i (64-bit wrap-around, as the
    // transcoder computes them).
    std::uint64_t const first = load_base(words + 1, header.wide);
    bool const has_deltas     = header.rows > 1;
    std::uint64_t const delta_min = has_deltas ? load_base(words + 1 + value_words, header.wide) : 0;
    std::uint32_t const* packed   = words + 1 + (has_deltas ? 2 : 1) * value_words;
    for (std::uint32_t i = threadIdx.x; i < CSD_TILE_ROWS; i += TILE_THREADS) {
      s_sum[i] = (i > 0 && i < header.rows) ? unpack_bits(packed, i, header.bit_width) : 0;
    }
    __syncthreads();
    tile_prefix_sum(s_sum);
    for (std::uint32_t i = threadIdx.x; i < header.rows; i += TILE_THREADS) {
      store_value(region, row0 + i, first + std::uint64_t(i) * delta_min + s_sum[i]);
    }
  }
}

/// Minimum string length of a LEN_FSST tile (19 bits split over its two header words).
__device__ std::uint32_t string_min_length(std::uint32_t word0, std::uint32_t word1)
{
  return ((word0 >> 21) & 0x7FFu) | (((word1 >> 24) & 0xFFu) << 11);
}

/// One block per string tile: the decoded bytes of its strings (the sum of their lengths).
__global__ void string_tile_bytes_kernel(csd_tile_ref const* tiles, std::uint64_t* tile_bytes)
{
  __shared__ std::uint64_t s_part[TILE_THREADS];
  csd_tile_ref const ref = tiles[blockIdx.x];
  std::uint64_t part     = 0;
  if (ref.src != nullptr) {
    auto const* words          = reinterpret_cast<std::uint32_t const*>(ref.src);
    tile_header const header   = read_tile_header(words[0]);
    std::uint32_t const minimum = string_min_length(words[0], words[1]);
    for (std::uint32_t i = threadIdx.x; i < header.rows; i += TILE_THREADS) {
      part += minimum + unpack_bits(words + 2, i, header.bit_width);
    }
  }
  s_part[threadIdx.x] = part;
  __syncthreads();
  for (unsigned half = TILE_THREADS / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) { s_part[threadIdx.x] += s_part[threadIdx.x + half]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { tile_bytes[blockIdx.x] = s_part[0]; }
}

/// Decode the FSST codes in[begin, end): a code is a symbol of 1..8 bytes, the escape code makes the next byte a
/// literal. `literal` says whether in[begin] is such a literal. Returns the decoded size; writes it to `out` unless
/// `out` is null.
__device__ std::uint32_t decode_fsst_piece(std::uint8_t const* in,
                                           std::uint32_t begin,
                                           std::uint32_t end,
                                           bool literal,
                                           std::uint64_t const* symbol_bytes,
                                           std::uint8_t const* symbol_lengths,
                                           std::uint8_t* out)
{
  std::uint32_t produced = 0;
  for (std::uint32_t p = begin; p < end; ++p) {
    std::uint8_t const code = in[p];
    if (literal) {
      if (out != nullptr) { out[produced] = code; }
      ++produced;
      literal = false;
    } else if (code == flowcsd::csd::FSST_ESCAPE_CODE) {
      literal = true;  // the literal may be the first byte of the next piece
    } else {
      std::uint32_t const length = symbol_lengths[code];
      if (out != nullptr) {
        std::uint64_t const bytes = symbol_bytes[code];
        for (std::uint32_t b = 0; b < length; ++b) { out[produced + b] = std::uint8_t(bytes >> (8 * b)); }
      }
      produced += length;
    }
  }
  return produced;
}

/// One block per string tile: string end offsets into the offsets column, string bytes into the chars buffer.
/// The strings of a tile are decoded back to back, so only the byte stream has to be decoded, not each string:
/// every thread decodes one piece of the FSST payload, first to size it, then (after the sizes of the pieces before
/// it are known) to write it.
__global__ void decode_string_tiles_kernel(csd_tile_ref const* tiles,
                                           csd_tile_region const* regions,
                                           std::uint64_t const* tile_starts,
                                           csd_string_column const* columns,
                                           std::uint64_t const* symbol_bytes,
                                           std::uint8_t const* symbol_lengths,
                                           std::uint32_t* error)
{
  __shared__ std::uint32_t s_end[CSD_TILE_ROWS];          // end offset of each string within the tile
  __shared__ std::uint32_t s_piece_bytes[TILE_THREADS];   // decoded size of each thread's payload piece
  csd_tile_ref const ref = tiles[blockIdx.x];
  if (ref.src == nullptr) { return; }
  csd_tile_region const region  = regions[ref.region];
  csd_string_column const column = columns[region.string_column];
  auto const* words              = reinterpret_cast<std::uint32_t const*>(ref.src);
  tile_header const header       = read_tile_header(words[0]);
  std::uint32_t const minimum    = string_min_length(words[0], words[1]);
  std::uint32_t const payload_bytes = words[1] & 0xFFFFFFu;

  // String end offsets: running sums of the lengths.
  for (std::uint32_t i = threadIdx.x; i < CSD_TILE_ROWS; i += TILE_THREADS) {
    s_end[i] = i < header.rows ? minimum + std::uint32_t(unpack_bits(words + 2, i, header.bit_width)) : 0;
  }
  __syncthreads();
  tile_prefix_sum(s_end);
  std::uint64_t const tile_start = tile_starts[blockIdx.x] - column.first_tile_start;  // column bytes before the tile
  auto* offsets = static_cast<std::int32_t*>(region.dst) + region.dst_row +
                  std::uint64_t(ref.tile_in_region) * CSD_TILE_ROWS + 1;
  for (std::uint32_t i = threadIdx.x; i < header.rows; i += TILE_THREADS) {
    offsets[i] = std::int32_t(tile_start + s_end[i]);
  }

  std::uint32_t const decoded_bytes = s_end[header.rows - 1];
  std::uint8_t const* payload       = ref.src + 8 + packed_bytes(header.rows, header.bit_width);
  std::uint8_t* chars               = column.chars + tile_start;
  bool const raw_bytes = header.wide != 0;
  if (raw_bytes) {
    // Raw string bytes (the transcoder keeps them when FSST would not shrink the tile).
    if (payload_bytes != decoded_bytes) {
      atomicCAS(error, 0u, ERROR_STRING_BYTES);
      return;
    }
    for (std::uint32_t p = threadIdx.x; p < payload_bytes; p += TILE_THREADS) { chars[p] = payload[p]; }
    return;
  }
  std::uint32_t const piece = (payload_bytes + TILE_THREADS - 1) / TILE_THREADS;
  std::uint32_t const begin = min(threadIdx.x * piece, payload_bytes);
  std::uint32_t const end   = min(begin + piece, payload_bytes);
  // The piece starts with a literal if an odd number of escape bytes ends right before it: the first escape byte of
  // a run is an escape code (the byte before it is not), the second its literal, and so on.
  bool literal = false;
  for (std::uint32_t p = begin; p > 0 && payload[p - 1] == flowcsd::csd::FSST_ESCAPE_CODE; --p) { literal = !literal; }
  s_piece_bytes[threadIdx.x] =
    decode_fsst_piece(payload, begin, end, literal, symbol_bytes, symbol_lengths, nullptr);
  __syncthreads();
  std::uint32_t before = 0;
  for (unsigned k = 0; k < threadIdx.x; ++k) { before += s_piece_bytes[k]; }
  std::uint32_t const after = before + s_piece_bytes[threadIdx.x];
  if (threadIdx.x == TILE_THREADS - 1 && after != decoded_bytes) { atomicCAS(error, 0u, ERROR_STRING_BYTES); }
  // A payload that decodes to more bytes than the lengths say must not write past the tile's bytes.
  if (after <= decoded_bytes) {
    decode_fsst_piece(payload, begin, end, literal, symbol_bytes, symbol_lengths, chars + before);
  }
}

void check_launch(char const* kernel)
{
  auto const status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("[csd_fpga] ") + kernel + " launch: " + cudaGetErrorString(status));
  }
}

void check_cuda_call(cudaError_t status, char const* what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("[csd_fpga] ") + what + ": " + cudaGetErrorString(status));
  }
}

// The static FSST table on the device (process lifetime): symbol bytes and lengths by code.
std::uint64_t* g_symbol_bytes  = nullptr;
std::uint8_t* g_symbol_lengths = nullptr;

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

void csd_tile_decoder::initialize()
{
  if (g_symbol_bytes != nullptr) { return; }
  std::array<std::uint64_t, FSST_SYMBOL_CODES> bytes{};
  std::array<std::uint8_t, FSST_SYMBOL_CODES> lengths{};  // length 0: no such code
  for (std::size_t code = 0; code < flowcsd::csd::kFsstStaticSymbols.size(); ++code) {
    bytes[code]   = flowcsd::csd::kFsstStaticSymbols[code].bytes;
    lengths[code] = flowcsd::csd::kFsstStaticSymbols[code].length;
  }
  check_cuda_call(cudaMalloc(&g_symbol_bytes, sizeof(bytes)), "cudaMalloc of the FSST symbols");
  check_cuda_call(cudaMalloc(&g_symbol_lengths, sizeof(lengths)), "cudaMalloc of the FSST symbol lengths");
  check_cuda_call(cudaMemcpy(g_symbol_bytes, bytes.data(), sizeof(bytes), cudaMemcpyHostToDevice), "FSST symbols copy");
  check_cuda_call(cudaMemcpy(g_symbol_lengths, lengths.data(), sizeof(lengths), cudaMemcpyHostToDevice),
                  "FSST symbol lengths copy");
}

csd_tile_decoder::csd_tile_decoder(csd_tile_region const* regions,
                                   std::size_t region_count,
                                   std::size_t value_tiles,
                                   std::size_t string_tiles,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
  : stream_(stream),
    mr_(mr),
    region_count_(region_count),
    value_tiles_(value_tiles),
    string_tiles_(string_tiles),
    regions_(region_count, stream, mr),
    tiles_(value_tiles + string_tiles, stream, mr),
    string_tile_starts_(string_tiles + 1, stream, mr),
    error_(1, stream, mr)
{
  check_cuda_call(cudaMemcpyAsync(regions_.data(), regions, region_count * sizeof(csd_tile_region),
                                  cudaMemcpyHostToDevice, stream_.get()),
                  "tile region descriptors copy");
  check_cuda_call(cudaMemsetAsync(error_.data(), 0, sizeof(std::uint32_t), stream_.get()), "tile error word");
}

void csd_tile_decoder::locate_tiles()
{
  if (region_count_ == 0) { return; }
  auto const blocks = static_cast<unsigned>((region_count_ + TILE_THREADS - 1) / TILE_THREADS);
  locate_tiles_kernel<<<blocks, TILE_THREADS, 0, stream_.get()>>>(regions_.data(), region_count_, tiles_.data(),
                                                                  error_.data());
  check_launch("locate_tiles_kernel");
}

void csd_tile_decoder::decode_value_tiles()
{
  if (value_tiles_ == 0) { return; }
  decode_value_tiles_kernel<<<static_cast<unsigned>(value_tiles_), TILE_THREADS, 0, stream_.get()>>>(
    tiles_.data(), regions_.data(), error_.data());
  check_launch("decode_value_tiles_kernel");
}

std::vector<std::uint64_t> csd_tile_decoder::string_bytes_before(std::vector<std::size_t> const& boundaries)
{
  // Decoded bytes of each string tile, a 0 after the last one, then running sums in place: element i becomes the
  // bytes before tile i and the last element the total.
  check_cuda_call(cudaMemsetAsync(string_tile_starts_.data() + string_tiles_, 0, sizeof(std::uint64_t), stream_.get()),
                  "string tile sizes");
  if (string_tiles_ > 0) {
    string_tile_bytes_kernel<<<static_cast<unsigned>(string_tiles_), TILE_THREADS, 0, stream_.get()>>>(
      tiles_.data() + value_tiles_, string_tile_starts_.data());
    check_launch("string_tile_bytes_kernel");
  }
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream_, mr_), string_tile_starts_.begin(), string_tile_starts_.end(),
                         string_tile_starts_.begin());
  std::vector<std::uint64_t> before(boundaries.size());
  for (std::size_t k = 0; k < boundaries.size(); ++k) {
    if (boundaries[k] > string_tiles_) { throw std::runtime_error("[csd_fpga] string tile boundary out of range"); }
    check_cuda_call(cudaMemcpyAsync(&before[k], string_tile_starts_.data() + boundaries[k], sizeof(std::uint64_t),
                                    cudaMemcpyDeviceToHost, stream_.get()),
                    "string tile boundary copy");
  }
  stream_.sync();
  return before;
}

void csd_tile_decoder::decode_string_tiles(csd_string_column const* columns, std::size_t column_count)
{
  if (string_tiles_ == 0) { return; }
  if (g_symbol_bytes == nullptr) { throw std::runtime_error("[csd_fpga] csd_tile_decoder::initialize has not run"); }
  rmm::device_uvector<csd_string_column> device_columns(column_count, stream_, mr_);
  check_cuda_call(cudaMemcpyAsync(device_columns.data(), columns, column_count * sizeof(csd_string_column),
                                  cudaMemcpyHostToDevice, stream_.get()),
                  "string column descriptors copy");
  decode_string_tiles_kernel<<<static_cast<unsigned>(string_tiles_), TILE_THREADS, 0, stream_.get()>>>(
    tiles_.data() + value_tiles_, regions_.data(), string_tile_starts_.data(), device_columns.data(), g_symbol_bytes,
    g_symbol_lengths, error_.data());
  check_launch("decode_string_tiles_kernel");
  stream_.sync();  // `device_columns` and the caller's host descriptors must outlive the kernel
}

void csd_tile_decoder::finish()
{
  std::uint32_t error = 0;
  check_cuda_call(cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream_.get()),
                  "tile error word copy");
  stream_.sync();
  if (error != 0) {
    throw std::runtime_error("[csd_fpga] malformed compact tile (error " + std::to_string(error) +
                             ": 1 tile header, 2 region size, 3 dictionary code, 4 string bytes)");
  }
}

}  // namespace sirius::op::scan
