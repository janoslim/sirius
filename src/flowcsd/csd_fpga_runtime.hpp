/*
 * Host runtime of the eight mango SmartSSD FPGAs running the five-stage page core scan engine
 * (`fsc_engine_build.xclbin`: engine_feeder -> fsc_engine_kernel -> two engine_drain units), Key Idea 1.
 *
 * Purpose:
 *   Scan one Parquet file on the SmartSSD that stores it. The host reads the needed column chunks from the card's own
 *   NVMe SSD straight into the card's DRAM (NVMe-to-FPGA peer-to-peer O_DIRECT reads into an XRT p2p buffer), runs
 *   the scan engine there (decode, filter, project), copies the engine's output (main stream with the output regions
 *   and the region index, chars stream with string bytes) into pinned host memory, and returns it with the parsed
 *   index. Every transfer is host-initiated.
 *
 * Output format: `gpu-smartssd-olap/fpga-offloading/five-stage-page-core/rtl/synth/fsc_engine_kernel.sv` (file head).
 * Card map: mount /mnt/dpu-olap/lake8/d<i> is served by the FPGA user function in `kCardBdfs[i]` (mango, after the
 *   2026-11-04 PCIe re-enumeration); `mango_gate_8cards.sh` checks this map before every campaign.
 * Threads: `scan()` is thread-safe. Each card has `slots_per_card` buffer sets; input reads and output copies of
 *   different slots overlap, kernel runs on one card are serialized (one compute-unit set per card).
 * Errors: every failure (unknown drive, buffer too small, kernel error code, timeout) throws std::runtime_error.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sirius::flowcsd::csd {

/// Physical types as the scan engine's job header encodes them.
enum class PhysicalType : std::uint32_t { INT32 = 0, INT64 = 1, BYTE_ARRAY = 2 };

/// Inclusive 64-bit interval; a predicate keeps a row if its value falls in any interval of the chunk.
struct Interval {
  std::int64_t low  = 0;
  std::int64_t high = 0;
};

/// One column chunk of one row group, as a scan job.
struct ChunkJob {
  std::uint32_t chunk_id = 0;  ///< unique within the file request; index records name it
  std::uint32_t row_group = 0;
  std::uint64_t rows = 0;
  PhysicalType type = PhysicalType::INT32;
  bool snappy = false;
  std::uint32_t max_definition_level = 0;
  std::uint64_t file_offset = 0;  ///< first byte of the chunk (its dictionary page if it has one)
  std::uint64_t bytes = 0;        ///< total compressed size of the chunk
  bool predicate = false;         ///< true: evaluate `intervals` into the row-group bit-vector
  std::vector<Interval> intervals;
};

/// One row group: predicate chunks first, then projected chunks (the engine applies the bit-vector to projections).
struct RowGroupJob {
  std::uint32_t row_group = 0;
  std::uint64_t rows = 0;
  std::vector<ChunkJob> chunks;
};

struct FileScanRequest {
  std::string path;  ///< Parquet file under /mnt/dpu-olap/lake8/d<i>/
  std::vector<RowGroupJob> row_groups;
};

enum RegionKind : std::uint32_t {
  REGION_FIXED     = 1,  ///< surviving values, `width` bytes each
  REGION_DICT      = 2,  ///< `dict_entries` entries (`width` bytes; 0 = string entries) then 4 B codes
  REGION_STRING    = 3,  ///< 4 B cumulative end offsets in main, bytes in chars
  REGION_BITVECTOR = 4,  ///< row-group bit-vector, ceil(elements / 8) bytes
};

enum RegionFlag : std::uint32_t { FLAG_PLAIN_VALUES = 1, FLAG_DICT_CODES = 2, FLAG_DICTIONARY = 4 };

/// One region of the output, as the engine's 64-byte index record describes it.
struct IndexRecord {
  std::uint32_t kind = 0, width = 0, chunk_id = 0, row_group = 0, elements = 0, dict_entries = 0;
  std::uint32_t selected = 0, flags = 0;
  std::uint64_t main_offset = 0, main_bytes = 0, chars_offset = 0, chars_bytes = 0;
};

/// Counters of the engine's statistics beat.
struct ScanStats {
  std::uint64_t engine_cycles = 0, selected_rows = 0, data_output_bytes = 0, dram_output_bytes = 0;
  std::uint64_t records = 0, string_beats = 0, main_data_beats = 0, chars_data_beats = 0;
  std::uint32_t index_records = 0, error_code = 0;
};

/// Host wall times of the three steps of one file scan (CLOCK_MONOTONIC, nanoseconds) and the bytes moved.
struct ScanTiming {
  std::uint64_t ssd_to_fpga_ns = 0, fpga_kernel_ns = 0, fpga_to_host_ns = 0;
  std::uint64_t input_bytes = 0;   ///< bytes read from the SSD into card DRAM (4 KiB-aligned extents)
  std::uint64_t output_bytes = 0;  ///< bytes copied from card DRAM to host memory
};

class Runtime;

/// Output of one file scan in pinned host memory. Keeps its card buffer set until destroyed.
class FileScanResult {
 public:
  ~FileScanResult();
  FileScanResult(FileScanResult const&)            = delete;
  FileScanResult& operator=(FileScanResult const&) = delete;

  [[nodiscard]] std::uint8_t const* main() const noexcept { return main_; }    ///< main data beats
  [[nodiscard]] std::uint8_t const* chars() const noexcept { return chars_; }  ///< chars data beats
  [[nodiscard]] std::vector<IndexRecord> const& index() const noexcept { return index_; }
  [[nodiscard]] ScanStats const& stats() const noexcept { return stats_; }
  [[nodiscard]] ScanTiming const& timing() const noexcept { return timing_; }
  [[nodiscard]] int card() const noexcept { return card_; }

 private:
  friend class Runtime;
  FileScanResult() = default;
  Runtime* runtime_ = nullptr;
  int card_ = -1;
  int slot_ = -1;
  std::uint8_t const* main_  = nullptr;
  std::uint8_t const* chars_ = nullptr;
  std::vector<IndexRecord> index_;
  ScanStats stats_{};
  ScanTiming timing_{};
};

class Runtime {
 public:
  struct Options {
    std::string xclbin_path;                         ///< the engine bitstream (already programmed or loaded here)
    unsigned slots_per_card       = 4;
    // Sizes cover every SF100 Q1-Q22 file scan: the largest input is 143,269,888 B (customer) and the largest chars
    // output 130,509,632 B (q10 customer); 128 MiB input and 64 MiB chars slots failed 81 of 7,709 file scans in the
    // 2026-11-11 board check. Per card: 4 x (256 + 128 + 256) MiB = 2.5 GiB of the 4 GiB DRAM (P2P BAR is 8 GiB).
    std::uint64_t input_slot_bytes = 256ULL << 20;   ///< card DRAM for one file's chunk extents
    std::uint64_t main_slot_bytes  = 128ULL << 20;   ///< card DRAM and pinned host memory, power of two
    std::uint64_t chars_slot_bytes = 256ULL << 20;   ///< card DRAM and pinned host memory, power of two
    /// Called once per pinned host buffer (for example cudaHostRegister) before XRT wraps it.
    std::function<void(void*, std::size_t)> register_host_memory;
  };

  /// Open all eight cards, load the bitstream (a no-op when it is already programmed), and allocate the buffers.
  static Runtime& open(Options const& options);
  /// The runtime opened by `open`; throws if `open` has not run.
  static Runtime& get();

  /// Card index (0..7) of a file under /mnt/dpu-olap/lake8/d<i>/; throws for any other path.
  [[nodiscard]] static int card_of(std::string const& path);

  /// Scan one file on its card and return the output in pinned host memory.
  [[nodiscard]] std::unique_ptr<FileScanResult> scan(FileScanRequest const& request);

  ~Runtime();

 private:
  friend class FileScanResult;
  struct Impl;
  explicit Runtime(Options const& options);
  void release(int card, int slot);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sirius::flowcsd::csd
