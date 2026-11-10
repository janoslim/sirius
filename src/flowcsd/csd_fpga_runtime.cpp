/*
 * Host runtime of the eight mango SmartSSD FPGAs running the five-stage page core scan engine (Key Idea 1).
 * Interface and contracts: csd_fpga_runtime.hpp.
 *
 * One file scan:
 *   1. build the job words (64 B headers + interval beats; layout in fsc_engine_kernel.sv) and the 4 KiB-aligned
 *      file extents holding the chunks; each chunk's byte address in the input buffer goes in header bits [255:192]
 *   2. take a free buffer set (slot) of the file's card
 *   3. "flowcsd::csd_ssd_to_fpga": O_DIRECT pread of every extent from the card's SSD into the slot's p2p buffer
 *      (NVMe -> card DRAM peer-to-peer), and copy the job words into card DRAM
 *   4. "flowcsd::csd_fpga_kernel": run both drains, then the feeder, and wait (one run per card at a time)
 *   5. "flowcsd::csd_fpga_to_host": copy the two summary pairs, check them, then copy the main stream (data beats and
 *      index records) and the chars data beats into the slot's pinned host memory and parse the index
 */
#include "flowcsd/csd_fpga_runtime.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <nvtx3/nvToolsExt.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>

#include <utility>

namespace sirius::flowcsd::csd {
namespace {

constexpr std::uint64_t SUMMARY_MAGIC   = 0xC0DEC0DE57E900ULL;
constexpr std::uint32_t INDEX_MAGIC     = 0x1DE7;
constexpr std::uint64_t BEAT_BYTES      = 64;
constexpr std::uint64_t FILE_ALIGNMENT  = 4096;
constexpr std::uint64_t MAX_PREAD_BYTES = 8ULL << 20;  // one O_DIRECT read request at most
constexpr std::size_t JOB_SLOT_BYTES    = 1ULL << 20;  // 16,384 job beats per file
constexpr std::size_t SUMMARY_BYTES     = 2 * BEAT_BYTES;
constexpr unsigned KERNEL_TIMEOUT_MS    = 60000;
constexpr std::size_t MAX_INTERVALS     = 32;
constexpr std::size_t NUM_CARDS         = 8;

// FPGA user functions of /mnt/dpu-olap/lake8/d0..d7 (serials S6FWNG0R600092, -286, -208, -094, -203, -202, -287, -292).
constexpr std::array<char const*, NUM_CARDS> kCardBdfs = {
  "0000:0d:00.1", "0000:1c:00.1", "0000:21:00.1", "0000:11:00.1",
  "0000:18:00.1", "0000:6d:00.1", "0000:76:00.1", "0000:71:00.1"};
constexpr char const* kLakePrefix = "/mnt/dpu-olap/lake8/d";

// Job kinds and argument positions of the engine bitstream.
constexpr std::uint32_t JOB_ROW_GROUP_START = 0, JOB_CHUNK = 1, JOB_ROW_GROUP_END = 2, JOB_SCAN_END = 3;
constexpr int FEEDER_JOBS = 0, FEEDER_CHUNKS = 1, FEEDER_JOB_COUNT = 2;
constexpr int DRAIN_OUTPUT = 1, DRAIN_SUMMARY = 2, DRAIN_MASK = 3;

using Beat = std::array<std::uint32_t, 16>;

[[noreturn]] void fail(std::string const& message) { throw std::runtime_error("[csd_fpga] " + message); }

std::uint64_t now_ns()
{
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return std::uint64_t(ts.tv_sec) * 1000000000ULL + std::uint64_t(ts.tv_nsec);
}

/// NVTX push/pop on the calling thread (global domain, the names the time attribution matches).
struct NvtxRange {
  explicit NvtxRange(char const* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(NvtxRange const&)            = delete;
  NvtxRange& operator=(NvtxRange const&) = delete;
};

void set_bits(Beat& beat, int lsb, int width, std::uint64_t value)
{
  for (int done = 0; done < width;) {
    int const position = lsb + done;
    int const shift    = position % 32;
    int const take     = std::min(32 - shift, width - done);
    auto const mask    = std::uint32_t(((std::uint64_t(1) << take) - 1) << shift);
    auto const part    = std::uint32_t((value >> done) << shift) & mask;
    beat[position / 32] = (beat[position / 32] & ~mask) | part;
    done += take;
  }
}

std::uint64_t get_u64(std::uint32_t const* words, int word)
{
  return std::uint64_t(words[word]) | (std::uint64_t(words[word + 1]) << 32);
}

std::uint64_t round_down(std::uint64_t v, std::uint64_t a) { return v / a * a; }
std::uint64_t round_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

/// A 4 KiB-aligned file range and where it lands in the slot's input buffer.
struct Extent {
  std::uint64_t file_start = 0, file_end = 0, buffer_offset = 0;
};

/// Merge the aligned ranges of all chunks (a column may appear as predicate and projection) and place them back to
/// back in the input buffer.
std::vector<Extent> plan_extents(FileScanRequest const& request)
{
  std::vector<Extent> ranges;
  for (auto const& rg : request.row_groups) {
    for (auto const& chunk : rg.chunks) {
      if (chunk.bytes == 0) { continue; }
      ranges.push_back({round_down(chunk.file_offset, FILE_ALIGNMENT),
                        round_up(chunk.file_offset + chunk.bytes, FILE_ALIGNMENT), 0});
    }
  }
  std::sort(ranges.begin(), ranges.end(),
            [](Extent const& a, Extent const& b) { return a.file_start < b.file_start; });
  std::vector<Extent> merged;
  for (auto const& range : ranges) {
    if (!merged.empty() && range.file_start <= merged.back().file_end) {
      merged.back().file_end = std::max(merged.back().file_end, range.file_end);
    } else {
      merged.push_back(range);
    }
  }
  std::uint64_t offset = 0;
  for (auto& extent : merged) {
    extent.buffer_offset = offset;
    offset += extent.file_end - extent.file_start;
  }
  return merged;
}

std::uint64_t buffer_address(std::vector<Extent> const& extents, std::uint64_t file_offset)
{
  for (auto const& extent : extents) {
    if (file_offset >= extent.file_start && file_offset < extent.file_end) {
      return extent.buffer_offset + (file_offset - extent.file_start);
    }
  }
  fail("chunk offset outside every extent");
}

/// Job words of one file (headers and interval beats) and the number of job headers the feeder walks.
std::vector<Beat> build_job_words(FileScanRequest const& request, std::vector<Extent> const& extents,
                                  unsigned& job_count)
{
  std::vector<Beat> words;
  job_count  = 0;
  bool first = true;
  auto header = [&](std::uint32_t kind, std::uint32_t row_group, std::uint64_t rows) {
    Beat beat{};
    set_bits(beat, 0, 2, kind);
    set_bits(beat, 2, 32, row_group);
    set_bits(beat, 34, 32, rows);
    set_bits(beat, 142, 1, 1);              // emit output beats (Mode 1)
    set_bits(beat, 143, 1, first ? 1 : 0);  // reset the statistics at the first job
    first = false;
    ++job_count;
    return beat;
  };
  for (auto const& rg : request.row_groups) {
    words.push_back(header(JOB_ROW_GROUP_START, rg.row_group, rg.rows));
    for (auto const& chunk : rg.chunks) {
      if (chunk.intervals.size() > MAX_INTERVALS) { fail("more than 32 intervals in " + request.path); }
      if (chunk.predicate == chunk.intervals.empty()) {
        fail("predicate chunks need intervals, projected chunks none: " + request.path);
      }
      Beat beat = header(JOB_CHUNK, rg.row_group, chunk.rows);
      set_bits(beat, 66, 32, chunk.chunk_id);
      set_bits(beat, 98, 32, chunk.bytes);
      set_bits(beat, 130, 2, std::uint32_t(chunk.type));
      set_bits(beat, 132, 1, chunk.snappy ? 1 : 0);
      set_bits(beat, 133, 2, chunk.max_definition_level);
      set_bits(beat, 135, 1, chunk.predicate ? 1 : 0);
      set_bits(beat, 136, 6, chunk.intervals.size());
      set_bits(beat, 192, 64, chunk.bytes ? buffer_address(extents, chunk.file_offset) : 0);
      words.push_back(beat);
      for (std::size_t first_iv = 0; first_iv < chunk.intervals.size(); first_iv += 4) {
        Beat iv{};
        for (std::size_t k = 0; k < 4 && first_iv + k < chunk.intervals.size(); ++k) {
          set_bits(iv, int(128 * k), 64, std::uint64_t(chunk.intervals[first_iv + k].low));
          set_bits(iv, int(128 * k + 64), 64, std::uint64_t(chunk.intervals[first_iv + k].high));
        }
        words.push_back(iv);
      }
    }
    words.push_back(header(JOB_ROW_GROUP_END, rg.row_group, rg.rows));
  }
  words.push_back(header(JOB_SCAN_END, 0, 0));
  return words;
}

IndexRecord parse_index_record(std::uint32_t const* w)
{
  IndexRecord r;
  if ((w[0] & 0xFFFF) != INDEX_MAGIC) { fail("index record without magic"); }
  r.kind         = (w[0] >> 16) & 0xFF;
  r.width        = w[0] >> 24;
  r.chunk_id     = w[1];
  r.row_group    = w[2];
  r.elements     = w[3];
  r.dict_entries = w[4];
  r.selected     = w[5];
  r.main_offset  = get_u64(w, 6);
  r.main_bytes   = get_u64(w, 8);
  r.chars_offset = get_u64(w, 10);
  r.chars_bytes  = get_u64(w, 12);
  r.flags        = w[14];
  return r;
}

/// Page-aligned host memory that is freed with `free`; move-only so a vector of slots never frees it twice.
struct HostBuffer {
  std::uint8_t* data = nullptr;
  HostBuffer() = default;
  HostBuffer(HostBuffer&& other) noexcept : data(std::exchange(other.data, nullptr)) {}
  HostBuffer& operator=(HostBuffer&&) = delete;
  HostBuffer(HostBuffer const&)       = delete;
  ~HostBuffer() { std::free(data); }
};

}  // namespace

struct Runtime::Impl {
  struct Slot {
    xrt::bo jobs, input, main_out, chars_out, main_summary, chars_summary;
    Beat* jobs_map              = nullptr;
    char* input_map             = nullptr;
    std::uint32_t* main_summary_map  = nullptr;
    std::uint32_t* chars_summary_map = nullptr;
    HostBuffer main_host, chars_host;
    xrt::run feeder_run, main_drain_run, chars_drain_run;
  };
  struct Card {
    xrt::device device;
    xrt::kernel feeder, main_drain, chars_drain;
    std::vector<Slot> slots;
    std::vector<bool> slot_busy;
    std::mutex slot_mutex;
    std::condition_variable slot_free;
    std::mutex kernel_mutex;  // one engine run at a time per card
  };
  Options options;
  std::array<Card, NUM_CARDS> cards;
};

namespace {
std::mutex g_open_mutex;
std::unique_ptr<Runtime> g_runtime;
}  // namespace

Runtime& Runtime::open(Options const& options)
{
  std::lock_guard const lock(g_open_mutex);
  if (g_runtime) { fail("runtime already open"); }
  g_runtime.reset(new Runtime(options));
  return *g_runtime;
}

Runtime& Runtime::get()
{
  std::lock_guard const lock(g_open_mutex);
  if (!g_runtime) { fail("runtime not open"); }
  return *g_runtime;
}

int Runtime::card_of(std::string const& path)
{
  auto const real = std::filesystem::canonical(path).string();
  std::string const prefix(kLakePrefix);
  if (real.compare(0, prefix.size(), prefix) != 0 || real.size() < prefix.size() + 2 ||
      real[prefix.size() + 1] != '/') {
    fail(real + " is not on a SmartSSD drive " + prefix + "<0..7>/");
  }
  int const card = real[prefix.size()] - '0';
  if (card < 0 || card >= int(NUM_CARDS)) { fail(real + " names no SmartSSD drive"); }
  return card;
}

// XRT 2.23 marks device.load_xclbin and kernel(device, uuid, name) deprecated in favour of hw_context; the SmartSSD
// runs in this project use this path.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
Runtime::Runtime(Options const& options) : impl_(std::make_unique<Impl>())
{
  impl_->options = options;
  auto const is_power_of_two = [](std::uint64_t v) { return v != 0 && (v & (v - 1)) == 0; };
  if (!is_power_of_two(options.main_slot_bytes) || !is_power_of_two(options.chars_slot_bytes)) {
    fail("output buffer sizes must be powers of two (the drain wraps word indices with a mask)");
  }
  if (options.slots_per_card == 0) { fail("slots_per_card must be positive"); }
  xrt::xclbin const image(options.xclbin_path);
  for (std::size_t c = 0; c < NUM_CARDS; ++c) {
    auto& card  = impl_->cards[c];
    card.device = xrt::device(std::string(kCardBdfs[c]));
    auto const uuid = card.device.load_xclbin(image);
    if (card.device.get_xclbin_uuid() != image.get_uuid()) {
      fail(std::string(kCardBdfs[c]) + " does not report the engine bitstream after loading it");
    }
    card.feeder      = xrt::kernel(card.device, uuid, "engine_feeder:{engine_feeder_1}");
    card.main_drain  = xrt::kernel(card.device, uuid, "engine_drain:{engine_drain_1}");
    card.chars_drain = xrt::kernel(card.device, uuid, "engine_drain:{engine_drain_2}");
    card.slots.resize(options.slots_per_card);
    card.slot_busy.assign(options.slots_per_card, false);
    for (auto& slot : card.slots) {
      slot.jobs  = xrt::bo(card.device, JOB_SLOT_BYTES, xrt::bo::flags::normal, card.feeder.group_id(FEEDER_JOBS));
      slot.input = xrt::bo(card.device, options.input_slot_bytes, xrt::bo::flags::p2p,
                           card.feeder.group_id(FEEDER_CHUNKS));
      slot.jobs_map  = slot.jobs.map<Beat*>();
      slot.input_map = slot.input.map<char*>();
      // Output memory: page-aligned host buffers, registered by the caller (pinned), wrapped as userptr BOs so the
      // card-to-host copy writes straight into them.
      auto make_output = [&](HostBuffer& host, std::uint64_t bytes, xrt::kernel const& drain) {
        host.data = static_cast<std::uint8_t*>(std::aligned_alloc(FILE_ALIGNMENT, bytes));
        if (host.data == nullptr) { fail("aligned_alloc of output memory"); }
        if (options.register_host_memory) { options.register_host_memory(host.data, bytes); }
        return xrt::bo(card.device, host.data, bytes, xrt::bo::flags::normal, drain.group_id(DRAIN_OUTPUT));
      };
      slot.main_out  = make_output(slot.main_host, options.main_slot_bytes, card.main_drain);
      slot.chars_out = make_output(slot.chars_host, options.chars_slot_bytes, card.chars_drain);
      slot.main_summary  = xrt::bo(card.device, SUMMARY_BYTES, xrt::bo::flags::normal,
                                   card.main_drain.group_id(DRAIN_SUMMARY));
      slot.chars_summary = xrt::bo(card.device, SUMMARY_BYTES, xrt::bo::flags::normal,
                                   card.chars_drain.group_id(DRAIN_SUMMARY));
      slot.main_summary_map  = slot.main_summary.map<std::uint32_t*>();
      slot.chars_summary_map = slot.chars_summary.map<std::uint32_t*>();
      slot.feeder_run      = xrt::run(card.feeder);
      slot.main_drain_run  = xrt::run(card.main_drain);
      slot.chars_drain_run = xrt::run(card.chars_drain);
      slot.feeder_run.set_arg(FEEDER_JOBS, slot.jobs);
      slot.feeder_run.set_arg(FEEDER_CHUNKS, slot.input);
      slot.main_drain_run.set_arg(DRAIN_OUTPUT, slot.main_out);
      slot.main_drain_run.set_arg(DRAIN_SUMMARY, slot.main_summary);
      slot.main_drain_run.set_arg(DRAIN_MASK, unsigned(options.main_slot_bytes / BEAT_BYTES - 1));
      slot.chars_drain_run.set_arg(DRAIN_OUTPUT, slot.chars_out);
      slot.chars_drain_run.set_arg(DRAIN_SUMMARY, slot.chars_summary);
      slot.chars_drain_run.set_arg(DRAIN_MASK, unsigned(options.chars_slot_bytes / BEAT_BYTES - 1));
    }
  }
}
#pragma GCC diagnostic pop

Runtime::~Runtime() = default;

void Runtime::release(int card, int slot)
{
  auto& c = impl_->cards[std::size_t(card)];
  {
    std::lock_guard const lock(c.slot_mutex);
    c.slot_busy[std::size_t(slot)] = false;
  }
  c.slot_free.notify_one();
}

FileScanResult::~FileScanResult()
{
  if (runtime_ != nullptr) { runtime_->release(card_, slot_); }
}

std::unique_ptr<FileScanResult> Runtime::scan(FileScanRequest const& request)
{
  auto const& options = impl_->options;
  int const card_index = card_of(request.path);
  auto& card           = impl_->cards[std::size_t(card_index)];

  // 1. Job words and input extents.
  auto const extents = plan_extents(request);
  std::uint64_t const input_bytes = extents.empty() ? 0 : extents.back().buffer_offset +
                                                            (extents.back().file_end - extents.back().file_start);
  if (input_bytes > options.input_slot_bytes) {
    fail(request.path + " needs " + std::to_string(input_bytes) + " input bytes, the slot holds " +
         std::to_string(options.input_slot_bytes));
  }
  unsigned job_count = 0;
  auto const job_words = build_job_words(request, extents, job_count);
  if (job_words.size() * sizeof(Beat) > JOB_SLOT_BYTES) { fail("job words exceed the job buffer: " + request.path); }

  // 2. A free buffer set of the card.
  int slot_index = -1;
  {
    std::unique_lock lock(card.slot_mutex);
    card.slot_free.wait(lock, [&] {
      return std::find(card.slot_busy.begin(), card.slot_busy.end(), false) != card.slot_busy.end();
    });
    auto const it = std::find(card.slot_busy.begin(), card.slot_busy.end(), false);
    *it           = true;
    slot_index    = int(it - card.slot_busy.begin());
  }
  std::unique_ptr<FileScanResult> result(new FileScanResult());
  result->runtime_ = this;
  result->card_    = card_index;
  result->slot_    = slot_index;
  auto& slot       = card.slots[std::size_t(slot_index)];
  result->timing_.input_bytes = input_bytes;

  // 3. SSD -> card DRAM (peer-to-peer O_DIRECT reads), job words -> card DRAM.
  std::uint64_t const t0 = now_ns();
  {
    NvtxRange const range("flowcsd::csd_ssd_to_fpga");
    int const fd = ::open(request.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) { fail("open O_DIRECT " + request.path + ": " + std::strerror(errno)); }
    struct stat info {};
    if (fstat(fd, &info) != 0) {
      ::close(fd);
      fail("fstat " + request.path);
    }
    auto const file_bytes = std::uint64_t(info.st_size);
    for (auto const& extent : extents) {
      // The last extent may run past the end of the file; only the bytes up to the end must arrive.
      std::uint64_t const needed = std::min(extent.file_end, file_bytes) - extent.file_start;
      std::uint64_t done         = 0;
      while (done < needed) {
        std::uint64_t const ask = std::min(extent.file_end - extent.file_start - done, MAX_PREAD_BYTES);
        ssize_t const got = ::pread(fd, slot.input_map + extent.buffer_offset + done, ask,
                                    off_t(extent.file_start + done));
        if (got <= 0) {
          ::close(fd);
          fail("O_DIRECT pread of " + request.path + " into card " + std::to_string(card_index) +
               " returned " + std::to_string(got) + ": " + std::strerror(errno));
        }
        done += std::uint64_t(got);
      }
    }
    ::close(fd);
    std::memcpy(slot.jobs_map, job_words.data(), job_words.size() * sizeof(Beat));
    slot.jobs.sync(XCL_BO_SYNC_BO_TO_DEVICE, job_words.size() * sizeof(Beat), 0);
    // Arm the summaries so a kernel that never finishes cannot pass for one that did.
    std::memset(slot.main_summary_map, 0xFF, SUMMARY_BYTES);
    std::memset(slot.chars_summary_map, 0xFF, SUMMARY_BYTES);
    slot.main_summary.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    slot.chars_summary.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  std::uint64_t const t1 = now_ns();

  // 4. Engine run.
  {
    std::lock_guard const kernel_lock(card.kernel_mutex);
    NvtxRange const range("flowcsd::csd_fpga_kernel");
    slot.feeder_run.set_arg(FEEDER_JOB_COUNT, job_count);
    slot.main_drain_run.start();
    slot.chars_drain_run.start();
    slot.feeder_run.start();
    auto const timeout = std::chrono::milliseconds(KERNEL_TIMEOUT_MS);
    if (slot.feeder_run.wait(timeout) != ERT_CMD_STATE_COMPLETED ||
        slot.main_drain_run.wait(timeout) != ERT_CMD_STATE_COMPLETED ||
        slot.chars_drain_run.wait(timeout) != ERT_CMD_STATE_COMPLETED) {
      fail("engine run did not complete on card " + std::to_string(card_index) + " for " + request.path);
    }
  }
  std::uint64_t const t2 = now_ns();

  // 5. Card DRAM -> pinned host memory, statistics and index.
  {
    NvtxRange const range("flowcsd::csd_fpga_to_host");
    slot.main_summary.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    slot.chars_summary.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::uint32_t const* stats_beat = slot.main_summary_map + 16;
    std::uint32_t const* chars_end  = slot.chars_summary_map + 16;
    auto& s             = result->stats_;
    s.engine_cycles     = get_u64(stats_beat, 0);
    s.selected_rows     = get_u64(stats_beat, 2);
    s.data_output_bytes = get_u64(stats_beat, 4);
    s.dram_output_bytes = get_u64(stats_beat, 6);
    std::uint64_t const records_word = get_u64(stats_beat, 8);
    s.records           = records_word & 0xFFFFFFFFFFFFULL;
    s.index_records     = std::uint32_t((records_word >> 48) & 0x1FF);
    s.string_beats      = get_u64(stats_beat, 10);
    s.main_data_beats   = stats_beat[12];
    s.chars_data_beats  = stats_beat[13];
    std::uint64_t const top = get_u64(stats_beat, 14);
    s.error_code        = std::uint32_t(top & 0xFF);
    if ((top >> 8) != SUMMARY_MAGIC || (get_u64(chars_end, 14) >> 8) != SUMMARY_MAGIC) {
      fail("missing summary magic on card " + std::to_string(card_index) + " for " + request.path);
    }
    if (s.error_code != 0) {
      fail("engine error code " + std::to_string(s.error_code) + " on card " + std::to_string(card_index) +
           " for " + request.path);
    }
    if (slot.chars_summary_map[0] != s.chars_data_beats) { fail("chars stream beat count mismatch: " + request.path); }
    std::uint64_t const main_bytes  = (s.main_data_beats + s.index_records) * BEAT_BYTES;
    std::uint64_t const chars_bytes = s.chars_data_beats * BEAT_BYTES;
    if (main_bytes > options.main_slot_bytes || chars_bytes > options.chars_slot_bytes) {
      fail("engine output of " + request.path + " exceeds the output buffers (main " + std::to_string(main_bytes) +
           " B, chars " + std::to_string(chars_bytes) + " B); the drain wrapped and overwrote it");
    }
    if (main_bytes) { slot.main_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, main_bytes, 0); }
    if (chars_bytes) { slot.chars_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, chars_bytes, 0); }
    result->timing_.output_bytes = main_bytes + chars_bytes;
    result->main_  = slot.main_host.data;
    result->chars_ = slot.chars_host.data;
    result->index_.reserve(s.index_records);
    for (std::uint32_t i = 0; i < s.index_records; ++i) {
      auto const* words = reinterpret_cast<std::uint32_t const*>(slot.main_host.data +
                                                                  (s.main_data_beats + i) * BEAT_BYTES);
      result->index_.push_back(parse_index_record(words));
    }
  }
  std::uint64_t const t3 = now_ns();
  result->timing_.ssd_to_fpga_ns  = t1 - t0;
  result->timing_.fpga_kernel_ns  = t2 - t1;
  result->timing_.fpga_to_host_ns = t3 - t2;
  return result;
}

}  // namespace sirius::flowcsd::csd
