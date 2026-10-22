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

// tutti_service over libflowcsd_tutti_bridge (see tutti_service.hpp).
//
// Only builds that link the bridge (SIRIUS_TUTTI_BRIDGE_DIR -> SIRIUS_WITH_TUTTI)
// get the implementation; otherwise instance() throws.
//
// read(): every range is widened to NVMe blocks, widened ranges that touch are
// merged, and the merged ranges are cut into pieces of at most
// min(max_request_bytes, slab bytes).  Each round takes as many free staging
// slabs as it can (at least one), packs the next pieces into them, has the
// bridge submit one Tutti request per piece and wait for all of them, copies
// exactly the requested bytes slab -> destination on the caller's stream,
// synchronizes it and returns the slabs.  write() runs the same rounds the other
// way: source -> slab copies first, then one Tutti WRITE per piece.
//
// With the FlowCSD event trace on (flowcsd::trace_enabled), each round adds
// sub-intervals to the caller's trace: a wait for a slab, the bridge's per-submit
// stream wait / host submit / device (launch -> completion seen) phases, and the
// staging copies, named per transfer kind (io.*, spill_w.*, spill_r.*).

#include "io/tutti/tutti_service.hpp"

#include <stdexcept>

#if defined(SIRIUS_WITH_TUTTI)
#include "cuda/device_copy_batch.hpp"
#include "flowcsd/flowcsd_config.hpp"
#include "flowcsd/flowcsd_stats.hpp"
#include "log/logging.hpp"
#include "telemetry/nvtx.hpp"

#include <flowcsd_tutti_bridge.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda/stream>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#endif

namespace sirius::io::tutti {

#if !defined(SIRIUS_WITH_TUTTI)

tutti_service& tutti_service::instance()
{
  throw std::runtime_error(
    "Sirius built without Tutti: configure Sirius with SIRIUS_TUTTI_BRIDGE_DIR set");
}

bool tutti_service::configured() noexcept { return false; }

#else

namespace {

namespace bridge = ::flowcsd_tutti_bridge;

// Slab bases are 2 MiB (GPU large page) aligned, which also meets Tutti's
// 64 KiB requirement (snvme pins GPU memory in 64 KiB pages).
constexpr std::uint64_t kSlabAlignmentBytes = 2ull << 20;

std::uint64_t round_down(std::uint64_t value, std::uint64_t step) { return value / step * step; }

std::uint64_t round_up(std::uint64_t value, std::uint64_t step)
{
  return (value + step - 1) / step * step;
}

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("tutti_service: ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

// -- Trace names ------------------------------------------------------------

/// The trace sub-intervals of one kind of transfer.
struct transfer_trace {
  flowcsd::trace_category slab_wait;
  flowcsd::trace_category stream_wait;
  flowcsd::trace_category submit;
  flowcsd::trace_category device;
  flowcsd::trace_category copy;  ///< slab -> destination (reads) or source -> slab (writes)
  char const* kind;              ///< label of its batches in Tutti's raw I/O records
};

using flowcsd::trace_category;
constexpr transfer_trace kScanReadTrace{trace_category::io_slab_wait,
                                        trace_category::io_stream_wait,
                                        trace_category::io_submit,
                                        trace_category::io_device,
                                        trace_category::io_copy,
                                        "scan"};
constexpr transfer_trace kSpillReadTrace{trace_category::spill_r_slab_wait,
                                         trace_category::spill_r_stream_wait,
                                         trace_category::spill_r_submit,
                                         trace_category::spill_r_device,
                                         trace_category::spill_r_copy_from_slab,
                                         "spill_read"};
constexpr transfer_trace kSpillWriteTrace{trace_category::spill_w_slab_wait,
                                          trace_category::spill_w_stream_wait,
                                          trace_category::spill_w_submit,
                                          trace_category::spill_w_device,
                                          trace_category::spill_w_copy_to_slab,
                                          "spill_write"};

// -- Transfer planning ------------------------------------------------------

/// One Tutti request: a block-aligned file range.
struct piece {
  std::uint64_t file_offset{0};
  std::uint64_t bytes{0};
};

/// Bytes that move between a piece's slab copy and one of the caller's buffers.
struct piece_copy {
  std::size_t piece_index{0};
  std::uint64_t offset_in_piece{0};
  std::size_t transfer_index{0};  ///< index into the caller's reads / writes
  std::uint64_t offset_in_transfer{0};
  std::uint64_t bytes{0};
};

/// The pieces to submit and the copies that connect them to the caller's buffers.
struct transfer_plan {
  std::vector<piece> pieces;
  std::vector<piece_copy> copies;  ///< ordered by piece_index
  std::uint64_t requested_bytes{0};
  std::uint64_t issued_bytes{0};
};

/// Append the pieces of the block-aligned extent [extent_begin, extent_end).
void cut_into_pieces(std::vector<piece>& pieces,
                     std::uint64_t extent_begin,
                     std::uint64_t extent_end,
                     std::uint64_t piece_bytes)
{
  for (auto offset = extent_begin; offset < extent_end; offset += piece_bytes) {
    pieces.push_back(piece{offset, std::min(piece_bytes, extent_end - offset)});
  }
}

/// Append the copies that carry [file_offset, file_offset + bytes) of one
/// transfer through the pieces of an extent cut from extent_begin onward.
void add_copies(std::vector<piece_copy>& copies,
                std::size_t first_piece,
                std::uint64_t extent_begin,
                std::uint64_t piece_bytes,
                std::size_t transfer_index,
                std::uint64_t file_offset,
                std::uint64_t bytes)
{
  std::uint64_t done = 0;
  while (done < bytes) {
    auto const position_in_extent = file_offset + done - extent_begin;
    auto const offset_in_piece    = position_in_extent % piece_bytes;
    auto const chunk_bytes        = std::min(bytes - done, piece_bytes - offset_in_piece);
    copies.push_back(piece_copy{first_piece + position_in_extent / piece_bytes,
                                offset_in_piece,
                                transfer_index,
                                done,
                                chunk_bytes});
    done += chunk_bytes;
  }
}

std::uint64_t total_bytes(std::vector<piece> const& pieces)
{
  std::uint64_t bytes = 0;
  for (auto const& p : pieces) {
    bytes += p.bytes;
  }
  return bytes;
}

/// Widen each read to blocks, merge widened ranges that touch, cut the merged
/// ranges into pieces and map every read onto the pieces it overlaps.
transfer_plan plan_reads(std::span<device_read const> reads,
                         std::uint64_t block_bytes,
                         std::uint64_t piece_bytes)
{
  // Block-aligned range of every non-empty read, by start offset.
  struct widened_read {
    std::uint64_t begin;
    std::uint64_t end;
    std::size_t read_index;
  };
  transfer_plan plan;
  std::vector<widened_read> widened;
  widened.reserve(reads.size());
  for (std::size_t index = 0; index < reads.size(); ++index) {
    auto const& read = reads[index];
    plan.requested_bytes += read.bytes;
    if (read.bytes == 0) { continue; }
    widened.push_back(widened_read{round_down(read.file_offset, block_bytes),
                                   round_up(read.file_offset + read.bytes, block_bytes),
                                   index});
  }
  std::sort(widened.begin(), widened.end(), [](widened_read const& lhs, widened_read const& rhs) {
    return lhs.begin < rhs.begin;
  });

  std::size_t next = 0;
  while (next < widened.size()) {
    // Grow one extent over every following read that overlaps or touches it.
    auto const first_read   = next;
    auto const extent_begin = widened[next].begin;
    auto extent_end         = widened[next].end;
    while (next < widened.size() && widened[next].begin <= extent_end) {
      extent_end = std::max(extent_end, widened[next].end);
      ++next;
    }
    auto const first_piece = plan.pieces.size();
    cut_into_pieces(plan.pieces, extent_begin, extent_end, piece_bytes);
    for (auto index = first_read; index < next; ++index) {
      auto const& read = reads[widened[index].read_index];
      add_copies(plan.copies,
                 first_piece,
                 extent_begin,
                 piece_bytes,
                 widened[index].read_index,
                 read.file_offset,
                 read.bytes);
    }
  }
  // Rounds consume copies in piece order; a long read emits copies for late
  // pieces before a shorter read that starts after it.
  std::stable_sort(plan.copies.begin(),
                   plan.copies.end(),
                   [](piece_copy const& lhs, piece_copy const& rhs) {
                     return lhs.piece_index < rhs.piece_index;
                   });
  plan.issued_bytes = total_bytes(plan.pieces);
  return plan;
}

/// Cut each write into pieces.  The last piece of a write is rounded up to a
/// block; the slab bytes it carries past the data are padding by contract.
transfer_plan plan_writes(std::span<device_write const> writes,
                          std::uint64_t block_bytes,
                          std::uint64_t piece_bytes)
{
  transfer_plan plan;
  for (std::size_t index = 0; index < writes.size(); ++index) {
    auto const& write = writes[index];
    plan.requested_bytes += write.bytes;
    if (write.bytes == 0) { continue; }
    // Widening the start would overwrite the bytes in front of it.
    if (write.file_offset % block_bytes != 0) {
      throw std::invalid_argument("tutti_service: write offset " +
                                  std::to_string(write.file_offset) + " is not a multiple of " +
                                  std::to_string(block_bytes));
    }
    auto const first_piece = plan.pieces.size();
    cut_into_pieces(plan.pieces,
                    write.file_offset,
                    write.file_offset + round_up(write.bytes, block_bytes),
                    piece_bytes);
    add_copies(
      plan.copies, first_piece, write.file_offset, piece_bytes, index, write.file_offset, write.bytes);
  }
  plan.issued_bytes = total_bytes(plan.pieces);
  return plan;
}

// -- Staging slabs ----------------------------------------------------------

/// A registered GPU slab: the only memory Tutti moves data into or out of.
struct staging_slab {
  std::byte* address{nullptr};
  std::uint32_t memory_id{0};
};

/// A fixed set of registered staging slabs, handed out one at a time.
class slab_pool {
 public:
  slab_pool(bridge::runtime& runtime,
            std::string name,
            std::size_t slab_count,
            std::uint64_t slab_bytes);
  ~slab_pool() { unregister_and_free(); }

  slab_pool(slab_pool const&)            = delete;
  slab_pool& operator=(slab_pool const&) = delete;

  [[nodiscard]] std::uint64_t slab_bytes() const noexcept { return _slab_bytes; }

  /// Wait until a slab is free; an actual wait is traced as @p wait_category.
  [[nodiscard]] staging_slab acquire(trace_category wait_category);

  /// A free slab, or nothing when every slab is in use.
  [[nodiscard]] std::optional<staging_slab> try_acquire();

  void release(staging_slab slab);

 private:
  void unregister_and_free() noexcept;

  bridge::runtime& _runtime;
  std::string _name;
  std::uint64_t _slab_bytes;
  void* _allocation{nullptr};
  std::vector<staging_slab> _slabs;

  std::mutex _mutex;
  std::condition_variable _slab_released;
  std::vector<staging_slab> _free;
};

slab_pool::slab_pool(bridge::runtime& runtime,
                     std::string name,
                     std::size_t slab_count,
                     std::uint64_t slab_bytes)
  : _runtime(runtime), _name(std::move(name)), _slab_bytes(slab_bytes)
{
  if (slab_count == 0) { return; }
  // One allocation per pool, carved into aligned slabs, each registered once.
  auto const slab_stride = round_up(slab_bytes, kSlabAlignmentBytes);
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{runtime.accel_id()}};
  check_cuda(cudaMalloc(&_allocation, slab_count * slab_stride + kSlabAlignmentBytes),
             "cudaMalloc of the staging slabs");
  auto const base = round_up(reinterpret_cast<std::uintptr_t>(_allocation), kSlabAlignmentBytes);
  try {
    for (std::size_t index = 0; index < slab_count; ++index) {
      auto* const address = reinterpret_cast<std::byte*>(base + index * slab_stride);
      _slabs.push_back(staging_slab{address, runtime.register_device_memory(address, slab_bytes)});
    }
  } catch (...) {
    // The destructor does not run for a half-built pool.
    unregister_and_free();
    throw;
  }
  _free = _slabs;
}

void slab_pool::unregister_and_free() noexcept
{
  for (auto const& slab : _slabs) {
    try {
      _runtime.unregister_device_memory(slab.memory_id);
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("tutti_service: unregistering a {} slab failed: {}", _name, e.what());
    }
  }
  _slabs.clear();
  if (_allocation != nullptr) {
    (void)cudaFree(_allocation);  // teardown: nothing useful to do with a failure
    _allocation = nullptr;
  }
}

staging_slab slab_pool::acquire(trace_category wait_category)
{
  std::unique_lock lock{_mutex};
  if (_slabs.empty()) {
    throw std::runtime_error("tutti_service: no " + _name +
                             " staging slabs are configured (sirius.flowcsd)");
  }
  if (_free.empty()) {
    flowcsd::trace_scope const waiting{wait_category};
    _slab_released.wait(lock, [this] { return !_free.empty(); });
  }
  auto const slab = _free.back();
  _free.pop_back();
  return slab;
}

std::optional<staging_slab> slab_pool::try_acquire()
{
  std::lock_guard lock{_mutex};
  if (_free.empty()) { return std::nullopt; }
  auto const slab = _free.back();
  _free.pop_back();
  return slab;
}

void slab_pool::release(staging_slab slab)
{
  {
    std::lock_guard lock{_mutex};
    _free.push_back(slab);
  }
  _slab_released.notify_one();
}

/// Where a piece sits for the current round.
struct slab_spot {
  staging_slab slab;
  std::uint64_t offset{0};
};

/// Slabs held by one round; they go back to the pool when the round ends or fails.
class round_slabs {
 public:
  explicit round_slabs(slab_pool& pool) : _pool(pool) {}
  ~round_slabs()
  {
    for (auto const& slab : _held) {
      _pool.release(slab);
    }
  }

  round_slabs(round_slabs const&)            = delete;
  round_slabs& operator=(round_slabs const&) = delete;

  /// Pack the leading pieces into slabs and return the spot of each placed
  /// piece.  Only the first slab is waited for (traced as @p wait_category);
  /// later ones are taken only while free, so a round never holds slabs while
  /// waiting for more (no deadlock between callers).  Every piece fits a slab,
  /// so at least one is placed.
  std::vector<slab_spot> place(std::span<piece const> pieces, trace_category wait_category)
  {
    std::vector<slab_spot> spots;
    _held.push_back(_pool.acquire(wait_category));
    std::uint64_t used_bytes = 0;
    for (auto const& p : pieces) {
      if (used_bytes + p.bytes > _pool.slab_bytes()) {
        auto next = _pool.try_acquire();
        if (!next) { break; }
        _held.push_back(*next);
        used_bytes = 0;
      }
      spots.push_back(slab_spot{_held.back(), used_bytes});
      used_bytes += p.bytes;
    }
    return spots;
  }

 private:
  slab_pool& _pool;
  std::vector<staging_slab> _held;
};

/// One bridge request per placed piece.
std::vector<bridge::io_request> nvme_requests(bridge::io_direction direction,
                                              std::uint32_t file_id,
                                              std::span<piece const> pieces,
                                              std::span<slab_spot const> spots)
{
  std::vector<bridge::io_request> requests;
  requests.reserve(spots.size());
  for (std::size_t index = 0; index < spots.size(); ++index) {
    requests.push_back(bridge::io_request{direction,
                                          spots[index].slab.memory_id,
                                          spots[index].offset,
                                          file_id,
                                          pieces[index].file_offset,
                                          pieces[index].bytes});
  }
  return requests;
}

// -- The service ------------------------------------------------------------

/// Largest Tutti request: max_request_bytes in whole blocks, at most one slab.
std::uint64_t piece_limit(flowcsd::flowcsd_config const& config, std::uint64_t block_bytes)
{
  if (config.staging_slab_bytes == 0 || config.staging_slab_bytes % block_bytes != 0) {
    throw std::invalid_argument("sirius.flowcsd.staging_slab_bytes must be a positive multiple of " +
                                std::to_string(block_bytes));
  }
  auto const piece_bytes = round_down(
    std::min<std::uint64_t>(config.max_request_bytes, config.staging_slab_bytes), block_bytes);
  if (piece_bytes == 0) {
    throw std::invalid_argument("sirius.flowcsd.max_request_bytes must be at least " +
                                std::to_string(block_bytes));
  }
  return piece_bytes;
}

class bridge_tutti_service final : public tutti_service {
 public:
  explicit bridge_tutti_service(flowcsd::flowcsd_config const& config)
    : _bridge(config.tutti_runtime_config),
      _block_bytes(_bridge.block_bytes()),
      _piece_bytes(piece_limit(config, _block_bytes)),
      _scan_slabs(_bridge, "scan", config.read_staging_slabs, config.staging_slab_bytes),
      _spill_slabs(_bridge, "spill", config.spill_staging_slabs, config.staging_slab_bytes)
  {
    SIRIUS_LOG_INFO(
      "tutti_service: runtime {} on GPU {}; {} scan + {} spill slabs of {} B; requests <= {} B",
      config.tutti_runtime_config,
      _bridge.accel_id(),
      config.read_staging_slabs,
      config.spill_staging_slabs,
      config.staging_slab_bytes,
      _piece_bytes);
    // Traced queries also get Tutti's raw per-command records (TUTTI_IO_TRACE=1),
    // written next to the JSONL when the query ends.  The service lives for the
    // rest of the process, so the writer may keep a reference to it.
    if (flowcsd::trace_enabled() && bridge::runtime::io_trace_enabled()) {
      flowcsd::add_trace_writer(
        [this](std::string const& base, std::chrono::steady_clock::time_point query_start) {
          _bridge.write_io_trace(base,
                                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   query_start.time_since_epoch())
                                   .count());
        });
    }
  }

  [[nodiscard]] std::uint64_t block_bytes() const noexcept override { return _block_bytes; }

  [[nodiscard]] std::uint32_t open_file(std::string const& path) override
  {
    // Held across the bridge open so concurrent opens of one path resolve it once.
    std::lock_guard lock{_files_mutex};
    if (auto const known = _file_ids.find(path); known != _file_ids.end()) {
      ++_files.at(known->second).references;
      return known->second;
    }
    auto const opened = _bridge.open_file(path);
    _file_ids.emplace(path, opened.id);
    _files.emplace(opened.id,
                   open_file_entry{path, opened.size_bytes, 1, flowcsd::tutti_device_index(path)});
    return opened.id;
  }

  void close_file(std::uint32_t file_id) override
  {
    std::lock_guard lock{_files_mutex};
    auto const entry = _files.find(file_id);
    if (entry == _files.end()) {
      throw std::invalid_argument("tutti_service: file id " + std::to_string(file_id) +
                                  " is not open");
    }
    if (--entry->second.references > 0) { return; }
    _file_ids.erase(entry->second.path);
    _files.erase(entry);
    _bridge.close_file(file_id);
  }

  [[nodiscard]] std::uint64_t file_size(std::uint32_t file_id) const override
  {
    std::lock_guard lock{_files_mutex};
    auto const entry = _files.find(file_id);
    if (entry == _files.end()) {
      throw std::invalid_argument("tutti_service: file id " + std::to_string(file_id) +
                                  " is not open");
    }
    return entry->second.size_bytes;
  }

  transfer_stats read(std::uint32_t file_id,
                      std::span<device_read const> reads,
                      staging_pool pool,
                      cudaStream_t stream) override
  {
    // Only the spill staging pool carries SSD-spill provenance; scan reads remain unmarked.
    std::optional<nvtx_scoped_range> control_range;
    if (pool == staging_pool::spill) { control_range.emplace("gsb::spill::control"); }
    auto const plan = plan_reads(reads, _block_bytes, _piece_bytes);
    auto const& names = pool == staging_pool::scan ? kScanReadTrace : kSpillReadTrace;
    auto const dev    = flowcsd::trace_enabled() ? device_of(file_id) : -1;
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_bridge.accel_id()}};
    std::size_t next_piece = 0;
    std::size_t next_copy  = 0;
    while (next_piece < plan.pieces.size()) {
      round_slabs round{slabs_for(pool)};
      auto const round_pieces = std::span{plan.pieces}.subspan(next_piece);
      auto const spots        = round.place(round_pieces, names.slab_wait);
      auto const round_end    = next_piece + spots.size();

      // NVMe -> slabs.
      {
        std::optional<nvtx_scoped_range> read_range;
        if (pool == staging_pool::spill) { read_range.emplace("gsb::spill::read"); }
        submit_and_wait(nvme_requests(bridge::io_direction::read,
                                      file_id,
                                      round_pieces.first(spots.size()),
                                      spots),
                        names,
                        dev);
      }

      // Slabs -> destinations: only the requested bytes, not the block padding.
      flowcsd::trace_scope copy_trace{names.copy};
      std::optional<nvtx_scoped_range> restore_range;
      if (pool == staging_pool::spill) { restore_range.emplace("gsb::spill::restore"); }
      sirius::cuda::device_copy_batch copies;
      std::uint64_t copied_bytes = 0;
      for (; next_copy < plan.copies.size() && plan.copies[next_copy].piece_index < round_end;
           ++next_copy) {
        auto const& copy = plan.copies[next_copy];
        auto const& spot = spots[copy.piece_index - next_piece];
        copies.add(static_cast<std::byte*>(reads[copy.transfer_index].dst) + copy.offset_in_transfer,
                   spot.slab.address + spot.offset + copy.offset_in_piece,
                   copy.bytes);
        copied_bytes += copy.bytes;
      }
      check_cuda(copies.enqueue(::cuda::stream_ref{stream}), "slab -> destination copies");
      // The slabs are reused as soon as the round ends, so the copies must be done.
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      copy_trace.set_io({copied_bytes, dev, static_cast<std::uint32_t>(spots.size())});
      next_piece = round_end;
    }
    return transfer_stats{plan.requested_bytes, plan.issued_bytes, plan.pieces.size()};
  }

  transfer_stats write(std::uint32_t file_id,
                       std::span<device_write const> writes,
                       staging_pool pool,
                       cudaStream_t stream) override
  {
    std::optional<nvtx_scoped_range> control_range;
    if (pool == staging_pool::spill) { control_range.emplace("gsb::spill::control"); }
    auto const plan = plan_writes(writes, _block_bytes, _piece_bytes);
    auto const dev  = flowcsd::trace_enabled() ? device_of(file_id) : -1;
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_bridge.accel_id()}};
    std::size_t next_piece = 0;
    std::size_t next_copy  = 0;
    while (next_piece < plan.pieces.size()) {
      round_slabs round{slabs_for(pool)};
      auto const round_pieces = std::span{plan.pieces}.subspan(next_piece);
      auto const spots        = round.place(round_pieces, kSpillWriteTrace.slab_wait);
      auto const round_end    = next_piece + spots.size();

      // Sources -> slabs.  Tutti's I/O kernels run on the bridge's own streams,
      // so the copies must be complete before the requests are submitted.
      {
        flowcsd::trace_scope copy_trace{kSpillWriteTrace.copy};
        std::optional<nvtx_scoped_range> pack_range;
        if (pool == staging_pool::spill) { pack_range.emplace("gsb::spill::pack"); }
        sirius::cuda::device_copy_batch copies;
        std::uint64_t copied_bytes = 0;
        for (; next_copy < plan.copies.size() && plan.copies[next_copy].piece_index < round_end;
             ++next_copy) {
          auto const& copy = plan.copies[next_copy];
          auto const& spot = spots[copy.piece_index - next_piece];
          copies.add(spot.slab.address + spot.offset + copy.offset_in_piece,
                     static_cast<std::byte const*>(writes[copy.transfer_index].src) +
                       copy.offset_in_transfer,
                     copy.bytes);
          copied_bytes += copy.bytes;
        }
        check_cuda(copies.enqueue(::cuda::stream_ref{stream}), "source -> slab copies");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        copy_trace.set_io({copied_bytes, dev, static_cast<std::uint32_t>(spots.size())});
      }

      // Slabs -> NVMe.
      {
        std::optional<nvtx_scoped_range> write_range;
        if (pool == staging_pool::spill) { write_range.emplace("gsb::spill::write"); }
        submit_and_wait(nvme_requests(bridge::io_direction::write,
                                      file_id,
                                      round_pieces.first(spots.size()),
                                      spots),
                        kSpillWriteTrace,
                        dev);
      }
      next_piece = round_end;
    }
    return transfer_stats{plan.requested_bytes, plan.issued_bytes, plan.pieces.size()};
  }

 private:
  struct open_file_entry {
    std::string path;
    std::uint64_t size_bytes{0};
    std::size_t references{0};
    std::int32_t device{-1};  ///< SSD index for the trace (flowcsd::tutti_device_index)
  };

  slab_pool& slabs_for(staging_pool pool) noexcept
  {
    return pool == staging_pool::scan ? _scan_slabs : _spill_slabs;
  }

  /// SSD index of an open file, for the trace.
  std::int32_t device_of(std::uint32_t file_id) const
  {
    std::lock_guard lock{_files_mutex};
    auto const entry = _files.find(file_id);
    return entry == _files.end() ? -1 : entry->second.device;
  }

  /// One bridge submit_and_wait.  When tracing, the bridge times every Tutti
  /// submit, and its stream wait, host submit and device phases become
  /// sub-intervals (named by @p names) of the calling thread.
  void submit_and_wait(std::vector<bridge::io_request> const& requests,
                       transfer_trace const& names,
                       std::int32_t dev)
  {
    if (!flowcsd::trace_enabled()) {
      _bridge.submit_and_wait(requests);
      return;
    }
    std::vector<bridge::submit_timing> timings;
    bridge::submit_trace const trace{&timings,
                                     flowcsd::trace_category_name(names.stream_wait),
                                     flowcsd::trace_category_name(names.submit),
                                     flowcsd::trace_category_name(names.device),
                                     names.kind};
    _bridge.submit_and_wait(requests, trace);
    for (auto const& timing : timings) {
      flowcsd::trace_io const io{timing.bytes, dev, timing.requests};
      if (timing.waited_for_stream) {
        flowcsd::record_trace(names.stream_wait, timing.stream_wait_begin, timing.submit_begin, io);
      }
      flowcsd::record_trace(names.submit, timing.submit_begin, timing.launched, io);
      flowcsd::record_trace(names.device, timing.launched, timing.completed, io);
    }
  }

  // Declared first, so it is destroyed last: the slab pools unregister their
  // memory through it, and its destructor closes files and shuts Tutti down.
  bridge::runtime _bridge;
  std::uint64_t _block_bytes;
  std::uint64_t _piece_bytes;
  slab_pool _scan_slabs;
  slab_pool _spill_slabs;

  mutable std::mutex _files_mutex;
  std::unordered_map<std::string, std::uint32_t> _file_ids;
  std::unordered_map<std::uint32_t, open_file_entry> _files;
};

}  // namespace

bool tutti_service::configured() noexcept
{
  return !flowcsd::config().tutti_runtime_config.empty();
}

tutti_service& tutti_service::instance()
{
  if (!configured()) {
    throw std::runtime_error("tutti_service: sirius.flowcsd.tutti_runtime_config is not set");
  }
  // Built on first use, so a process that never reads a Tutti path never
  // connects to the daemon.  Being a function-local static it is destroyed at
  // exit, before the CUDA runtime it was built after, which returns the daemon
  // lease and the NVMe queues for the next process.
  static bridge_tutti_service service{flowcsd::config()};
  return service;
}

#endif

}  // namespace sirius::io::tutti
