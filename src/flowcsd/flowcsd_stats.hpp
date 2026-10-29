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

// Per-query statistics for the FlowCSD study (PLAN section 5.5).
//
// Byte/event counters are relaxed atomics.  Time is recorded as intervals
// [start, end) per category on whichever thread did the work; at query end the
// intervals are swept once and each wall-clock instant is charged to the
// highest-priority category active at that instant (enum order below), the rest
// to "other".  The exclusive times therefore sum to the query wall time even
// though scan, decode, spill and compute run concurrently on different threads.
// Plain per-category busy time (sum of interval lengths) is reported as well.
//
// Event trace (FLOWCSD_TRACE_JSONL=<path>, off by default): at query end every
// recorded interval -- the time categories above plus trace-only sub-intervals
// (trace_category) -- is appended to <path> as one JSON line, with times relative
// to the query start, and the same intervals are NVTX ranges in the "flowcsd"
// domain while the query runs.  Sub-intervals never enter the time_ms
// attribution, so traced and untraced runs report comparable metrics.

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace sirius::flowcsd {

/// Latency-breakdown categories, highest attribution priority first.
enum class time_category : std::uint8_t {
  spill_write,     ///< GPU -> NVMe spill of a batch (staging copies + Tutti writes)
  spill_read,      ///< NVMe -> GPU restore of a spilled batch
  nvme_read,       ///< Parquet chunk/page reads (Tutti submit -> completion + staging copy)
  decode,          ///< cuDF decompression + decode of the read pages
  hash_partition,  ///< PARTITION operator (hash scatter of build/probe batches)
  compute,         ///< every other operator: filter, join build/probe, aggregate, sort, projection
  count
};

/// Trace-only sub-intervals: what a thread did inside a time category (waiting
/// for a staging slab, building a Tutti submit, waiting for the SSD, copying).
/// trace_category_name() gives the trace / NVTX name in brackets.
enum class trace_category : std::uint8_t {
  meta_pread,              ///< [meta.pread] O_DIRECT host read of a footer or page index
  io_slab_wait,            ///< [io.slab_wait] scan read waits for a free scan staging slab
  io_stream_wait,          ///< [io.stream_wait] waits for a free Tutti submit stream (16 in flight)
  io_submit,               ///< [io.submit] host builds one Tutti submit and launches its I/O kernel
  io_device,               ///< [io.device] kernel launched -> host saw its completion (SSD + PCIe)
  io_copy,                 ///< [io.copy] slab -> destination D2D copies + stream sync
  spill_w_prepare,         ///< [spill_w.prepare] wait for the batch's producers, plan layout, take region
  spill_w_slab_wait,       ///< [spill_w.slab_wait] spill write waits for a free spill staging slab
  spill_w_copy_to_slab,    ///< [spill_w.copy_to_slab] batch buffers -> slab D2D copies + stream sync
  spill_w_stream_wait,     ///< [spill_w.stream_wait] waits for a free Tutti submit stream
  spill_w_submit,          ///< [spill_w.submit] host builds one Tutti WRITE submit and launches it
  spill_w_device,          ///< [spill_w.device] kernel launched -> host saw its completion
  spill_r_prepare,         ///< [spill_r.prepare] allocate the restore buffers
  spill_r_slab_wait,       ///< [spill_r.slab_wait] spill read waits for a free spill staging slab
  spill_r_stream_wait,     ///< [spill_r.stream_wait] waits for a free Tutti submit stream
  spill_r_submit,          ///< [spill_r.submit] host builds one Tutti READ submit and launches it
  spill_r_device,          ///< [spill_r.device] kernel launched -> host saw its completion
  spill_r_copy_from_slab,  ///< [spill_r.copy_from_slab] slab -> restore buffers D2D copies + sync
  spill_r_unpack,          ///< [spill_r.unpack] rebuild cudf columns from the buffers + stream sync
  sched_mem_wait,          ///< [sched.mem_wait] a ready pipeline task waits for its GPU reservation
  count
};

enum class byte_counter : std::uint8_t {
  chunk_bytes_full,      ///< whole column chunks of the selected row groups (no page skipping)
  nvme_read_requested,   ///< page/chunk bytes requested after page skipping
  nvme_read_issued,      ///< bytes actually transferred by NVMe after sector alignment
  decoded_materialized,  ///< decoded column-buffer bytes produced by the reader (before row filter)
  spill_write,           ///< bytes written to NVMe by direct spill (incl. alignment padding)
  spill_read,            ///< bytes read back from NVMe by spill restore (incl. alignment padding)
  metadata_read,         ///< footer and page-index bytes read on the host
  host_spill_write,      ///< bytes the downgrade executor moved into the HOST tier (0 with direct spill)
  spill_unread,          ///< spilled bytes whose batch was dropped before being read back
  spill_live_peak,       ///< largest spill-store occupancy seen during the query (a maximum, not a sum)
  managed_build_copy,    ///< build-side batch bytes copied into managed memory
                         ///< (sirius.memory.gpu.large_hash_join_in_managed_memory): the PARTITION's
                         ///< copies on arrival and the fold's copy of a lone build batch
  managed_hash_state_peak,  ///< largest managed hash-join state live during the query, both
                            ///< managed memories (a maximum)
  count
};

enum class event_counter : std::uint8_t {
  row_groups_total,  ///< row groups in the scanned files
  row_groups_read,   ///< row groups left after statistics pruning
  pages_total,       ///< data pages of the read column chunks
  pages_read,        ///< data pages actually read after page skipping
  io_requests,       ///< NVMe requests submitted through Tutti (scan + spill)
  spilled_batches,   ///< batches written by direct spill
  restored_batches,  ///< spilled batches read back
  hash_partitions,   ///< partitions produced by PARTITION operators (latched counts, summed)
  managed_hash_joins,  ///< hash joins that built their hash table in managed memory
  count
};

/// Add to a byte counter of the running query (no-op outside a query).
void add_bytes(byte_counter counter, std::uint64_t bytes) noexcept;

/// Raise a byte counter of the running query to @p bytes if it is lower (for
/// maxima such as spill_live_peak; no-op outside a query).
void max_bytes(byte_counter counter, std::uint64_t bytes) noexcept;

/// Add to an event counter of the running query (no-op outside a query).
void add_count(event_counter counter, std::uint64_t events) noexcept;

/// Scan counters of one (table, column) pair, the byte_counter/event_counter
/// totals broken down so they can be compared with a per-column page oracle.
struct column_scan_counters {
  std::uint64_t chunk_bytes_full{0};     ///< whole column chunks of the selected row groups
  std::uint64_t nvme_read_requested{0};  ///< bytes requested after page skipping
  std::uint64_t pages_total{0};          ///< data pages of those chunks
  std::uint64_t pages_read{0};           ///< data pages read after page skipping
};

/// Add to the per-column breakdown of the running query under @p key
/// ("table.column"); no-op outside a query.
void add_column_scan(std::string const& key, column_scan_counters const& counters);

/// Records the interval [construction, destruction) under one category (and,
/// when tracing, an NVTX range of the category name).
class timed_scope {
 public:
  explicit timed_scope(time_category category) noexcept;
  ~timed_scope();

  timed_scope(timed_scope const&)            = delete;
  timed_scope& operator=(timed_scope const&) = delete;

 private:
  time_category _category;
  bool _tracing;
  std::chrono::steady_clock::time_point _start;
};

/// True when $FLOWCSD_TRACE_JSONL names an event-trace file (read once per process).
[[nodiscard]] bool trace_enabled() noexcept;

/// Trace and NVTX name of a sub-interval, e.g. "io.device".
[[nodiscard]] char const* trace_category_name(trace_category category) noexcept;

/// The I/O fields of a trace line.
struct trace_io {
  std::uint64_t bytes{0};  ///< bytes moved (reservation bytes for sched.mem_wait)
  std::int32_t dev{-1};    ///< SSD: position of the file's prefix in tutti_path_prefixes; -1 none
  std::uint32_t n_req{0};  ///< Tutti requests (io_requests) behind the interval
};

/// Record a sub-interval [begin, end) that already happened on this thread (for
/// phases timed elsewhere, e.g. inside the Tutti bridge).  No-op unless a query
/// is being traced; no NVTX range (the timer of the phase emits it).
void record_trace(trace_category category,
                  std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end,
                  trace_io const& io) noexcept;

/// Records [construction, destruction) as a sub-interval and an NVTX range of the
/// same name.  No-op unless tracing.
class trace_scope {
 public:
  explicit trace_scope(trace_category category) noexcept;
  ~trace_scope();

  trace_scope(trace_scope const&)            = delete;
  trace_scope& operator=(trace_scope const&) = delete;

  /// Attach I/O fields to the line written when the scope ends.
  void set_io(trace_io const& io) noexcept
  {
    _io     = io;
    _has_io = true;
  }

 private:
  trace_category _category;
  bool _tracing;
  bool _has_io{false};
  trace_io _io;
  std::chrono::steady_clock::time_point _start;
};

/// Extra per-query trace output written when a traced query ends, next to the
/// JSONL: @p base is the trace path without ".trace.jsonl" (the Tutti service
/// writes <base>.tutti_*.csv), @p query_start the trace's t = 0.
using trace_writer =
  std::function<void(std::string const& base, std::chrono::steady_clock::time_point query_start)>;

/// Register a writer (kept for the life of the process).
void add_trace_writer(trace_writer writer);

/// Start collecting for one GPU query: clears counters and intervals.
void begin_query(std::uint64_t query_id);

/// Close the query, sweep the intervals and append one JSON line to the stats
/// file (flowcsd_config::stats_jsonl or $FLOWCSD_STATS_JSONL).  A failed query
/// is recorded with "ok": false.  When tracing, also append the event trace.
void end_query(std::uint64_t query_id, bool succeeded);

}  // namespace sirius::flowcsd
