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

#include <chrono>
#include <cstdint>

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

enum class byte_counter : std::uint8_t {
  chunk_bytes_full,      ///< whole column chunks of the selected row groups (no page skipping)
  nvme_read_requested,   ///< page/chunk bytes requested after page skipping
  nvme_read_issued,      ///< bytes actually transferred by NVMe after sector alignment
  decoded_materialized,  ///< decoded column-buffer bytes produced by the reader (before row filter)
  spill_write,           ///< bytes written to NVMe by direct spill (incl. alignment padding)
  spill_read,            ///< bytes read back from NVMe by spill restore (incl. alignment padding)
  metadata_read,         ///< footer and page-index bytes read on the host
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
  count
};

/// Add to a byte counter of the running query (no-op outside a query).
void add_bytes(byte_counter counter, std::uint64_t bytes) noexcept;

/// Add to an event counter of the running query (no-op outside a query).
void add_count(event_counter counter, std::uint64_t events) noexcept;

/// Records the interval [construction, destruction) under one category.
class timed_scope {
 public:
  explicit timed_scope(time_category category) noexcept;
  ~timed_scope();

  timed_scope(timed_scope const&)            = delete;
  timed_scope& operator=(timed_scope const&) = delete;

 private:
  time_category _category;
  std::chrono::steady_clock::time_point _start;
};

/// Start collecting for one GPU query: clears counters and intervals.
void begin_query(std::uint64_t query_id);

/// Close the query, sweep the intervals and append one JSON line to the stats
/// file (flowcsd_config::stats_jsonl or $FLOWCSD_STATS_JSONL).  A failed query
/// is recorded with "ok": false.
void end_query(std::uint64_t query_id, bool succeeded);

}  // namespace sirius::flowcsd
