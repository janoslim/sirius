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

#include "flowcsd/flowcsd_stats.hpp"

#include "flowcsd/flowcsd_config.hpp"
#include "log/logging.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace sirius::flowcsd {

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr std::size_t num_time_categories = static_cast<std::size_t>(time_category::count);
constexpr std::size_t num_byte_counters   = static_cast<std::size_t>(byte_counter::count);
constexpr std::size_t num_event_counters  = static_cast<std::size_t>(event_counter::count);
constexpr double ns_per_ms                = 1e6;

// JSON key names, in enum order (PLAN section 5.5 names where the contract has one).
constexpr std::array<char const*, num_time_categories> time_names = {
  "spill_write", "spill_read", "nvme_read", "decode", "hash_partition", "compute"};
constexpr std::array<char const*, num_byte_counters> byte_names = {"chunk_bytes_full",
                                                                   "nvme_read_requested",
                                                                   "nvme_read_issued",
                                                                   "decoded_materialized",
                                                                   "spill_write",
                                                                   "spill_read",
                                                                   "metadata_read",
                                                                   "host_spill_write",
                                                                   "spill_unread",
                                                                   "spill_live_peak"};
constexpr std::array<char const*, num_event_counters> event_names = {"row_groups_total",
                                                                     "row_groups_read",
                                                                     "pages_total",
                                                                     "pages_read",
                                                                     "io_requests",
                                                                     "spilled_batches",
                                                                     "restored_batches",
                                                                     "hash_partitions"};

/// One recorded interval, in nanoseconds since the query start.
struct interval {
  std::int64_t start_ns;
  std::int64_t end_ns;
  time_category category;
};

/// Everything collected for the one query that runs at a time (Sirius serializes
/// queries per context and the study runs one context per process).
struct query_state {
  std::atomic<bool> active{false};
  std::uint64_t query_id{0};
  steady_clock::time_point start;
  std::array<std::atomic<std::uint64_t>, num_byte_counters> bytes{};
  std::array<std::atomic<std::uint64_t>, num_event_counters> events{};
  std::mutex intervals_mutex;  ///< guards intervals and columns
  std::vector<interval> intervals;
  std::map<std::string, column_scan_counters> columns;  ///< by "table.column"
};

query_state state;
std::mutex output_mutex;

/// Per-category times derived from the intervals of one query.
struct time_breakdown {
  std::array<double, num_time_categories> exclusive_ns{};  ///< priority-attributed, sums to wall
  std::array<double, num_time_categories> union_ns{};      ///< wall time the category was active
  std::array<double, num_time_categories> busy_ns{};       ///< summed interval lengths
  double other_ns{0};
};

/// Sweep the intervals once; charge each instant to the highest-priority active category.
time_breakdown sweep(std::vector<interval> const& intervals, std::int64_t wall_ns)
{
  struct edge {
    std::int64_t time_ns;
    int delta;
    std::size_t category;
  };
  std::vector<edge> edges;
  edges.reserve(intervals.size() * 2);
  time_breakdown out;
  for (auto const& iv : intervals) {
    // Clamp to the query window: a scope may start before begin_query or end after end_query.
    auto const start = std::clamp<std::int64_t>(iv.start_ns, 0, wall_ns);
    auto const end   = std::clamp<std::int64_t>(iv.end_ns, 0, wall_ns);
    if (end <= start) { continue; }
    auto const category = static_cast<std::size_t>(iv.category);
    out.busy_ns[category] += static_cast<double>(end - start);
    edges.push_back({start, +1, category});
    edges.push_back({end, -1, category});
  }
  std::sort(edges.begin(), edges.end(), [](edge const& a, edge const& b) {
    return a.time_ns < b.time_ns;
  });

  std::array<int, num_time_categories> active{};
  std::int64_t previous_ns = 0;
  for (auto const& e : edges) {
    auto const span_ns = static_cast<double>(e.time_ns - previous_ns);
    if (span_ns > 0) {
      // The enum order is the attribution priority: the first active category wins.
      auto const winner =
        std::find_if(active.begin(), active.end(), [](int count) { return count > 0; });
      if (winner == active.end()) {
        out.other_ns += span_ns;
      } else {
        out.exclusive_ns[static_cast<std::size_t>(winner - active.begin())] += span_ns;
      }
      for (std::size_t c = 0; c < num_time_categories; ++c) {
        if (active[c] > 0) { out.union_ns[c] += span_ns; }
      }
    }
    active[e.category] += e.delta;
    previous_ns = e.time_ns;
  }
  out.other_ns += static_cast<double>(wall_ns - previous_ns);
  return out;
}

/// `"name": value` pairs of one JSON object.
template <typename Values, typename Names>
std::string json_object(Values const& values, Names const& names, double scale)
{
  std::string text = "{";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) { text += ", "; }
    text += std::format("\"{}\": {:.3f}", names[i], static_cast<double>(values[i]) / scale);
  }
  return text + "}";
}

template <typename Counters, typename Names>
std::string json_counters(Counters const& counters, Names const& names)
{
  std::string text = "{";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) { text += ", "; }
    text += std::format("\"{}\": {}", names[i], counters[i].load(std::memory_order_relaxed));
  }
  return text + "}";
}

/// `"table.column": {...}` entries of the per-column scan breakdown.
std::string json_columns(std::map<std::string, column_scan_counters> const& columns)
{
  std::string text = "{";
  for (auto const& [key, c] : columns) {
    if (text.size() > 1) { text += ", "; }
    text += std::format(
      "\"{}\": {{\"chunk_bytes_full\": {}, \"nvme_read_requested\": {}, \"pages_total\": {}, "
      "\"pages_read\": {}}}",
      key,
      c.chunk_bytes_full,
      c.nvme_read_requested,
      c.pages_total,
      c.pages_read);
  }
  return text + "}";
}

/// Where the JSON lines go: $FLOWCSD_STATS_JSONL, else the config, else nowhere.
std::string stats_path()
{
  if (char const* env = std::getenv("FLOWCSD_STATS_JSONL"); env != nullptr && *env != '\0') {
    return env;
  }
  return config().stats_jsonl;
}

}  // namespace

void add_bytes(byte_counter counter, std::uint64_t bytes) noexcept
{
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  state.bytes[static_cast<std::size_t>(counter)].fetch_add(bytes, std::memory_order_relaxed);
}

void max_bytes(byte_counter counter, std::uint64_t bytes) noexcept
{
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  auto& value        = state.bytes[static_cast<std::size_t>(counter)];
  std::uint64_t seen = value.load(std::memory_order_relaxed);
  // compare_exchange_weak reloads `seen` on failure; stop once the stored value is not lower.
  while (seen < bytes && !value.compare_exchange_weak(seen, bytes, std::memory_order_relaxed)) {}
}

void add_count(event_counter counter, std::uint64_t events) noexcept
{
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  state.events[static_cast<std::size_t>(counter)].fetch_add(events, std::memory_order_relaxed);
}

void add_column_scan(std::string const& key, column_scan_counters const& counters)
{
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  std::lock_guard lock(state.intervals_mutex);
  auto& total = state.columns[key];
  total.chunk_bytes_full += counters.chunk_bytes_full;
  total.nvme_read_requested += counters.nvme_read_requested;
  total.pages_total += counters.pages_total;
  total.pages_read += counters.pages_read;
}

timed_scope::timed_scope(time_category category) noexcept
  : _category(category), _start(steady_clock::now())
{
}

timed_scope::~timed_scope()
{
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  auto const end = steady_clock::now();
  std::lock_guard lock(state.intervals_mutex);
  // Re-check under the lock: end_query swaps the vector out while holding it.
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  auto const to_ns = [](steady_clock::duration d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
  };
  state.intervals.push_back({to_ns(_start - state.start), to_ns(end - state.start), _category});
}

void begin_query(std::uint64_t query_id)
{
  std::lock_guard lock(state.intervals_mutex);
  state.intervals.clear();
  state.columns.clear();
  for (auto& b : state.bytes) {
    b.store(0, std::memory_order_relaxed);
  }
  for (auto& e : state.events) {
    e.store(0, std::memory_order_relaxed);
  }
  state.query_id = query_id;
  state.start    = steady_clock::now();
  state.active.store(true, std::memory_order_release);
}

void end_query(std::uint64_t query_id, bool succeeded)
{
  // Stop collecting first so late scopes (e.g. a background spill) cannot race the sweep.
  std::vector<interval> intervals;
  std::map<std::string, column_scan_counters> columns;
  std::int64_t wall_ns = 0;
  {
    std::lock_guard lock(state.intervals_mutex);
    if (!state.active.load(std::memory_order_relaxed) || state.query_id != query_id) { return; }
    state.active.store(false, std::memory_order_release);
    wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(steady_clock::now() - state.start)
                .count();
    intervals.swap(state.intervals);
    columns.swap(state.columns);
  }

  auto const path = stats_path();
  if (path.empty()) { return; }
  auto const breakdown = sweep(intervals, wall_ns);

  // time_ms: exclusive attribution plus "other" (sums to wall_ms).
  std::string time_ms = json_object(breakdown.exclusive_ns, time_names, ns_per_ms);
  time_ms.pop_back();
  time_ms += std::format(", \"other\": {:.3f}}}", breakdown.other_ns / ns_per_ms);

  auto const& cfg  = config();
  std::string line = std::format(
    "{{\"query_id\": {}, \"ok\": {}, \"wall_ms\": {:.3f}, \"time_ms\": {}, \"union_ms\": {}, "
    "\"busy_ms\": {}, \"bytes\": {}, \"counts\": {}, \"columns\": {}, \"intervals\": {}, "
    "\"flowcsd\": {{\"tutti\": {}, \"page_skipping\": {}, \"direct_disk_spill\": {}}}}}\n",
    query_id,
    succeeded ? "true" : "false",
    static_cast<double>(wall_ns) / ns_per_ms,
    time_ms,
    json_object(breakdown.union_ns, time_names, ns_per_ms),
    json_object(breakdown.busy_ns, time_names, ns_per_ms),
    json_counters(state.bytes, byte_names),
    json_counters(state.events, event_names),
    json_columns(columns),
    intervals.size(),
    cfg.tutti_runtime_config.empty() ? "false" : "true",
    cfg.page_skipping ? "true" : "false",
    cfg.direct_disk_spill ? "true" : "false");

  std::lock_guard lock(output_mutex);
  std::ofstream out(path, std::ios::app);
  out << line;
  if (!out) { SIRIUS_LOG_ERROR("[flowcsd] could not append query stats to {}", path); }
}

}  // namespace sirius::flowcsd
