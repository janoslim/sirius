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

#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::flowcsd {

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr std::size_t num_time_categories  = static_cast<std::size_t>(time_category::count);
constexpr std::size_t num_trace_categories = static_cast<std::size_t>(trace_category::count);
constexpr std::size_t num_byte_counters    = static_cast<std::size_t>(byte_counter::count);
constexpr std::size_t num_event_counters   = static_cast<std::size_t>(event_counter::count);
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
// Trace / NVTX names of the sub-intervals, in trace_category order.
constexpr std::array<char const*, num_trace_categories> trace_names = {"meta.pread",
                                                                       "io.slab_wait",
                                                                       "io.stream_wait",
                                                                       "io.submit",
                                                                       "io.device",
                                                                       "io.copy",
                                                                       "spill_w.prepare",
                                                                       "spill_w.slab_wait",
                                                                       "spill_w.copy_to_slab",
                                                                       "spill_w.stream_wait",
                                                                       "spill_w.submit",
                                                                       "spill_w.device",
                                                                       "spill_r.prepare",
                                                                       "spill_r.slab_wait",
                                                                       "spill_r.stream_wait",
                                                                       "spill_r.submit",
                                                                       "spill_r.device",
                                                                       "spill_r.copy_from_slab",
                                                                       "spill_r.unpack",
                                                                       "sched.mem_wait"};
// A missing name would leave a null entry instead of failing to compile.
static_assert(trace_names.back() != nullptr, "one trace name per trace_category");

/// One recorded interval of a time category, in nanoseconds since the query start.
struct interval {
  std::int64_t start_ns;
  std::int64_t end_ns;
  std::int32_t tid;  ///< Linux thread id (as nsys reports it)
  time_category category;
};

/// One trace-only sub-interval.
struct trace_interval {
  std::int64_t start_ns;
  std::int64_t end_ns;
  trace_io io;
  std::int32_t tid;
  trace_category category;
  bool has_io;  ///< false: the bytes / dev / n_req fields do not apply (null in the trace)
};

/// Everything collected for the one query that runs at a time (Sirius serializes
/// queries per context and the study runs one context per process).
struct query_state {
  std::atomic<bool> active{false};
  std::uint64_t query_id{0};
  steady_clock::time_point start;
  std::array<std::atomic<std::uint64_t>, num_byte_counters> bytes{};
  std::array<std::atomic<std::uint64_t>, num_event_counters> events{};
  std::mutex intervals_mutex;  ///< guards intervals, trace_intervals and columns
  std::vector<interval> intervals;
  std::vector<trace_interval> trace_intervals;  ///< filled only when tracing
  std::map<std::string, column_scan_counters> columns;  ///< by "table.column"
  std::int32_t query_tid{0};      ///< thread that began the query
  nvtxRangeId_t query_range{0};   ///< NVTX flowcsd_query range, when tracing
};

query_state state;
std::mutex output_mutex;

// Writers of extra per-query trace files (add_trace_writer).
std::mutex trace_writers_mutex;
std::vector<trace_writer> trace_writers;

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

/// The event-trace file: $FLOWCSD_TRACE_JSONL, or "" (tracing off).  Read once.
std::string const& trace_path()
{
  static std::string const path = [] {
    char const* env = std::getenv("FLOWCSD_TRACE_JSONL");
    return std::string(env != nullptr ? env : "");
  }();
  return path;
}

/// Linux thread id of the calling thread, the id nsys shows for it.
std::int32_t current_tid() noexcept
{
  thread_local std::int32_t const tid = static_cast<std::int32_t>(::syscall(SYS_gettid));
  return tid;
}

/// Nanoseconds from the query start to @p time.  Caller holds intervals_mutex.
std::int64_t ns_since_start(steady_clock::time_point time)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time - state.start).count();
}

// -- NVTX ("flowcsd" domain; every call is a no-op without a profiler attached) --

nvtxDomainHandle_t nvtx_domain()
{
  static nvtxDomainHandle_t const domain = nvtxDomainCreateA("flowcsd");
  return domain;
}

nvtxEventAttributes_t nvtx_message(char const* name)
{
  nvtxEventAttributes_t attributes{};
  attributes.version       = NVTX_VERSION;
  attributes.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.messageType   = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = name;
  return attributes;
}

/// Thread-local nested range, for the scoped timers.
void nvtx_push(char const* name)
{
  auto const attributes = nvtx_message(name);
  nvtxDomainRangePushEx(nvtx_domain(), &attributes);
}

void nvtx_pop() { nvtxDomainRangePop(nvtx_domain()); }

/// Store one sub-interval of the running query (no-op unless tracing a query).
void push_trace(trace_category category,
                steady_clock::time_point begin,
                steady_clock::time_point end,
                bool has_io,
                trace_io const& io) noexcept
{
  if (!trace_enabled() || !state.active.load(std::memory_order_relaxed)) { return; }
  auto const tid = current_tid();
  std::lock_guard lock(state.intervals_mutex);
  // Re-check under the lock: end_query swaps the vectors out while holding it.
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  state.trace_intervals.push_back(
    trace_interval{ns_since_start(begin), ns_since_start(end), io, tid, category, has_io});
}

/// Append the event trace of one query to trace_path(): a flowcsd_query line spanning
/// [0, wall_ns), then every interval sorted by start time.
void write_trace(std::vector<interval> const& intervals,
                 std::vector<trace_interval> const& sub_intervals,
                 std::uint64_t query_id,
                 bool succeeded,
                 std::int64_t wall_ns,
                 std::int32_t query_tid)
{
  struct trace_line {
    std::int64_t start_ns;
    std::int64_t end_ns;
    char const* name;
    std::int32_t tid;
    bool has_io;
    trace_io io;
  };
  std::vector<trace_line> lines;
  lines.reserve(intervals.size() + sub_intervals.size());
  for (auto const& iv : intervals) {
    lines.push_back({iv.start_ns,
                     iv.end_ns,
                     time_names[static_cast<std::size_t>(iv.category)],
                     iv.tid,
                     false,
                     {}});
  }
  for (auto const& iv : sub_intervals) {
    lines.push_back({iv.start_ns,
                     iv.end_ns,
                     trace_names[static_cast<std::size_t>(iv.category)],
                     iv.tid,
                     iv.has_io,
                     iv.io});
  }
  std::stable_sort(lines.begin(), lines.end(), [](trace_line const& lhs, trace_line const& rhs) {
    return lhs.start_ns < rhs.start_ns;
  });

  std::string text;
  text.reserve(160 * (lines.size() + 1));
  auto out = std::back_inserter(text);
  std::format_to(out,
                 "{{\"cat\": \"flowcsd_query\", \"tid\": {}, \"t0_ns\": 0, \"t1_ns\": {}, "
                 "\"bytes\": null, \"dev\": null, \"n_req\": null, \"query_id\": {}, \"ok\": {}}}\n",
                 query_tid,
                 wall_ns,
                 query_id,
                 succeeded ? "true" : "false");
  for (auto const& line : lines) {
    std::format_to(
      out, "{{\"cat\": \"{}\", \"tid\": {}, \"t0_ns\": {}, \"t1_ns\": {}, ", line.name, line.tid,
      line.start_ns, line.end_ns);
    if (!line.has_io) {
      std::format_to(out, "\"bytes\": null, \"dev\": null, \"n_req\": null}}\n");
    } else if (line.io.dev < 0) {
      std::format_to(
        out, "\"bytes\": {}, \"dev\": null, \"n_req\": {}}}\n", line.io.bytes, line.io.n_req);
    } else {
      std::format_to(out,
                     "\"bytes\": {}, \"dev\": {}, \"n_req\": {}}}\n",
                     line.io.bytes,
                     line.io.dev,
                     line.io.n_req);
    }
  }

  std::lock_guard lock(output_mutex);
  std::ofstream file(trace_path(), std::ios::app);
  file << text;
  if (!file) { SIRIUS_LOG_ERROR("[flowcsd] could not append the event trace to {}", trace_path()); }
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

void add_trace_writer(trace_writer writer)
{
  std::lock_guard lock(trace_writers_mutex);
  trace_writers.push_back(std::move(writer));
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
  : _category(category), _tracing(trace_enabled()), _start(steady_clock::now())
{
  if (_tracing) { nvtx_push(time_names[static_cast<std::size_t>(category)]); }
}

timed_scope::~timed_scope()
{
  if (_tracing) { nvtx_pop(); }
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  auto const end = steady_clock::now();
  auto const tid = current_tid();
  std::lock_guard lock(state.intervals_mutex);
  // Re-check under the lock: end_query swaps the vector out while holding it.
  if (!state.active.load(std::memory_order_relaxed)) { return; }
  state.intervals.push_back({ns_since_start(_start), ns_since_start(end), tid, _category});
}

bool trace_enabled() noexcept
{
  static bool const enabled = !trace_path().empty();
  return enabled;
}

char const* trace_category_name(trace_category category) noexcept
{
  return trace_names[static_cast<std::size_t>(category)];
}

void record_trace(trace_category category,
                  steady_clock::time_point begin,
                  steady_clock::time_point end,
                  trace_io const& io) noexcept
{
  push_trace(category, begin, end, true, io);
}

trace_scope::trace_scope(trace_category category) noexcept
  : _category(category),
    _tracing(trace_enabled()),
    _start(_tracing ? steady_clock::now() : steady_clock::time_point{})
{
  if (_tracing) { nvtx_push(trace_category_name(category)); }
}

trace_scope::~trace_scope()
{
  if (!_tracing) { return; }
  nvtx_pop();
  push_trace(_category, _start, steady_clock::now(), _has_io, _io);
}

void begin_query(std::uint64_t query_id)
{
  std::lock_guard lock(state.intervals_mutex);
  state.intervals.clear();
  state.trace_intervals.clear();
  state.columns.clear();
  for (auto& b : state.bytes) {
    b.store(0, std::memory_order_relaxed);
  }
  for (auto& e : state.events) {
    e.store(0, std::memory_order_relaxed);
  }
  state.query_id  = query_id;
  state.query_tid = current_tid();
  // The flowcsd_query NVTX range starts with t = 0 of the trace, so nsys times
  // align with the trace by subtracting the range's start.  A start/end range,
  // because end_query may run on another thread.
  if (trace_enabled()) {
    auto const attributes = nvtx_message("flowcsd_query");
    state.query_range     = nvtxDomainRangeStartEx(nvtx_domain(), &attributes);
  }
  state.start = steady_clock::now();
  state.active.store(true, std::memory_order_release);
}

void end_query(std::uint64_t query_id, bool succeeded)
{
  // Stop collecting first so late scopes (e.g. a background spill) cannot race the sweep.
  std::vector<interval> intervals;
  std::vector<trace_interval> sub_intervals;
  std::map<std::string, column_scan_counters> columns;
  std::int64_t wall_ns     = 0;
  std::int32_t query_tid   = 0;
  nvtxRangeId_t query_range = 0;
  steady_clock::time_point query_start;
  {
    std::lock_guard lock(state.intervals_mutex);
    if (!state.active.load(std::memory_order_relaxed) || state.query_id != query_id) { return; }
    state.active.store(false, std::memory_order_release);
    wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(steady_clock::now() - state.start)
                .count();
    intervals.swap(state.intervals);
    sub_intervals.swap(state.trace_intervals);
    columns.swap(state.columns);
    query_tid   = state.query_tid;
    query_range = state.query_range;
    query_start = state.start;
  }

  if (trace_enabled()) {
    nvtxDomainRangeEnd(nvtx_domain(), query_range);
    write_trace(intervals, sub_intervals, query_id, succeeded, wall_ns, query_tid);
    // Extra trace files share the JSONL's name without ".trace.jsonl".
    constexpr std::string_view jsonl_suffix = ".trace.jsonl";
    std::string base = trace_path();
    if (base.ends_with(jsonl_suffix)) { base.resize(base.size() - jsonl_suffix.size()); }
    std::lock_guard lock(trace_writers_mutex);
    for (auto const& writer : trace_writers) {
      writer(base, query_start);
    }
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
