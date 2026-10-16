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

// The only translation unit that sees Tutti's types.
//
// Maps the integer ids of flowcsd_tutti_bridge.hpp onto Tutti handles and turns
// a list of io_requests into Tutti submits that stay inside the local-NVMe
// DataPath's per-submit limits, with at most kMaxSubmitsInFlight in flight.
// When asked (submit_trace), it timestamps each submit -- stream wait, host
// submit, kernel launch, completion seen -- and marks the same phases as NVTX
// ranges, so Sirius's event trace can split an NVMe read into waiting and work.

#include "flowcsd_tutti_bridge.hpp"

#include <tutti/accelerator_device_guard.h>
#include <tutti/io_trace.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/status.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flowcsd_tutti_bridge {

namespace {

// Requests are block granular.  Tutti maps a file through its FIEMAP extents,
// which ext4 reports in 4 KiB blocks, and 4 KiB is a multiple of the 512-byte
// LBA both data disks are formatted with, so a block-aligned request is also
// LBA aligned.
constexpr std::uint64_t kBlockBytes = 4096;

// Limits of Tutti's local-NVMe and device-group DataPaths when the runtime YAML
// leaves max_in_flight_operations and max_batch_entries at 0
// (local_nvme_data_path.cpp, striped_data_path.cpp).  A submit beyond any of
// them is rejected as a whole, so the bridge stays under them instead of
// retrying rejected submits.
constexpr std::size_t kMaxSubmitsInFlight    = 16;
constexpr std::size_t kMaxRequestsPerSubmit  = 256;
constexpr std::uint64_t kMaxEntriesPerSubmit = 256;

// Tutti splits a request into NVMe commands of at most MDTS bytes, and again at
// every extent boundary.  128 KiB is the MDTS Tutti assumes before it has asked
// the controller; the data disks report 1-2 MiB, so the estimate is safe.
constexpr std::uint64_t kMdtsFloorBytes = 128ull << 10;

// A submit may carry at most max_batch_entries * MDTS bytes.
constexpr std::uint64_t kMaxSubmitBytes = kMaxEntriesPerSubmit * kMdtsFloorBytes;

// A submit's I/O kernel fails a command after its own CQ poll budget, so wait()
// returns long before this unless Tutti itself is wedged.
constexpr std::uint64_t kWaitTimeoutMs = 60'000;

std::string status_text(std::string const& operation, tutti::Status const& status)
{
  return "Tutti " + operation + " failed: " + status.message();
}

[[noreturn]] void throw_status(std::string const& operation, tutti::Status const& status)
{
  throw std::runtime_error(status_text(operation, status));
}

void check_cuda(cudaError_t error, char const* operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("flowcsd_tutti_bridge: ") + operation +
                             " failed: " + cudaGetErrorString(error));
  }
}

/// Upper bound on the NVMe commands one request becomes: one per MDTS, plus one
/// for an extent boundary inside it (the dataset files have few large extents,
/// so a request of at most a few MiB crosses at most one).
std::uint64_t estimated_entries(std::uint64_t bytes)
{
  return (bytes + kMdtsFloorBytes - 1) / kMdtsFloorBytes + 1;
}

/// Cut requests into consecutive submits that each stay inside the per-submit limits.
std::vector<std::span<tutti::IoRequest const>> split_into_submits(
  std::span<tutti::IoRequest const> requests)
{
  std::vector<std::span<tutti::IoRequest const>> submits;
  std::size_t first     = 0;
  std::uint64_t entries = 0;
  std::uint64_t bytes   = 0;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto const request_bytes   = requests[index].length;
    auto const request_entries = estimated_entries(request_bytes);
    bool const full            = index - first == kMaxRequestsPerSubmit ||
                      entries + request_entries > kMaxEntriesPerSubmit ||
                      bytes + request_bytes > kMaxSubmitBytes;
    // A request that alone exceeds a limit still gets a submit; Tutti then names the limit.
    if (full && index > first) {
      submits.push_back(requests.subspan(first, index - first));
      first   = index;
      entries = 0;
      bytes   = 0;
    }
    entries += request_entries;
    bytes += request_bytes;
  }
  if (first < requests.size()) { submits.push_back(requests.subspan(first)); }
  return submits;
}

/// Keep the first failure of a call; later ones are usually its consequences.
void keep_first(std::string& first_error, std::string message)
{
  if (first_error.empty()) { first_error = std::move(message); }
}

/// The "flowcsd" NVTX domain, shared with Sirius's FlowCSD trace (same name, same domain in nsys).
nvtxDomainHandle_t nvtx_domain()
{
  static nvtxDomainHandle_t const domain = nvtxDomainCreateA("flowcsd");
  return domain;
}

/// Start an NVTX range that may end on another thread or overlap others.  Without a
/// profiler attached NVTX calls do nothing.  Only call with a non-null name.
nvtxRangeId_t nvtx_start(char const* name)
{
  nvtxEventAttributes_t attributes{};
  attributes.version       = NVTX_VERSION;
  attributes.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.messageType   = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = name;
  return nvtxDomainRangeStartEx(nvtx_domain(), &attributes);
}

void nvtx_end(nvtxRangeId_t range) { nvtxDomainRangeEnd(nvtx_domain(), range); }

/// An accepted Tutti submit and the stream its I/O kernel runs on.
struct pending_submit {
  tutti::IoHandle io;
  cudaStream_t stream;
  std::size_t timing_index{0};    ///< its entry in submit_trace::timings, when traced
  nvtxRangeId_t device_range{0};  ///< its launch -> completion NVTX range, when traced
};

}  // namespace

struct runtime::impl {
  std::unique_ptr<tutti::TuttiRuntime> owner;
  tutti::StorageRuntime* storage{nullptr};
  std::int32_t accel_id{-1};

  // Integer ids handed out for Tutti handles (one counter for both kinds).
  std::mutex handles_mutex;
  std::uint32_t next_id{1};
  std::unordered_map<std::uint32_t, tutti::TargetHandle> files;
  std::unordered_map<std::uint32_t, tutti::MemoryHandle> memories;

  // One CUDA stream per submit that may be in flight.  Tutti runs a submit's
  // I/O (SQ writes, doorbell, CQ polling) as a GPU kernel on the submit's
  // stream, so submits on separate streams overlap, and the stream count caps
  // the submits in flight at the DataPath limit.
  std::mutex streams_mutex;
  std::condition_variable stream_returned;
  std::vector<cudaStream_t> streams;
  std::vector<cudaStream_t> free_streams;

  ~impl();

  void create_streams();
  /// A free stream; waits while every stream is busy.  @p waited tells whether it
  /// had to, and @p nvtx_wait (if not null) names an NVTX range over that wait.
  cudaStream_t acquire_stream(bool& waited, char const* nvtx_wait);
  /// nullptr when every stream is in use.
  cudaStream_t try_acquire_stream();
  void release_stream(cudaStream_t stream);

  // Raw I/O trace (TUTTI_IO_TRACE=1): labels of submit kinds (tag = index + 1)
  // and the last %globaltimer <-> steady_clock pairing.
  std::mutex trace_mutex;
  std::vector<std::string> kinds;
  tutti::io_trace::ClockSample clock_start;
  /// Tutti tag for a kind label (registered on first use); 0 for no label.
  std::uint32_t kind_tag(char const* kind);

  std::vector<tutti::IoRequest> to_tutti(std::span<io_request const> requests);

  /// Wait for one submit and release its handle; a failure goes to first_error.
  void finish(pending_submit const& submit, std::string& first_error);
};

runtime::impl::~impl()
{
  // Destructors cannot throw, so failures are only reported.
  auto report = [](std::string const& message) {
    std::fprintf(stderr, "flowcsd_tutti_bridge: %s\n", message.c_str());
  };
  if (storage != nullptr) {
    for (auto const& [id, target] : files) {
      if (auto const status = storage->close(target); !status.ok()) {
        report(status_text("close", status));
      }
    }
    for (auto const& [id, memory] : memories) {
      if (auto const status = storage->unregister_memory(memory); !status.ok()) {
        report(status_text("unregister_memory", status));
      }
    }
  }
  if (owner) {
    // Releases the daemon lease and the NVMe queues for the next process.
    if (auto const status = owner->shutdown(); !status.ok()) {
      report(status_text("shutdown", status));
    }
  }
  for (auto const stream : streams) {
    (void)cudaStreamDestroy(stream);
  }
}

void runtime::impl::create_streams()
{
  tutti::DeviceGuard guard{accel_id};
  if (!guard.ok()) { throw_status("select accelerator", guard.status()); }
  for (std::size_t index = 0; index < kMaxSubmitsInFlight; ++index) {
    cudaStream_t stream = nullptr;
    // Non-blocking: an I/O kernel must not wait for unrelated work on the legacy default stream.
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    streams.push_back(stream);
  }
  free_streams = streams;
  if (auto const restored = guard.restore(); !restored.ok()) {
    throw_status("restore device", restored);
  }
}

cudaStream_t runtime::impl::acquire_stream(bool& waited, char const* nvtx_wait)
{
  std::unique_lock lock{streams_mutex};
  waited = free_streams.empty();
  if (waited) {
    nvtxRangeId_t const range = nvtx_wait != nullptr ? nvtx_start(nvtx_wait) : 0;
    stream_returned.wait(lock, [this] { return !free_streams.empty(); });
    if (nvtx_wait != nullptr) { nvtx_end(range); }
  }
  auto const stream = free_streams.back();
  free_streams.pop_back();
  return stream;
}

cudaStream_t runtime::impl::try_acquire_stream()
{
  std::lock_guard lock{streams_mutex};
  if (free_streams.empty()) { return nullptr; }
  auto const stream = free_streams.back();
  free_streams.pop_back();
  return stream;
}

void runtime::impl::release_stream(cudaStream_t stream)
{
  {
    std::lock_guard lock{streams_mutex};
    free_streams.push_back(stream);
  }
  stream_returned.notify_one();
}

std::vector<tutti::IoRequest> runtime::impl::to_tutti(std::span<io_request const> requests)
{
  std::lock_guard lock{handles_mutex};
  std::vector<tutti::IoRequest> converted;
  converted.reserve(requests.size());
  for (auto const& request : requests) {
    auto const memory = memories.find(request.memory_id);
    auto const file   = files.find(request.file_id);
    if (memory == memories.end() || file == files.end()) {
      throw std::invalid_argument("flowcsd_tutti_bridge: unknown memory id " +
                                  std::to_string(request.memory_id) + " or file id " +
                                  std::to_string(request.file_id));
    }
    auto const direction = request.direction == io_direction::read ? tutti::IoDirection::READ
                                                                    : tutti::IoDirection::WRITE;
    converted.push_back(tutti::IoRequest{direction,
                                         memory->second,
                                         request.memory_offset,
                                         file->second,
                                         request.file_offset,
                                         request.bytes});
  }
  return converted;
}

void runtime::impl::finish(pending_submit const& submit, std::string& first_error)
{
  auto const waited = storage->wait(submit.io, kWaitTimeoutMs);
  if (waited.observation_status.code() == tutti::StatusCode::TIMEOUT) {
    // The NVMe may still DMA into memory the caller reuses as soon as we return
    // or throw, so neither is safe: stop the process loudly instead.
    std::fprintf(stderr,
                 "flowcsd_tutti_bridge: a Tutti submit did not complete within %llu ms\n",
                 static_cast<unsigned long long>(kWaitTimeoutMs));
    std::abort();
  }
  if (!waited.observation_status.ok()) {
    keep_first(first_error, status_text("wait", waited.observation_status));
  } else if (!waited.result.has_value()) {
    keep_first(first_error, "Tutti wait returned no terminal result");
  } else if (waited.result->state != tutti::IoState::COMPLETED || !waited.result->status.ok()) {
    auto const& detail = waited.result->detail;
    keep_first(first_error,
               status_text("I/O", waited.result->status) + " (failure kind " +
                 std::to_string(static_cast<int>(detail.failure_kind)) + ", NVMe CQ status " +
                 std::to_string(detail.raw_cq_status) + ")");
  }
  if (auto const released = storage->release_io(submit.io); !released.ok()) {
    keep_first(first_error, status_text("release_io", released));
  }
}

runtime::runtime(std::string const& config_path) : _impl(std::make_unique<impl>())
{
  auto created = tutti::TuttiRuntime::create(config_path);
  if (!created.ok()) { throw_status("runtime create from " + config_path, created.status()); }
  _impl->owner   = std::move(created).value();
  _impl->storage = _impl->owner->storage_runtime();
  if (_impl->storage == nullptr) {
    throw std::runtime_error("flowcsd_tutti_bridge: " + config_path +
                             " did not assemble a StorageRuntime");
  }
  _impl->accel_id = _impl->storage->accel_id();
  _impl->create_streams();
  // Pair the clocks before the first traced I/O; write_io_trace pairs them again.
  if (tutti::io_trace::enabled()) {
    _impl->clock_start = tutti::io_trace::sample_clock(_impl->accel_id);
  }
}

runtime::~runtime() = default;

int runtime::accel_id() const noexcept { return _impl->accel_id; }

std::uint64_t runtime::block_bytes() const noexcept { return kBlockBytes; }

opened_file runtime::open_file(std::string const& path)
{
  auto opened = _impl->storage->open("file://" + path, tutti::OpenOptions{});
  if (!opened.ok()) { throw_status("open " + path, opened.status()); }
  auto const target = opened.value();
  auto const info   = _impl->storage->query_target(target);
  if (!info.ok()) {
    (void)_impl->storage->close(target);
    throw_status("query_target " + path, info.status());
  }
  std::lock_guard lock{_impl->handles_mutex};
  auto const id = _impl->next_id++;
  _impl->files.emplace(id, target);
  return opened_file{id, info.value().logical_size};
}

void runtime::close_file(std::uint32_t file_id)
{
  tutti::TargetHandle target;
  {
    std::lock_guard lock{_impl->handles_mutex};
    auto const it = _impl->files.find(file_id);
    if (it == _impl->files.end()) {
      throw std::invalid_argument("flowcsd_tutti_bridge: unknown file id " +
                                  std::to_string(file_id));
    }
    target = it->second;
    _impl->files.erase(it);
  }
  if (auto const status = _impl->storage->close(target); !status.ok()) {
    throw_status("close", status);
  }
}

std::uint32_t runtime::register_device_memory(void* address, std::uint64_t bytes)
{
  auto registered = _impl->storage->register_memory(tutti::MemoryView{address,
                                                                      bytes,
                                                                      tutti::MemoryKind::DEVICE,
                                                                      tutti::MemoryOwnership::CALLER_OWNED,
                                                                      _impl->accel_id,
                                                                      TUTTI_COMPILED_ACCELERATOR_PROFILE});
  if (!registered.ok()) { throw_status("register_memory", registered.status()); }
  std::lock_guard lock{_impl->handles_mutex};
  auto const id = _impl->next_id++;
  _impl->memories.emplace(id, registered.value());
  return id;
}

void runtime::unregister_device_memory(std::uint32_t memory_id)
{
  tutti::MemoryHandle memory;
  {
    std::lock_guard lock{_impl->handles_mutex};
    auto const it = _impl->memories.find(memory_id);
    if (it == _impl->memories.end()) {
      throw std::invalid_argument("flowcsd_tutti_bridge: unknown memory id " +
                                  std::to_string(memory_id));
    }
    memory = it->second;
    _impl->memories.erase(it);
  }
  if (auto const status = _impl->storage->unregister_memory(memory); !status.ok()) {
    throw_status("unregister_memory", status);
  }
}

void runtime::submit_and_wait(std::span<io_request const> requests, submit_trace const& trace)
{
  using clock          = std::chrono::steady_clock;
  auto const converted = _impl->to_tutti(requests);
  std::deque<pending_submit> pending;
  std::string first_error;
  // Wait for one submit; when traced, stamp when the host saw its completion
  // and close its launch -> completion NVTX range.
  auto const complete = [&](pending_submit const& submit) {
    _impl->finish(submit, first_error);
    if (trace.timings != nullptr) { (*trace.timings)[submit.timing_index].completed = clock::now(); }
    if (trace.nvtx_device != nullptr) { nvtx_end(submit.device_range); }
  };
  // Tutti labels the batches these submits launch with this thread's tag (its
  // DataPath submit runs on the calling thread); cleared on every way out.
  tutti::io_trace::set_thread_tag(_impl->kind_tag(trace.kind));
  struct clear_tag {
    ~clear_tag() { tutti::io_trace::set_thread_tag(0); }
  } const tag_guard{};
  for (auto const submit : split_into_submits(converted)) {
    submit_timing timing;
    timing.stream_wait_begin = clock::now();
    // Never block for a stream while holding one: reuse this call's oldest
    // stream instead, so two callers that each hold streams cannot deadlock.
    cudaStream_t stream = pending.empty()
                            ? _impl->acquire_stream(timing.waited_for_stream, trace.nvtx_stream_wait)
                            : _impl->try_acquire_stream();
    if (stream == nullptr) {
      complete(pending.front());
      stream = pending.front().stream;
      pending.pop_front();
    }
    // The submit just finished may have failed; nothing more is submitted then.
    if (!first_error.empty()) {
      _impl->release_stream(stream);
      break;
    }

    // Host part of the submit: Tutti stages the entries and launches the I/O kernel.
    timing.requests = static_cast<std::uint32_t>(submit.size());
    for (auto const& request : submit) {
      timing.bytes += request.length;
    }
    timing.submit_begin = clock::now();
    nvtxRangeId_t const submit_range =
      trace.nvtx_submit != nullptr ? nvtx_start(trace.nvtx_submit) : 0;
    tutti::HostSubmitContext const context{
      tutti::ExecutionDomain::DEVICE_EXECUTION, _impl->accel_id, stream};
    auto const outcome = _impl->storage->submit(submit.data(), submit.size(), context);
    if (trace.nvtx_submit != nullptr) { nvtx_end(submit_range); }
    timing.launched = clock::now();
    if (outcome.io.has_value()) {
      pending_submit accepted{*outcome.io, stream};
      if (trace.timings != nullptr) {
        accepted.timing_index = trace.timings->size();
        trace.timings->push_back(timing);
      }
      if (trace.nvtx_device != nullptr) { accepted.device_range = nvtx_start(trace.nvtx_device); }
      pending.push_back(accepted);
    } else {
      _impl->release_stream(stream);
    }
    // Tutti leaves rejected requests to the caller (partial-commit contract).
    // Submits stay under the DataPath limits, so a rejection means a bad request
    // (alignment, bounds, registration), not back-pressure: report, do not retry.
    for (auto const& state : outcome.initial_states) {
      if (state.state == tutti::IoRequestState::REJECTED) {
        keep_first(first_error, status_text("submit", state.status));
        break;
      }
    }
    if (!first_error.empty()) { break; }
  }

  // Drain everything in flight, also after a failure: the caller reuses its
  // memory as soon as this returns or throws.
  while (!pending.empty()) {
    complete(pending.front());
    _impl->release_stream(pending.front().stream);
    pending.pop_front();
  }
  if (!first_error.empty()) { throw std::runtime_error(first_error); }
}

std::uint32_t runtime::impl::kind_tag(char const* kind)
{
  if (kind == nullptr || !tutti::io_trace::enabled()) { return 0; }
  std::lock_guard lock{trace_mutex};
  auto const found = std::find(kinds.begin(), kinds.end(), kind);
  if (found != kinds.end()) { return static_cast<std::uint32_t>(found - kinds.begin()) + 1; }
  kinds.emplace_back(kind);
  return static_cast<std::uint32_t>(kinds.size());
}

bool runtime::io_trace_enabled() { return tutti::io_trace::enabled(); }

namespace {

/// Maps %globaltimer to ns since the query start: the line through two clock
/// pairings, or an offset when only one is valid.  Deltas are taken in integers
/// first, since globaltimer values (~1e18) lose ns precision as doubles.
struct gpu_clock_fit {
  std::int64_t host_ref_ns{0};  ///< steady_clock ns of the reference pairing, minus the query start
  std::uint64_t gpu_ref_ns{0};
  double scale{1.0};            ///< host ns per GPU ns
  bool valid{false};

  /// Query-relative ns of a GPU stamp; "" for 0 (a step not reached).
  [[nodiscard]] std::string operator()(std::uint64_t gpu_ns) const
  {
    if (gpu_ns == 0 || !valid) { return {}; }
    auto const delta = static_cast<double>(static_cast<std::int64_t>(gpu_ns - gpu_ref_ns));
    return std::to_string(host_ref_ns + static_cast<std::int64_t>(delta * scale));
  }
};

gpu_clock_fit fit_gpu_clock(tutti::io_trace::ClockSample const& start,
                            tutti::io_trace::ClockSample const& end,
                            std::int64_t query_start_ns)
{
  gpu_clock_fit fit;
  auto const& reference = start.valid ? start : end;
  if (!reference.valid) { return fit; }
  fit.valid       = true;
  fit.host_ref_ns = reference.host_ns - query_start_ns;
  fit.gpu_ref_ns  = reference.gpu_ns;
  if (start.valid && end.valid && end.gpu_ns > start.gpu_ns) {
    fit.scale = static_cast<double>(end.host_ns - start.host_ns) /
                static_cast<double>(end.gpu_ns - start.gpu_ns);
  }
  return fit;
}

/// Query-relative ns of a host steady_clock stamp.
std::int64_t host_rel(std::uint64_t host_ns, std::int64_t query_start_ns)
{
  return static_cast<std::int64_t>(host_ns) - query_start_ns;
}

std::string clock_sample_json(tutti::io_trace::ClockSample const& sample, std::int64_t query_start_ns)
{
  return std::format(
    "{{\"valid\": {}, \"host_ns\": {}, \"gpu_ns\": {}, \"uncertainty_ns\": {}}}",
    sample.valid ? "true" : "false",
    sample.valid ? sample.host_ns - query_start_ns : 0,
    sample.gpu_ns,
    sample.uncertainty_ns);
}

void write_text(std::string const& path, std::string const& text)
{
  std::ofstream file(path, std::ios::trunc);
  file << text;
  if (!file) { std::fprintf(stderr, "flowcsd_tutti_bridge: could not write %s\n", path.c_str()); }
}

}  // namespace

void runtime::write_io_trace(std::string const& base, std::int64_t query_start_ns)
{
  if (!tutti::io_trace::enabled()) { return; }
  std::vector<tutti::io_trace::CommandRecord> commands;
  std::vector<tutti::io_trace::WorkerRecord> workers;
  std::vector<tutti::io_trace::BatchRecord> batches;
  tutti::io_trace::drain(commands, workers, batches);
  auto const clock_end = tutti::io_trace::sample_clock(_impl->accel_id);
  std::vector<std::string> kinds;
  tutti::io_trace::ClockSample clock_start;
  {
    std::lock_guard lock{_impl->trace_mutex};
    kinds       = _impl->kinds;
    clock_start = _impl->clock_start;
    // The next query's records are bracketed by this pairing and the next one.
    _impl->clock_start = clock_end;
  }
  auto const to_query_ns = fit_gpu_clock(clock_start, clock_end, query_start_ns);

  // One row per NVMe command.
  std::string text =
    "batch,worker,device,direction,bytes,status,smid,warpid,"
    "t_entry_ns,t_published_ns,t_cqe_ns,t_reclaimed_ns\n";
  text.reserve(text.size() + 110 * commands.size());
  for (auto const& c : commands) {
    std::format_to(std::back_inserter(text),
                   "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                   c.batch, c.worker, c.device, c.direction, c.bytes, c.status, c.smid,
                   c.warpid, to_query_ns(c.t_entry), to_query_ns(c.t_published),
                   to_query_ns(c.t_cqe), to_query_ns(c.t_reclaimed));
  }
  write_text(base + ".tutti_commands.csv", text);

  // One row per I/O-kernel thread (one-warp blocks).
  text = "batch,worker,block,smid,warpid,commands,t_start_ns,t_end_ns\n";
  text.reserve(text.size() + 70 * workers.size());
  for (auto const& w : workers) {
    std::format_to(std::back_inserter(text),
                   "{},{},{},{},{},{},{},{}\n",
                   w.batch, w.worker, w.block, w.smid, w.warpid, w.commands,
                   to_query_ns(w.t_start), to_query_ns(w.t_end));
  }
  write_text(base + ".tutti_workers.csv", text);

  // One row per I/O-kernel launch; kind is the caller's label (submit_trace::kind).
  text =
    "batch,kind,stream,commands,bytes,blocks,threads_per_block,workers,pool_workers,"
    "host_submit_ns,host_launch_ns,host_launched_ns,host_complete_ns\n";
  for (auto const& b : batches) {
    auto const kind = b.tag > 0 && b.tag <= kinds.size() ? kinds[b.tag - 1] : std::string{};
    std::format_to(std::back_inserter(text),
                   "{},{},{:#x},{},{},{},{},{},{},{},{},{},{}\n",
                   b.batch, kind, b.stream, b.commands, b.bytes, b.blocks, b.threads_per_block,
                   b.workers, b.pool_workers, host_rel(b.host_submit_ns, query_start_ns),
                   host_rel(b.host_launch_ns, query_start_ns),
                   host_rel(b.host_launched_ns, query_start_ns),
                   host_rel(b.host_complete_ns, query_start_ns));
  }
  write_text(base + ".tutti_batches.csv", text);

  // How GPU stamps became query-relative ns, so they can be checked or redone.
  write_text(base + ".tutti_clock.json",
             std::format("{{\"query_start_steady_ns\": {}, \"start\": {}, \"end\": {}, "
                         "\"fit\": {{\"valid\": {}, \"host_ref_ns\": {}, \"gpu_ref_ns\": {}, "
                         "\"scale\": {:.12f}}}, \"globaltimer_step_ns\": {}, \"sm_count\": {}, "
                         "\"records\": {{\"commands\": {}, \"workers\": {}, \"batches\": {}}}}}\n",
                         query_start_ns,
                         clock_sample_json(clock_start, query_start_ns),
                         clock_sample_json(clock_end, query_start_ns),
                         to_query_ns.valid ? "true" : "false",
                         to_query_ns.host_ref_ns,
                         to_query_ns.gpu_ref_ns,
                         to_query_ns.scale,
                         clock_end.globaltimer_step_ns,
                         clock_end.sm_count,
                         commands.size(),
                         workers.size(),
                         batches.size()));
}

}  // namespace flowcsd_tutti_bridge
