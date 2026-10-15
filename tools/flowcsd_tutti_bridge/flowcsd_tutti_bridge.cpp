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

#include "flowcsd_tutti_bridge.hpp"

#include <tutti/accelerator_device_guard.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/status.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
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

/// An accepted Tutti submit and the stream its I/O kernel runs on.
struct pending_submit {
  tutti::IoHandle io;
  cudaStream_t stream;
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
  cudaStream_t acquire_stream();
  /// nullptr when every stream is in use.
  cudaStream_t try_acquire_stream();
  void release_stream(cudaStream_t stream);

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

cudaStream_t runtime::impl::acquire_stream()
{
  std::unique_lock lock{streams_mutex};
  stream_returned.wait(lock, [this] { return !free_streams.empty(); });
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

void runtime::submit_and_wait(std::span<io_request const> requests)
{
  auto const converted = _impl->to_tutti(requests);
  std::deque<pending_submit> pending;
  std::string first_error;
  for (auto const submit : split_into_submits(converted)) {
    // Never block for a stream while holding one: reuse this call's oldest
    // stream instead, so two callers that each hold streams cannot deadlock.
    cudaStream_t stream = pending.empty() ? _impl->acquire_stream() : _impl->try_acquire_stream();
    if (stream == nullptr) {
      _impl->finish(pending.front(), first_error);
      stream = pending.front().stream;
      pending.pop_front();
    }
    // The submit just finished may have failed; nothing more is submitted then.
    if (!first_error.empty()) {
      _impl->release_stream(stream);
      break;
    }

    tutti::HostSubmitContext const context{
      tutti::ExecutionDomain::DEVICE_EXECUTION, _impl->accel_id, stream};
    auto const outcome = _impl->storage->submit(submit.data(), submit.size(), context);
    if (outcome.io.has_value()) {
      pending.push_back(pending_submit{*outcome.io, stream});
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
    _impl->finish(pending.front(), first_error);
    _impl->release_stream(pending.front().stream);
    pending.pop_front();
  }
  if (!first_error.empty()) { throw std::runtime_error(first_error); }
}

}  // namespace flowcsd_tutti_bridge
