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

#include "io/tutti/tutti_ioctx.hpp"

#include "flowcsd/flowcsd_config.hpp"
#include "flowcsd/flowcsd_stats.hpp"
#include "io/tutti/tutti_service.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

namespace sirius::io::tutti {

namespace {

tutti_io_object const& as_tutti(io_object const& obj)
{
  // A mismatch is a programmer error (mixing io_objects across backends), but
  // the cast is checked so it surfaces as a clear error instead of UB.
  auto const* typed = dynamic_cast<tutti_io_object const*>(&obj);
  if (typed == nullptr) {
    throw std::invalid_argument("tutti_ioctx: io_object '" + obj.object_path() +
                                "' was not created by this backend");
  }
  return *typed;
}

/// Bytes actually available at @p offset; callers may ask past the end of the file.
[[nodiscard]] size_t clamp_to_object(io_object const& obj, size_t offset, size_t size) noexcept
{
  return obj.size() > offset ? std::min(size, obj.size() - offset) : 0;
}

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("tutti_ioctx: ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

struct free_deleter {
  void operator()(void* pointer) const noexcept { std::free(pointer); }
};

/// Read [offset, offset + size) into host memory through the O_DIRECT descriptor.
///
/// O_DIRECT keeps the footer and page-index reads out of the page cache, so
/// every byte of a Tutti-served file comes from the NVMe and none is cached in
/// host DRAM.  O_DIRECT needs a block-aligned range and buffer, hence the
/// aligned bounce buffer.
size_t direct_host_read(tutti_io_object const& file, size_t offset, size_t size, std::uint8_t* dst)
{
  size = clamp_to_object(file, offset, size);
  if (size == 0) { return 0; }
  auto const aligned_begin = offset / IO_BLOCK_SIZE * IO_BLOCK_SIZE;
  auto const wanted_end    = offset + size;
  auto const aligned_end   = (wanted_end + IO_BLOCK_SIZE - 1) / IO_BLOCK_SIZE * IO_BLOCK_SIZE;
  auto const aligned_bytes = aligned_end - aligned_begin;

  void* allocation = nullptr;
  if (::posix_memalign(&allocation, IO_BLOCK_SIZE, aligned_bytes) != 0) { throw std::bad_alloc(); }
  std::unique_ptr<std::uint8_t, free_deleter> const bounce{static_cast<std::uint8_t*>(allocation)};

  // Stop as soon as the wanted bytes are in: at the end of the file pread
  // returns short, and a second pread from that unaligned position would fail
  // under O_DIRECT.
  size_t filled = 0;
  while (aligned_begin + filled < wanted_end) {
    auto const result = ::pread(file.direct_fd(),
                                bounce.get() + filled,
                                aligned_bytes - filled,
                                static_cast<off_t>(aligned_begin + filled));
    if (result < 0 && errno == EINTR) { continue; }
    if (result < 0) {
      throw std::system_error(
        errno, std::generic_category(), "tutti_ioctx: pread " + file.object_path());
    }
    if (result == 0) {
      throw std::runtime_error("tutti_ioctx: " + file.object_path() +
                               " ended before its recorded size");
    }
    filled += static_cast<size_t>(result);
  }
  // metadata_read counts what the kernel read (whole blocks), so it can be checked
  // against the block-layer counters of the Tutti SSDs.
  flowcsd::add_bytes(flowcsd::byte_counter::metadata_read, filled);
  std::memcpy(dst, bounce.get() + (offset - aligned_begin), size);
  return size;
}

/// Mirror a host slice into its device destination as well: the contract of
/// slices that carry both buffers, as in kvikio_context.
void copy_host_to_device(prepared_io_slice const& slice, std::uint8_t const* host)
{
  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { check_cuda(cudaGetDevice(&device_id), "cudaGetDevice"); }
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{device_id}};
  auto const stream = slice.d_buffer.stream.get();
  check_cuda(
    cudaMemcpyAsync(slice.d_buffer.data, host, slice.size(), cudaMemcpyDefault, stream),
    "cudaMemcpyAsync");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

/// The device destinations of one call that share a CUDA stream (one service read).
struct stream_reads {
  cudaStream_t stream{nullptr};
  std::vector<std::size_t> slice_indices;
  std::vector<device_read> reads;
};

stream_reads& reads_for_stream(std::vector<stream_reads>& groups, cudaStream_t stream)
{
  for (auto& group : groups) {
    if (group.stream == stream) { return group; }
  }
  return groups.emplace_back(stream_reads{stream, {}, {}});
}

/// One tutti_service::read, with this backend's share of the FlowCSD statistics.
void read_through_tutti(tutti_service& service, tutti_io_object const& file, stream_reads const& group)
{
  transfer_stats stats;
  {
    // nvme_read covers Tutti submit -> completion plus the slab -> destination copies.
    flowcsd::timed_scope const nvme_read{flowcsd::time_category::nvme_read};
    stats = service.read(file.file_id(), group.reads, staging_pool::scan, group.stream);
  }
  flowcsd::add_bytes(flowcsd::byte_counter::nvme_read_issued, stats.issued_bytes);
  flowcsd::add_count(flowcsd::event_counter::io_requests, stats.requests);
}

/// Read every slice and return the bytes read.  @p settle(index, true) runs as
/// soon as a slice's bytes are in place; a throw leaves the rest unsettled.
template <class Settle>
size_t read_slices(tutti_service& service,
                   tutti_io_object const& file,
                   std::vector<prepared_io_slice>& slices,
                   Settle&& settle)
{
  size_t total = 0;
  std::vector<stream_reads> device_groups;

  // Host destinations are read right away; device destinations are gathered per stream.
  for (std::size_t index = 0; index < slices.size(); ++index) {
    auto& slice    = slices[index];
    slice.rng.size = clamp_to_object(file, slice.offset(), slice.size());
    if (slice.size() == 0) {
      settle(index, true);
      continue;
    }
    if (slice.is_fragmented()) {
      throw std::runtime_error("tutti_ioctx does not accept fragmented cache buffers");
    }
    if (slice.is_contiguous()) {
      auto* host = std::get<std::uint8_t*>(slice.h_buffer.buffer);
      total += direct_host_read(file, slice.offset(), slice.size(), host);
      if (slice.has_device_request()) { copy_host_to_device(slice, host); }
      settle(index, true);
      continue;
    }
    if (!slice.has_device_request()) {
      throw std::invalid_argument("prepared slice has no destination");
    }
    auto& group = reads_for_stream(device_groups, slice.d_buffer.stream.get());
    group.slice_indices.push_back(index);
    group.reads.push_back(device_read{slice.offset(), slice.size(), slice.d_buffer.data});
  }

  // One Tutti read per stream: NVMe -> registered slabs -> destinations.
  for (auto const& group : device_groups) {
    read_through_tutti(service, file, group);
    for (auto const index : group.slice_indices) {
      total += slices[index].size();
      settle(index, true);
    }
  }
  return total;
}

}  // namespace

tutti_io_object::tutti_io_object(tutti_service& service,
                                 std::string path,
                                 std::uint32_t file_id,
                                 std::uint64_t size_bytes,
                                 file_descriptor direct_fd)
  : _service(service),
    _path(std::move(path)),
    _file_id(file_id),
    _size_bytes(size_bytes),
    _direct_fd(std::move(direct_fd))
{
}

tutti_io_object::~tutti_io_object()
{
  try {
    _service.close_file(_file_id);
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("tutti_io_object: closing {} in Tutti failed: {}", _path, e.what());
  }
}

tutti_ioctx::tutti_ioctx() : _service(tutti_service::instance()) {}

bool tutti_ioctx::claims(std::string_view path) noexcept
{
  auto const& prefixes = flowcsd::config().tutti_path_prefixes;
  return std::any_of(prefixes.begin(), prefixes.end(), [path](std::string const& prefix) {
    return path.starts_with(prefix);
  });
}

std::shared_ptr<io_object> tutti_ioctx::create_io_object(std::string path)
{
  // Host reads bypass the page cache (see direct_host_read).
  file_descriptor direct_fd{::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC)};
  if (!direct_fd) {
    throw std::system_error(errno, std::generic_category(), "tutti_ioctx: open " + path);
  }
  // The object's size is the file's own size (the Parquet footer sits at its
  // end), not Tutti's logical target size, which only bounds NVMe requests.
  struct stat file_status {};
  if (::fstat(direct_fd.get(), &file_status) != 0) {
    throw std::system_error(errno, std::generic_category(), "tutti_ioctx: fstat " + path);
  }
  auto const file_id = _service.open_file(path);
  return std::make_shared<tutti_io_object>(_service,
                                           std::move(path),
                                           file_id,
                                           static_cast<std::uint64_t>(file_status.st_size),
                                           std::move(direct_fd));
}

std::vector<cudf::io::text::byte_range_info> tutti_ioctx::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges,
  std::optional<size_t> /*alignment*/) const noexcept
{
  return {ranges.begin(), ranges.end()};
}

size_t tutti_ioctx::host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  return direct_host_read(as_tutti(obj), offset, size, dst);
}

exec::semi_future<size_t> tutti_ioctx::mixed_readv_async_io(
  const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept
{
  // make_semi_future_with runs the lambda now: a Tutti read blocks until its bytes landed.
  return exec::make_semi_future_with([this, &obj, slices = std::move(slices)]() mutable -> size_t {
    std::vector<bool> settled(slices.size(), false);
    // Every slice is reported exactly once: success when its bytes are in
    // place, failure for all slices still unsettled when the call fails.
    auto settle = [&slices, &settled](std::size_t index, bool success) {
      settled[index] = true;
      if (slices[index].on_complete != nullptr) {
        (*slices[index].on_complete)(slices[index].h_buffer.fragments(), success);
      }
    };
    try {
      return read_slices(_service, as_tutti(obj), slices, settle);
    } catch (...) {
      for (std::size_t index = 0; index < slices.size(); ++index) {
        if (!settled[index]) { settle(index, false); }
      }
      throw;
    }
  });
}

}  // namespace sirius::io::tutti
