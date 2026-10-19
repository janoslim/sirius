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

#include "io/kvikio/kvikio_context.hpp"

#include "flowcsd/flowcsd_stats.hpp"

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::io {

namespace {

constexpr std::string_view k_s3_scheme = "s3://";

[[nodiscard]] bool is_s3_uri(std::string_view path) noexcept
{
  return path.size() > k_s3_scheme.size() && path.compare(0, k_s3_scheme.size(), k_s3_scheme) == 0;
}

const kvikio_object& as_kvikio(const io_object& obj)
{
  // A mismatch is a programmer error (mixing io_objects across backends), but
  // the cast is checked so it surfaces as a clear error instead of UB.
  const auto* typed = dynamic_cast<const kvikio_object*>(&obj);
  if (typed == nullptr) {
    throw std::invalid_argument("kvikio_context: io_object '" + obj.object_path() +
                                "' was not created by this backend");
  }
  return *typed;
}

/// Bytes actually available at @p offset.  kvikIO reads are unclamped, unlike
/// the @c cudf::io::datasource this backend used to wrap, so callers past EOF
/// would otherwise depend on kvikIO's short-read behaviour.
[[nodiscard]] size_t clamp_to_object(const io_object& obj, size_t offset, size_t size) noexcept
{
  return obj.size() > offset ? std::min(size, obj.size() - offset) : 0;
}

/// kvikIO falls back to the AWS_* environment when an argument is nullopt, so
/// every credential is passed engaged — an unconfigured store is rejected in
/// create_io_object before we get here.
[[nodiscard]] std::optional<std::string> engaged(std::string const& value)
{
  return std::optional<std::string>{value};
}

}  // namespace

std::size_t kvikio_io_object::read_at(void* dst, std::size_t size, std::size_t offset) const
{
  size = clamp_to_object(*this, offset, size);
  if (size == 0) { return 0; }
  return handle().pread(dst, size, offset).get();
}

std::size_t kvikio_remote_io_object::read_at(void* dst, std::size_t size, std::size_t offset) const
{
  size = clamp_to_object(*this, offset, size);
  if (size == 0) { return 0; }
  return handle().pread(dst, size, offset).get();
}

void apply_kvikio_defaults(kvikio_config const& cfg)
{
  // Validate before touching anything so a bad config leaves kvikIO's globals
  // untouched rather than half-applied.
  if (cfg.nthreads && *cfg.nthreads == 0) {
    throw std::invalid_argument("kvikio_config: nthreads must be non-zero");
  }
  if (cfg.task_size && *cfg.task_size == 0) {
    throw std::invalid_argument("kvikio_config: task_size must be non-zero");
  }
  if (cfg.bounce_buffer_size && *cfg.bounce_buffer_size == 0) {
    throw std::invalid_argument("kvikio_config: bounce_buffer_size must be non-zero");
  }

  // Only engaged fields are pushed, so an unset field keeps whatever kvikIO
  // seeded from its environment variable.
  //
  // Order matters for the two thread-pool knobs: set the per-block-device flag
  // first so that pools created afterwards are sized by the nthreads below,
  // rather than rebuilding a global pool we are about to replace anyway.
  if (cfg.thread_pool_per_block_device) {
    kvikio::defaults::set_thread_pool_per_block_device(*cfg.thread_pool_per_block_device);
  }
  if (cfg.nthreads) { kvikio::defaults::set_thread_pool_nthreads(*cfg.nthreads); }
  if (cfg.task_size) { kvikio::defaults::set_task_size(*cfg.task_size); }
  if (cfg.gds_threshold) { kvikio::defaults::set_gds_threshold(*cfg.gds_threshold); }
  if (cfg.bounce_buffer_size) { kvikio::defaults::set_bounce_buffer_size(*cfg.bounce_buffer_size); }
  if (cfg.auto_direct_io_read) {
    kvikio::defaults::set_auto_direct_io_read(*cfg.auto_direct_io_read);
  }
  if (cfg.auto_direct_io_read_overread) {
    kvikio::defaults::set_auto_direct_io_read_overread(*cfg.auto_direct_io_read_overread);
  }
  // compat_mode is deliberately NOT set globally — it rides the FileHandle
  // constructor in create_io_object so it scopes to this ioctx's files.
}

kvikio_context::kvikio_context(kvikio_config cfg, object_store_config os)
  : _config(std::move(cfg)), _object_store(std::move(os))
{
  apply_kvikio_defaults(_config);
}

std::shared_ptr<io_object> kvikio_context::create_io_object(std::string path)
{
  if (is_s3_uri(path)) {
    if (_object_store.endpoint.empty() || _object_store.region.empty() ||
        _object_store.access_key.empty() || _object_store.secret_key.empty()) {
      throw std::runtime_error(
        "kvikio_context: cannot open '" + path +
        "': object store not configured (endpoint / region / credentials missing)");
    }
    auto bucket_and_object = kvikio::S3Endpoint::parse_s3_url(path);
    auto endpoint          = std::make_unique<kvikio::S3Endpoint>(std::move(bucket_and_object),
                                                         engaged(_object_store.region),
                                                         engaged(_object_store.access_key),
                                                         engaged(_object_store.secret_key),
                                                         engaged(_object_store.endpoint),
                                                         engaged(_object_store.session_token));
    kvikio::RemoteHandle handle{std::move(endpoint)};
    auto const object_size = handle.nbytes();
    return std::make_shared<kvikio_remote_io_object>(
      std::move(path), std::move(handle), object_size);
  }
  // Read-only: this ioctx serves the scan path only.  The handle owns the fd
  // (and any cuFile registration) for the io_object's lifetime, and the
  // io_object outlives any single datasource wrapping it.
  //
  // compat_mode is passed per handle (rather than through kvikio::defaults) so
  // it applies only to files this ioctx opens; unset falls back to kvikIO's own
  // default, which honours KVIKIO_COMPAT_MODE.
  kvikio::FileHandle handle =
    _config.compat_mode
      ? kvikio::FileHandle{path, "r", kvikio::FileHandle::m644, *_config.compat_mode}
      : kvikio::FileHandle{path, "r"};
  auto const file_size = handle.nbytes();
  return std::make_shared<kvikio_io_object>(std::move(path), std::move(handle), file_size);
}

bool kvikio_context::supports(std::string_view /*path*/) const noexcept
{
  // Universal fallback: kvikIO handles local paths, and the actual feasibility
  // check happens at create_io_object time, where opening the file may throw.
  // The registry consults this last, so an explicit backend always wins.
  return true;
}

std::vector<cudf::io::text::byte_range_info> kvikio_context::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges,
  std::optional<size_t> /*alignment*/) const noexcept
{
  return {ranges.begin(), ranges.end()};
}

size_t kvikio_context::host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  // read_at dispatches on the destination pointer type, so the same call serves
  // host and device buffers; here it is always host memory.
  flowcsd::timed_scope const io_scope{flowcsd::time_category::nvme_read};
  auto const completed = as_kvikio(obj).read_at(dst, size, offset);
  flowcsd::add_bytes(flowcsd::byte_counter::metadata_read, completed);
  flowcsd::add_bytes(flowcsd::byte_counter::nvme_read_issued, completed);
  flowcsd::add_count(flowcsd::event_counter::io_requests, 1);
  return completed;
}

exec::semi_future<size_t> kvikio_context::mixed_readv_async_io(
  const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept
{
  // make_semi_future_with invokes eagerly. Local device slices are dispatched
  // in parallel across KvikIO's thread pool via FileHandle::pread.
  return exec::make_semi_future_with([&obj, slices = std::move(slices)]() mutable -> size_t {
    flowcsd::timed_scope const io_scope{flowcsd::time_category::nvme_read};
    auto const& object = as_kvikio(obj);
    auto const* local  = dynamic_cast<kvikio_io_object const*>(&object);
    std::size_t total  = 0;

    struct pending_device_pread {
      std::size_t index;
      std::size_t expected_bytes;
      std::future<std::size_t> future;
    };
    std::vector<pending_device_pread> pending_preads;
    std::vector<cudaStream_t> synced_streams;

    for (std::size_t index = 0; index < slices.size(); ++index) {
      auto& slice = slices[index];
      try {
        auto const bytes = clamp_to_object(obj, slice.offset(), slice.size());
        slice.rng.size   = bytes;

        if (bytes == 0) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), true);
          }
          continue;
        }

        if (slice.is_fragmented()) {
          throw std::runtime_error("kvikio_context does not accept fragmented cache buffers");
        }

        if (!slice.is_contiguous() && slice.has_device_request() && local != nullptr) {
          int device_id = slice.d_buffer.device_id;
          if (device_id < 0) {
            auto const status = cudaGetDevice(&device_id);
            if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
          }
          rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

          // Drain caller's stream once before KvikIO worker threads issue synchronous cuFileRead
          // into stream-ordered pool buffers allocated on that stream.
          cudaStream_t const stream = slice.d_buffer.stream.get();
          if (std::find(synced_streams.begin(), synced_streams.end(), stream) ==
              synced_streams.end()) {
            auto const sync_status = cudaStreamSynchronize(stream);
            if (sync_status != cudaSuccess) {
              throw std::runtime_error(cudaGetErrorString(sync_status));
            }
            synced_streams.push_back(stream);
          }

          pending_preads.push_back(pending_device_pread{
            .index          = index,
            .expected_bytes = bytes,
            .future         = local->handle().pread(slice.d_buffer.data,
                                            bytes,
                                            slice.offset(),
                                            kvikio::defaults::task_size(),
                                            kvikio::defaults::gds_threshold(),
                                            /*sync_default_stream=*/false),
          });
          continue;
        }

        std::size_t completed = 0;
        if (slice.is_contiguous()) {
          auto* host = std::get<std::uint8_t*>(slice.h_buffer.buffer);
          completed  = object.read_at(host, bytes, slice.offset());
          flowcsd::add_bytes(flowcsd::byte_counter::metadata_read, completed);
          flowcsd::add_bytes(flowcsd::byte_counter::nvme_read_issued, completed);
          flowcsd::add_count(flowcsd::event_counter::io_requests, 1);

          if (completed == bytes && slice.has_device_request()) {
            int device_id = slice.d_buffer.device_id;
            if (device_id < 0) {
              auto const status = cudaGetDevice(&device_id);
              if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
            }
            rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
            auto const copy_status = cudaMemcpyAsync(
              slice.d_buffer.data, host, bytes, cudaMemcpyDefault, slice.d_buffer.stream.get());
            if (copy_status != cudaSuccess) {
              throw std::runtime_error(cudaGetErrorString(copy_status));
            }
            auto const sync_status = cudaStreamSynchronize(slice.d_buffer.stream.get());
            if (sync_status != cudaSuccess) {
              throw std::runtime_error(cudaGetErrorString(sync_status));
            }
          }
        } else if (slice.has_device_request()) {
          int device_id = slice.d_buffer.device_id;
          if (device_id < 0) {
            auto const status = cudaGetDevice(&device_id);
            if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
          }
          rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
          auto const sync_status = cudaStreamSynchronize(slice.d_buffer.stream.get());
          if (sync_status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(sync_status));
          }
          completed = object.read_at(slice.d_buffer.data, bytes, slice.offset());
          flowcsd::add_bytes(flowcsd::byte_counter::nvme_read_issued, completed);
          flowcsd::add_count(flowcsd::event_counter::io_requests, 1);
        } else {
          throw std::invalid_argument("prepared slice has no destination");
        }

        if (completed != bytes) { throw std::runtime_error("kvikio_context: short read"); }
        total += completed;

        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), true);
        }
      } catch (...) {
        for (auto& pending : pending_preads) {
          try {
            std::ignore = pending.future.get();
          } catch (...) {
          }
          auto& p_slice = slices[pending.index];
          if (p_slice.on_complete != nullptr) {
            (*p_slice.on_complete)(p_slice.h_buffer.fragments(), false);
          }
        }
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), false);
        }
        for (++index; index < slices.size(); ++index) {
          auto& skipped = slices[index];
          if (skipped.on_complete != nullptr) {
            (*skipped.on_complete)(skipped.h_buffer.fragments(), false);
          }
        }
        throw;
      }
    }

    // Collect parallel device pread completions in order.
    for (std::size_t p_idx = 0; p_idx < pending_preads.size(); ++p_idx) {
      auto& pending = pending_preads[p_idx];
      auto& slice   = slices[pending.index];
      try {
        auto const completed = pending.future.get();
        if (completed != pending.expected_bytes) {
          throw std::runtime_error("kvikio_context: short read");
        }
        flowcsd::add_bytes(flowcsd::byte_counter::nvme_read_issued, completed);
        flowcsd::add_count(flowcsd::event_counter::io_requests, 1);
        total += completed;
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), true);
        }
      } catch (...) {
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), false);
        }
        for (std::size_t rem = p_idx + 1; rem < pending_preads.size(); ++rem) {
          try {
            std::ignore = pending_preads[rem].future.get();
          } catch (...) {
          }
          auto& rem_slice = slices[pending_preads[rem].index];
          if (rem_slice.on_complete != nullptr) {
            (*rem_slice.on_complete)(rem_slice.h_buffer.fragments(), false);
          }
        }
        throw;
      }
    }
    return total;
  });
}

}  // namespace sirius::io
