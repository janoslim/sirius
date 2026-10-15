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

// ioctx that serves local Parquet files through Tutti (FlowCSD study).
//
// With `scan_manager.backend: tutti`, io_context_registry routes every path
// under flowcsd_config::tutti_path_prefixes here.  Device destinations travel
// NVMe -> registered GPU slab -> destination through tutti_service; host
// destinations (footer and page-index reads) use pread on an O_DIRECT
// descriptor.  The backend has no host staging and cannot carry a prefetching
// cache, so no byte of these files is cached in host DRAM.

#include "io/io_context.hpp"
#include "io/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io::tutti {

class tutti_service;

/**
 * @brief A file opened in the Tutti service, plus an O_DIRECT descriptor for
 *        host reads.  Holds one reference on the service's file entry.
 */
class tutti_io_object final : public io_object {
 public:
  tutti_io_object(tutti_service& service,
                  std::string path,
                  std::uint32_t file_id,
                  std::uint64_t size_bytes,
                  file_descriptor direct_fd);

  /// Drops this object's reference on the Tutti file.
  ~tutti_io_object() override;

  tutti_io_object(tutti_io_object const&)            = delete;
  tutti_io_object& operator=(tutti_io_object const&) = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _size_bytes; }

  [[nodiscard]] std::uint32_t file_id() const noexcept { return _file_id; }
  [[nodiscard]] int direct_fd() const noexcept { return _direct_fd.get(); }

 private:
  tutti_service& _service;
  std::string _path;
  std::uint32_t _file_id;
  std::uint64_t _size_bytes;
  file_descriptor _direct_fd;
};

/**
 * @brief Eager (no reactor) backend over the process-wide tutti_service,
 *        modeled on @c kvikio_context.
 */
class tutti_ioctx final : public ioctx {
 public:
  /// Binds tutti_service::instance(), starting the service on first use.
  /// Throws when Sirius was built without Tutti or the runtime cannot start.
  tutti_ioctx();

  ~tutti_ioctx() override
  {
    // See ioctx::pre_destroy.  No cache is ever attached, but the contract holds.
    this->pre_destroy();
  }

  /// Whether @p path lies under one of flowcsd::config().tutti_path_prefixes.
  /// Static so io_context_registry can use it as this backend's path checker.
  [[nodiscard]] static bool claims(std::string_view path) noexcept;

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::tutti; }

  void shutdown() noexcept override {}

  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    return claims(path);
  }

  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return true; }

  /// Both false, so the prefetching cache (host DRAM) never attaches to this backend.
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }

  /// True: Parquet then takes the bulk route (one vectored device read per file),
  /// which the service turns into batched Tutti submits.
  [[nodiscard]] bool prefers_bulk_io() const noexcept override { return true; }

  /// 1 and 0: callers hand over the exact ranges they need, and the service
  /// widens to NVMe blocks itself, so requested and issued bytes stay separate.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept override { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept override { return 0; }

  /// Zero: no cache to read ahead into.
  [[nodiscard]] std::size_t n_max_concurrent_scans() const noexcept override { return 0; }

  /// Ranges pass through unchanged (see min_alignment_requirement).
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> alignment) const noexcept override;

  // -- Backend primitives ---------------------------------------------------

  size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) override;

  /// Eager like kvikio_context: host slices are read with O_DIRECT pread, the
  /// device slices of one stream with a single tutti_service::read.
  exec::semi_future<size_t> mixed_readv_async_io(
    const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept override;

 protected:
  /// Open the O_DIRECT descriptor and the Tutti file (FIEMAP) for @p path.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

 private:
  tutti_service& _service;
};

}  // namespace sirius::io::tutti
