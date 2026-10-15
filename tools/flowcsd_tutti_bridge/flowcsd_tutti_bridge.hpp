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

// Plain C++ facade over Tutti's userspace runtime (FlowCSD study).
//
// libflowcsd_tutti_bridge.so links Tutti's runtime, resolver and local-NVMe
// DataPath; this header exposes none of their types, so Sirius only needs this
// file and the .so (see tools/flowcsd_tutti_bridge/CMakeLists.txt).  Files and
// device-memory regions are named by small integer ids.  Every failure throws
// std::runtime_error carrying Tutti's status text.

#include <cstdint>
#include <memory>
#include <span>
#include <string>

// Not flowcsd::...: inside Sirius, `flowcsd::` already names sirius::flowcsd.
namespace flowcsd_tutti_bridge {

enum class io_direction : std::uint8_t { read, write };

/// One NVMe transfer between a registered device-memory region and an open
/// file.  Offsets and length are bytes and must be multiples of block_bytes().
struct io_request {
  io_direction direction{io_direction::read};
  std::uint32_t memory_id{0};
  std::uint64_t memory_offset{0};
  std::uint32_t file_id{0};
  std::uint64_t file_offset{0};
  std::uint64_t bytes{0};
};

struct opened_file {
  std::uint32_t id{0};
  std::uint64_t size_bytes{0};
};

class runtime {
 public:
  /// Build the Tutti runtime from its YAML: connects to tutti_daemon, leases
  /// NVMe queues and starts the local-NVMe DataPath on the configured GPU.
  explicit runtime(std::string const& config_path);

  /// Closes every file and memory region still open, then shuts Tutti down.
  ~runtime();

  runtime(runtime const&)            = delete;
  runtime& operator=(runtime const&) = delete;

  /// CUDA device the runtime drives (device memory must live there).
  [[nodiscard]] int accel_id() const noexcept;

  /// Granularity of every offset and length in an io_request, in bytes.
  [[nodiscard]] std::uint64_t block_bytes() const noexcept;

  /// Resolve a local file (FIEMAP) so it can be read or written.
  [[nodiscard]] opened_file open_file(std::string const& path);
  void close_file(std::uint32_t file_id);

  /// Register device memory on accel_id() for NVMe DMA.  The base must be
  /// 64 KiB aligned (Tutti pins GPU pages at that granularity).
  [[nodiscard]] std::uint32_t register_device_memory(void* address, std::uint64_t bytes);
  void unregister_device_memory(std::uint32_t memory_id);

  /// Submit every request and return once all of them completed.  Thread safe.
  /// On failure it still waits for everything already submitted, so the memory
  /// regions are idle when the exception reaches the caller.
  void submit_and_wait(std::span<io_request const> requests);

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace flowcsd_tutti_bridge
