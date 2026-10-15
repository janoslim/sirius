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

// Process-wide Tutti client shared by the scan backend (tutti_ioctx) and the
// direct spill converters.
//
// Tutti DMA can only target GPU memory registered for P2P, and the RTX A4000
// exposes a 256 MiB BAR1 window, so arbitrary cuDF/RMM buffers cannot be
// registered.  The service registers a fixed set of staging slabs once, moves
// data NVMe <-> slab with batched Tutti requests (GPU-initiated SQE/doorbell/CQ
// polling) and slab <-> caller buffer with device-to-device copies.  Every NVMe
// transfer is widened to whole blocks; the widening is reported as issued bytes.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <span>
#include <string>

namespace sirius::io::tutti {

/// Copy [file_offset, file_offset + bytes) of a Tutti file into device memory at dst.
struct device_read {
  std::uint64_t file_offset{0};
  std::uint64_t bytes{0};
  void* dst{nullptr};
};

/// Write bytes from device memory src to [file_offset, file_offset + bytes).
/// file_offset must be block aligned.  The tail of the last block is padded with
/// whatever the slab holds, so writers must leave the padding unused (the spill
/// layout aligns every buffer to a block).
struct device_write {
  std::uint64_t file_offset{0};
  std::uint64_t bytes{0};
  void const* src{nullptr};
};

/// What one read()/write() call moved.
struct transfer_stats {
  std::uint64_t requested_bytes{0};  ///< sum of the caller's ranges
  std::uint64_t issued_bytes{0};     ///< block-aligned bytes sent to NVMe
  std::uint64_t requests{0};         ///< Tutti requests submitted
};

/// Which slab set a transfer uses, so a scan never waits on spill traffic.
enum class staging_pool { scan, spill };

class tutti_service {
 public:
  /// The service built from flowcsd::config() on first use.  Throws when
  /// `tutti_runtime_config` is empty or the runtime cannot be created.
  static tutti_service& instance();

  /// True when flowcsd::config() names a Tutti runtime (cheap; does not connect).
  [[nodiscard]] static bool configured() noexcept;

  virtual ~tutti_service() = default;

  /// NVMe transfer granularity in bytes (offsets and sizes are widened to it).
  [[nodiscard]] virtual std::uint64_t block_bytes() const noexcept = 0;

  /// Resolve a file once (FIEMAP) and return its id.  Opening an already open
  /// path returns the same id and adds a reference; close_file drops one.
  [[nodiscard]] virtual std::uint32_t open_file(std::string const& path) = 0;
  virtual void close_file(std::uint32_t file_id) = 0;

  /// Size in bytes of an open file.
  [[nodiscard]] virtual std::uint64_t file_size(std::uint32_t file_id) const = 0;

  /// Read every range into device memory.  Returns after all bytes are in the
  /// destinations; the staging copies run on `stream`, which is synchronized
  /// before returning.  Thread safe.
  virtual transfer_stats read(std::uint32_t file_id,
                              std::span<device_read const> reads,
                              staging_pool pool,
                              cudaStream_t stream) = 0;

  /// Write every range from device memory.  Returns after NVMe completed every
  /// request.  Thread safe.
  virtual transfer_stats write(std::uint32_t file_id,
                               std::span<device_write const> writes,
                               staging_pool pool,
                               cudaStream_t stream) = 0;
};

}  // namespace sirius::io::tutti
