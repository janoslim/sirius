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

// FlowCSD study knobs: the `sirius.flowcsd` block of sirius.yaml.
//
// One process runs one Sirius context, so the parsed block is installed once as
// process-wide state and read by the Tutti I/O backend, the page-skipping scan
// route, the direct spill converters and the per-query statistics.

#include <cstddef>
#include <string>
#include <vector>

namespace YAML {
class Node;
}

namespace sirius::flowcsd {

struct flowcsd_config {
  /// TuttiRuntime YAML (daemon endpoint and NVMe device group).  Empty disables
  /// every Tutti path (scan backend and direct spill).
  std::string tutti_runtime_config;

  /// Local path prefixes served by Tutti when `scan_manager.backend: tutti`.
  /// Other local paths keep the io_uring backend.
  std::vector<std::string> tutti_path_prefixes{
    "/mnt/smartssd0/", "/mnt/smartssd1/", "/mnt/tutti0/", "/mnt/tutti1/"};

  /// Size of one registered (P2P-visible) GPU staging slab.
  std::size_t staging_slab_bytes{8ull << 20};

  /// Staging slabs for scan reads and for spill writes/reads.  Their total
  /// (slabs x slab bytes) must stay inside the GPU BAR1 window (256 MiB on the
  /// RTX A4000) together with Tutti's own queue memory.
  std::size_t read_staging_slabs{12};
  std::size_t spill_staging_slabs{8};

  /// Largest single Tutti request; longer extents are split.
  std::size_t max_request_bytes{1ull << 20};

  /// Read only Parquet data pages that hold at least one surviving row
  /// (OffsetIndex) on the bulk scan route.
  bool page_skipping{true};

  /// Spill GPU batches straight to the DISK tier through Tutti (no HOST tier).
  bool direct_disk_spill{false};

  /// Directories holding the preallocated spill files (one directory per mount).
  std::vector<std::string> spill_dirs;

  /// JSON-lines file that receives one statistics record per GPU query.  The
  /// FLOWCSD_STATS_JSONL environment variable overrides it; empty disables it.
  std::string stats_jsonl;

  static void from_yaml(YAML::Node const& node, flowcsd_config& out);
};

/// Install the process-wide config; called once from SiriusContext initialization.
void install_config(flowcsd_config config);

/// The installed config, or the defaults above when none was installed.
[[nodiscard]] flowcsd_config const& config();

/// SSD index of a Tutti-served path: the position of the first tutti_path_prefixes
/// entry it starts with (0 for /mnt/tutti0/, 1 for /mnt/tutti1/), or -1.
[[nodiscard]] int tutti_device_index(std::string const& path) noexcept;

}  // namespace sirius::flowcsd
