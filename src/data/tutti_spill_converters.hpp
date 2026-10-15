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

// Direct GPU <-> NVMe spill for the FlowCSD study.
//
// With `sirius.flowcsd.direct_disk_spill` on, the downgrade executor sends idle GPU batches
// straight to the DISK tier (no pinned HOST copy).  The converters registered here move the
// batch buffers GPU -> NVMe with Tutti P2P writes into preallocated spill files (one directory
// per Tutti mount, created by 05_prealloc_spill.sh) and read them back with Tutti reads when a
// task restores the batch.

#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius {
namespace spill {

/// A byte range of one preallocated spill file; offset and size are multiples of 4 KiB.
struct tutti_spill_region {
  std::size_t file_index{0};      ///< position of the file in the spill store
  std::uint32_t file_id{0};       ///< Tutti id of that file
  std::uint64_t offset_bytes{0};  ///< start of the region in the file
  std::uint64_t size_bytes{0};    ///< region length; 0 for a batch without buffers
};

/**
 * @brief A GPU batch spilled through Tutti into a region of a preallocated spill file.
 *
 * Derives from disk_data_representation because the downgrade path converts to that exact type
 * and DISK-tier code (size accounting, row counts) reads its disk table.  The disk table keeps
 * cuCascade's column_metadata with buffer offsets relative to the region, data_size is the
 * region length, and file_path is a synthetic `tutti-spill://<file>#<offset>` label that never
 * exists, so the base destructor's best-effort file removal deletes nothing.  Destroying this
 * representation (after a restore, or when the batch is dropped) returns the region to the
 * spill store.
 */
class tutti_spill_representation : public cucascade::disk_data_representation {
 public:
  tutti_spill_representation(tutti_spill_region region,
                             std::vector<cucascade::memory::column_metadata> columns,
                             cucascade::memory::memory_space& disk_space);

  ~tutti_spill_representation() noexcept override;

  /// Where the batch buffers live; column offsets in get_disk_table() are relative to it.
  [[nodiscard]] tutti_spill_region const& region() const noexcept { return _region; }

 private:
  tutti_spill_region _region;
};

/**
 * @brief Route GPU <-> DISK spills through Tutti when the FlowCSD config asks for it.
 *
 * Replaces the builtin GPU -> DISK converter with one that writes the batch into a spill-store
 * region through Tutti, and adds tutti_spill_representation -> GPU for the restore.  The builtin
 * DISK -> GPU converter stays for plain cuCascade disk files (HOST -> DISK spills): the registry
 * dispatches on the exact source type, so the two never collide.  No-op unless
 * flowcsd::config().direct_disk_spill is set and a Tutti runtime is configured, so
 * flowcsd::install_config must run before this.
 *
 * @param registry The converter registry to modify (builtin converters already registered, the
 *                 registry not yet published).
 */
void register_tutti_spill_converters(cucascade::representation_converter_registry& registry);

}  // namespace spill
}  // namespace sirius
