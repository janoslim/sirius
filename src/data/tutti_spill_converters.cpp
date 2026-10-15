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

// Tutti spill store and the GPU <-> DISK converters built on it (see the header).
//
// On-disk layout of one spilled batch: a region of a preallocated spill file that holds, for
// every column in pre-order (column, then its children), the null mask and then the data buffer,
// each starting on a 4 KiB boundary of the region.  That is the layout cuCascade's
// convert_gpu_to_disk writes into its own batch files; the column metadata stays in memory in
// the representation's disk table.  The region is returned to the store when the representation
// is destroyed (after a restore, or when the batch is dropped while spilled).

#include "data/tutti_spill_converters.hpp"

#include "flowcsd/flowcsd_config.hpp"
#include "flowcsd/flowcsd_stats.hpp"
#include "io/tutti/tutti_service.hpp"
#include "log/logging.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/memory/disk_table.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {
namespace spill {

namespace {

// Buffers start on this boundary inside a region, and regions start and end on it, as in
// cuCascade's GPU -> DISK layout.  Tutti widens every write to whole NVMe blocks, so this must be
// a multiple of its block size (checked when the store opens); otherwise the padding of one
// buffer's last block would overwrite the next buffer.
constexpr std::size_t SPILL_ALIGNMENT_BYTES = cucascade::DISK_FILE_ALIGNMENT;

// Extension of the files 05_prealloc_spill.sh creates in every spill dir.
constexpr char const* SPILL_FILE_EXTENSION = ".spill";

// =============================================================================
// Spill store: regions of the preallocated spill files
// =============================================================================

/// The `*.spill` files of one spill dir, sorted so every run lays the store out the same way.
std::vector<std::string> list_spill_files(std::string const& spill_dir)
{
  std::vector<std::string> paths;
  for (auto const& entry : std::filesystem::directory_iterator(spill_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == SPILL_FILE_EXTENSION) {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

/// Cut `bytes` from the lowest-offset free range that can hold them; returns their offset.
std::optional<std::uint64_t> take_first_fit(std::map<std::uint64_t, std::uint64_t>& free_ranges,
                                            std::uint64_t bytes)
{
  for (auto range = free_ranges.begin(); range != free_ranges.end(); ++range) {
    auto const [offset_bytes, length_bytes] = *range;  // copies: the entry is erased below
    if (length_bytes < bytes) { continue; }
    free_ranges.erase(range);
    if (length_bytes > bytes) { free_ranges.emplace(offset_bytes + bytes, length_bytes - bytes); }
    return offset_bytes;
  }
  return std::nullopt;
}

/**
 * @brief Process-wide allocator of regions in the preallocated spill files.
 *
 * Opens every `*.spill` file of each flowcsd spill dir once through Tutti (on the first spill)
 * and keeps a first-fit free list per file.  Successive batches start on the next spill dir
 * (round robin, one dir per NVMe drive), so spill traffic alternates between the drives; a full
 * drive passes the batch on to the others.
 */
class tutti_spill_store {
 public:
  /// The store; built on first use and never destroyed (see the definition).
  static tutti_spill_store& instance();

  /// Reserve a region of `bytes` (a multiple of SPILL_ALIGNMENT_BYTES).  Throws when full.
  tutti_spill_region allocate(std::uint64_t bytes);

  /// Return a region to its file's free list, merging it with adjacent free ranges.
  void release(tutti_spill_region const& region) noexcept;

  [[nodiscard]] std::string const& file_path(std::size_t file_index) const
  {
    return _files[file_index].path;
  }

 private:
  struct spill_file {
    std::string path;
    std::uint32_t file_id{0};
    std::size_t dir_index{0};                            ///< position of its dir in spill_dirs
    std::map<std::uint64_t, std::uint64_t> free_ranges;  ///< offset -> length, both aligned
  };

  tutti_spill_store();

  /// First fit over the files of one spill dir, in file order.  Caller holds _mutex.
  std::optional<tutti_spill_region> allocate_in_dir(std::size_t dir_index, std::uint64_t bytes);

  std::vector<spill_file> _files;  ///< fixed after construction; free_ranges guarded by _mutex
  std::size_t _dir_count{0};
  std::uint64_t _capacity_bytes{0};

  std::mutex _mutex;
  std::size_t _next_dir_index{0};
  std::uint64_t _used_bytes{0};
};

tutti_spill_store& tutti_spill_store::instance()
{
  // Leaked on purpose: a spilled batch may be released during process teardown, after
  // function-local statics are destroyed.
  static tutti_spill_store* const store = new tutti_spill_store();
  return *store;
}

tutti_spill_store::tutti_spill_store()
{
  auto& service = io::tutti::tutti_service::instance();
  if (SPILL_ALIGNMENT_BYTES % service.block_bytes() != 0) {
    throw std::runtime_error(
      std::format("tutti spill: Tutti block size {} B does not divide the {} B spill alignment",
                  service.block_bytes(),
                  SPILL_ALIGNMENT_BYTES));
  }

  // Open every spill file once; the whole file (rounded down to the alignment) starts free.
  auto const& spill_dirs = flowcsd::config().spill_dirs;
  if (spill_dirs.empty()) {
    throw std::runtime_error("tutti spill: direct_disk_spill needs sirius.flowcsd.spill_dirs");
  }
  for (std::size_t dir_index = 0; dir_index < spill_dirs.size(); ++dir_index) {
    auto const paths = list_spill_files(spill_dirs[dir_index]);
    // An empty dir would silently put all spill traffic on the other drives.
    if (paths.empty()) {
      throw std::runtime_error(
        std::format("tutti spill: no *{} files in {} (create them with 05_prealloc_spill.sh)",
                    SPILL_FILE_EXTENSION,
                    spill_dirs[dir_index]));
    }
    for (auto const& path : paths) {
      spill_file file{.path = path, .file_id = service.open_file(path), .dir_index = dir_index};
      auto const usable_bytes =
        rmm::align_down(service.file_size(file.file_id), SPILL_ALIGNMENT_BYTES);
      if (usable_bytes > 0) { file.free_ranges.emplace(0, usable_bytes); }
      _capacity_bytes += usable_bytes;
      _files.push_back(std::move(file));
    }
  }
  _dir_count = spill_dirs.size();
  SIRIUS_LOG_INFO("[flowcsd] tutti spill store: {} files, {} bytes in {} spill dirs",
                  _files.size(),
                  _capacity_bytes,
                  _dir_count);
}

tutti_spill_region tutti_spill_store::allocate(std::uint64_t bytes)
{
  // A batch without buffers needs no file space.
  if (bytes == 0) { return tutti_spill_region{}; }

  std::lock_guard lock(_mutex);
  auto const first_dir_index = _next_dir_index;
  _next_dir_index            = (_next_dir_index + 1) % _dir_count;
  for (std::size_t step = 0; step < _dir_count; ++step) {
    auto const dir_index = (first_dir_index + step) % _dir_count;
    if (auto const region = allocate_in_dir(dir_index, bytes)) {
      _used_bytes += bytes;
      return *region;
    }
  }
  throw std::runtime_error(
    std::format("tutti spill: no free region of {} bytes ({} of {} spill bytes in use); create "
                "more or larger spill files with 05_prealloc_spill.sh",
                bytes,
                _used_bytes,
                _capacity_bytes));
}

std::optional<tutti_spill_region> tutti_spill_store::allocate_in_dir(std::size_t dir_index,
                                                                     std::uint64_t bytes)
{
  for (std::size_t file_index = 0; file_index < _files.size(); ++file_index) {
    auto& file = _files[file_index];
    if (file.dir_index != dir_index) { continue; }
    if (auto const offset_bytes = take_first_fit(file.free_ranges, bytes)) {
      return tutti_spill_region{.file_index   = file_index,
                                .file_id      = file.file_id,
                                .offset_bytes = *offset_bytes,
                                .size_bytes   = bytes};
    }
  }
  return std::nullopt;
}

void tutti_spill_store::release(tutti_spill_region const& region) noexcept
{
  if (region.size_bytes == 0) { return; }

  std::lock_guard lock(_mutex);
  _used_bytes -= region.size_bytes;
  auto& free_ranges = _files[region.file_index].free_ranges;
  auto start_bytes  = region.offset_bytes;
  auto end_bytes    = region.offset_bytes + region.size_bytes;

  // Absorb the free range that starts where the region ends.
  if (auto next = free_ranges.find(end_bytes); next != free_ranges.end()) {
    end_bytes += next->second;
    free_ranges.erase(next);
  }
  // Absorb the free range that ends where the region starts.
  if (auto following = free_ranges.lower_bound(start_bytes); following != free_ranges.begin()) {
    auto const previous = std::prev(following);
    if (previous->first + previous->second == start_bytes) {
      start_bytes = previous->first;
      free_ranges.erase(previous);
    }
  }
  free_ranges.emplace(start_bytes, end_bytes - start_bytes);
}

}  // namespace

// =============================================================================
// tutti_spill_representation
// =============================================================================

tutti_spill_representation::tutti_spill_representation(
  tutti_spill_region region,
  std::vector<cucascade::memory::column_metadata> columns,
  cucascade::memory::memory_space& disk_space)
  : cucascade::disk_data_representation(
      std::make_unique<cucascade::memory::disk_table_allocation>(
        std::format("tutti-spill://{}#{}",
                    tutti_spill_store::instance().file_path(region.file_index),
                    region.offset_bytes),
        std::move(columns),
        region.size_bytes),
      disk_space),
    _region(region)
{
}

tutti_spill_representation::~tutti_spill_representation() noexcept
{
  tutti_spill_store::instance().release(_region);
}

namespace {

// =============================================================================
// GPU -> DISK
// =============================================================================

/// Place a buffer of `size_bytes` at the next aligned offset; returns its offset (0 when empty).
std::size_t place_buffer(std::size_t size_bytes, std::size_t& cursor_bytes)
{
  if (size_bytes == 0) { return 0; }
  cursor_bytes            = rmm::align_up(cursor_bytes, SPILL_ALIGNMENT_BYTES);
  auto const offset_bytes = cursor_bytes;
  cursor_bytes += size_bytes;
  return offset_bytes;
}

/// Bytes of the column's own data buffer: STRING chars or fixed-width elements.  Nested types and
/// dictionaries keep their payload in children (as in cuCascade's plan_column_copy).
std::size_t data_buffer_bytes(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  switch (column.type().id()) {
    case cudf::type_id::STRING:
      if (column.size() == 0 || column.num_children() == 0 || column.data<char>() == nullptr) {
        return 0;
      }
      // Reads the last offset on `stream`; handles INT32 and INT64 offsets.
      return static_cast<std::size_t>(cudf::strings_column_view(column).chars_size(stream));
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::EMPTY: return 0;
    default: return static_cast<std::size_t>(column.size()) * cudf::size_of(column.type());
  }
}

/**
 * @brief Plan one column (null mask, data buffer, then children) from `cursor_bytes` on.
 *
 * Gives the column_metadata cuCascade's convert_gpu_to_disk records (plan_column_copy followed
 * by recompute_file_offsets), with offsets relative to the start of the region.
 */
cucascade::memory::column_metadata plan_spill_column(cudf::column_view const& column,
                                                     std::size_t& cursor_bytes,
                                                     ::cuda::stream_ref stream)
{
  // Buffers are written from their base pointers, so a sliced view would spill the wrong rows.
  if (column.offset() != 0) {
    throw std::invalid_argument("tutti spill: sliced column views are not supported");
  }

  cucascade::memory::column_metadata meta{};
  meta.type_id    = static_cast<int32_t>(column.type().id());
  meta.num_rows   = column.size();
  meta.null_count = column.null_count();
  meta.scale      = cudf::is_fixed_point(column.type()) ? column.type().scale() : 0;

  meta.has_null_mask = column.nullable();
  meta.null_mask_size =
    meta.has_null_mask ? cudf::bitmask_allocation_size_bytes(column.size()) : 0;
  meta.null_mask_offset = place_buffer(meta.null_mask_size, cursor_bytes);

  meta.data_size   = data_buffer_bytes(column, stream);
  meta.has_data    = meta.data_size > 0;
  meta.data_offset = place_buffer(meta.data_size, cursor_bytes);

  meta.children.reserve(static_cast<std::size_t>(column.num_children()));
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    meta.children.push_back(plan_spill_column(column.child(i), cursor_bytes, stream));
  }
  return meta;
}

/// Queue one Tutti write per non-empty buffer of the column tree, at file offsets.
void collect_column_writes(cudf::column_view const& column,
                           cucascade::memory::column_metadata const& meta,
                           std::uint64_t region_offset_bytes,
                           std::vector<io::tutti::device_write>& writes)
{
  if (meta.has_null_mask && meta.null_mask_size > 0) {
    writes.push_back(
      io::tutti::device_write{.file_offset = region_offset_bytes + meta.null_mask_offset,
                              .bytes       = meta.null_mask_size,
                              .src         = column.null_mask()});
  }
  if (meta.has_data && meta.data_size > 0) {
    writes.push_back(
      io::tutti::device_write{.file_offset = region_offset_bytes + meta.data_offset,
                              .bytes       = meta.data_size,
                              .src         = column.data<std::uint8_t>()});
  }
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    collect_column_writes(
      column.child(i), meta.children[static_cast<std::size_t>(i)], region_offset_bytes, writes);
  }
}

/**
 * @brief GPU -> DISK: write the batch buffers into a spill-store region with one Tutti call.
 *
 * Same layout as cuCascade's convert_gpu_to_disk, but the bytes move GPU -> Tutti staging slab ->
 * NVMe instead of GPU -> pinned host buffer -> pwrite.  Returns after NVMe completed the writes,
 * so the caller may free the source buffers.
 */
std::unique_ptr<cucascade::idata_representation> convert_gpu_to_tutti_spill(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  // Open the spill files (first spill only) before the timer: setup, not part of moving a batch.
  auto& store = tutti_spill_store::instance();
  flowcsd::timed_scope spill_timer{flowcsd::time_category::spill_write};
  nvtx3::scoped_range nvtx_range{"sirius::spill::gpu_to_tutti"};
  auto& gpu_source             = source.cast<cucascade::gpu_table_representation>();
  cudf::table_view const table = gpu_source.get_table_view();

  // Order `stream` after the kernels that produced the batch: planning reads string offsets on it.
  if (cudaEvent_t const writer_event = gpu_source.get_writer_event(); writer_event != nullptr) {
    cucascade::cuda::cuda_event_view{writer_event}.wait(stream);
  }

  // Plan the region layout.
  std::size_t layout_bytes = 0;
  std::vector<cucascade::memory::column_metadata> columns;
  columns.reserve(static_cast<std::size_t>(table.num_columns()));
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    columns.push_back(plan_spill_column(table.column(i), layout_bytes, stream));
  }

  // Take a region.  The representation owns it from here on, so a failed write frees it again.
  auto const region = store.allocate(rmm::align_up(layout_bytes, SPILL_ALIGNMENT_BYTES));
  auto spilled = std::make_unique<tutti_spill_representation>(
    region, std::move(columns), const_cast<cucascade::memory::memory_space&>(*target_memory_space));

  // Write every buffer with one Tutti call.  Tutti copies the buffers into its staging slabs, so
  // every producer must have finished, whichever stream those copies run on.
  std::vector<io::tutti::device_write> writes;
  auto const& disk_columns = spilled->get_disk_table().columns;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    collect_column_writes(
      table.column(i), disk_columns[static_cast<std::size_t>(i)], region.offset_bytes, writes);
  }
  stream.sync();
  std::uint64_t issued_bytes = 0;
  if (!writes.empty()) {
    auto const transfer = io::tutti::tutti_service::instance().write(
      region.file_id, writes, io::tutti::staging_pool::spill, stream.get());
    issued_bytes = transfer.issued_bytes;
    flowcsd::add_bytes(flowcsd::byte_counter::spill_write, transfer.issued_bytes);
    flowcsd::add_count(flowcsd::event_counter::io_requests, transfer.requests);
  }
  flowcsd::add_count(flowcsd::event_counter::spilled_batches, 1);
  // One line per batch: shows which drive (spill dir) every spill went to.
  SIRIUS_LOG_INFO("[flowcsd] spill write {} B ({} rows) to {} at {}",
                  issued_bytes,
                  table.num_rows(),
                  store.file_path(region.file_index),
                  region.offset_bytes);
  return spilled;
}

// =============================================================================
// DISK -> GPU
// =============================================================================

/// Device buffers of one spilled column tree, allocated before the single Tutti read fills them.
struct restored_buffers {
  rmm::device_buffer null_mask;
  rmm::device_buffer data;
  std::vector<restored_buffers> children;
};

/// Allocate the buffers of one column tree and queue a Tutti read into each.
restored_buffers allocate_column_buffers(cucascade::memory::column_metadata const& meta,
                                         std::uint64_t region_offset_bytes,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr,
                                         std::vector<io::tutti::device_read>& reads)
{
  restored_buffers buffers;
  if (meta.has_null_mask) {
    // create_null_mask(num_rows) allocates bitmask_allocation_size_bytes(num_rows), the size the
    // writer recorded, so the read fills the whole mask.
    buffers.null_mask =
      cudf::create_null_mask(meta.num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    if (buffers.null_mask.size() > 0) {
      reads.push_back(io::tutti::device_read{
        .file_offset = region_offset_bytes + meta.null_mask_offset,
        .bytes       = std::min(buffers.null_mask.size(), meta.null_mask_size),
        .dst         = buffers.null_mask.data()});
    }
  }
  if (meta.has_data && meta.data_size > 0) {
    buffers.data = rmm::device_buffer(meta.data_size, stream, mr);
    reads.push_back(io::tutti::device_read{.file_offset = region_offset_bytes + meta.data_offset,
                                           .bytes       = meta.data_size,
                                           .dst         = buffers.data.data()});
  }
  buffers.children.reserve(meta.children.size());
  for (auto const& child_meta : meta.children) {
    buffers.children.push_back(
      allocate_column_buffers(child_meta, region_offset_bytes, stream, mr, reads));
  }
  return buffers;
}

std::unique_ptr<cudf::column> build_restored_column(cucascade::memory::column_metadata const& meta,
                                                    restored_buffers buffers,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

/// Build every child column of `meta` from its filled buffers, in order.
std::vector<std::unique_ptr<cudf::column>> build_restored_children(
  cucascade::memory::column_metadata const& meta,
  restored_buffers& buffers,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(meta.children.size());
  for (std::size_t i = 0; i < meta.children.size(); ++i) {
    children.push_back(
      build_restored_column(meta.children[i], std::move(buffers.children[i]), stream, mr));
  }
  return children;
}

/**
 * @brief Build a cudf column from its filled buffers.
 *
 * Mirrors cuCascade's reconstruct_column_from_disk, so a batch comes back with the same column
 * types whichever spill path wrote it.
 */
std::unique_ptr<cudf::column> build_restored_column(cucascade::memory::column_metadata const& meta,
                                                    restored_buffers buffers,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto const type_id               = static_cast<cudf::type_id>(meta.type_id);
  cudf::size_type const null_count = meta.has_null_mask ? meta.null_count : 0;

  if (type_id == cudf::type_id::STRING) {
    // cudf::make_empty_column(STRING) has no offsets child; recreate it the same way.
    if (meta.children.empty() && meta.num_rows == 0) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }
    if (meta.children.empty()) {
      throw std::invalid_argument("tutti spill: STRING column metadata has no offsets child");
    }
    auto offsets =
      build_restored_column(meta.children[0], std::move(buffers.children[0]), stream, mr);
    // cuCascade widens INT32 string offsets to INT64 on restore; do the same.
    if (offsets->type().id() == cudf::type_id::INT32) {
      offsets = cudf::cast(offsets->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return cudf::make_strings_column(meta.num_rows,
                                     std::move(offsets),
                                     std::move(buffers.data),
                                     null_count,
                                     std::move(buffers.null_mask));
  }

  // Built directly, as cuCascade does: LIST offsets stay INT32 and make_lists_column's null purge
  // is skipped (the spilled null masks are already consistent).
  if (type_id == cudf::type_id::LIST || type_id == cudf::type_id::STRUCT) {
    if (type_id == cudf::type_id::LIST && meta.children.size() < 2) {
      throw std::invalid_argument(
        "tutti spill: LIST column metadata needs offsets and values children");
    }
    auto children = build_restored_children(meta, buffers, stream, mr);
    return std::make_unique<cudf::column>(cudf::data_type{type_id},
                                          meta.num_rows,
                                          rmm::device_buffer{},
                                          std::move(buffers.null_mask),
                                          null_count,
                                          std::move(children));
  }

  if (type_id == cudf::type_id::DICTIONARY32) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "tutti spill: DICTIONARY32 column metadata needs indices and keys children");
    }
    // cuDF stores [0] = indices, [1] = keys; make_dictionary_column takes (keys, indices).
    auto indices =
      build_restored_column(meta.children[0], std::move(buffers.children[0]), stream, mr);
    auto keys = build_restored_column(meta.children[1], std::move(buffers.children[1]), stream, mr);
    return cudf::make_dictionary_column(
      std::move(keys), std::move(indices), std::move(buffers.null_mask), null_count);
  }

  // Fixed-width (and EMPTY) columns own one data buffer; fixed-point types carry their scale.
  auto const dtype = cudf::is_fixed_point(cudf::data_type{type_id})
                       ? cudf::data_type{type_id, meta.scale}
                       : cudf::data_type{type_id};
  return std::make_unique<cudf::column>(
    dtype, meta.num_rows, std::move(buffers.data), std::move(buffers.null_mask), null_count);
}

/**
 * @brief DISK -> GPU: read a spilled batch back with one Tutti call and rebuild its columns.
 *
 * Leaves the source untouched; its region goes back to the store when the caller drops the
 * source representation after installing the result.
 */
std::unique_ptr<cucascade::idata_representation> convert_tutti_spill_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  flowcsd::timed_scope spill_timer{flowcsd::time_category::spill_read};
  nvtx3::scoped_range nvtx_range{"sirius::spill::tutti_to_gpu"};
  auto const& spilled = source.cast<tutti_spill_representation>();
  auto const& region  = spilled.region();
  auto const& columns = spilled.get_disk_table().columns;

  // Allocate on the target GPU from its default allocator, like cuCascade's convert_disk_to_gpu.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};
  auto mr = target_memory_space->get_default_allocator();

  // Allocate every buffer first, then read the whole batch with one Tutti call (its staging
  // copies run on `stream` and have finished when it returns).
  std::vector<io::tutti::device_read> reads;
  std::vector<restored_buffers> column_buffers;
  column_buffers.reserve(columns.size());
  for (auto const& meta : columns) {
    column_buffers.push_back(allocate_column_buffers(meta, region.offset_bytes, stream, mr, reads));
  }
  if (!reads.empty()) {
    auto const transfer = io::tutti::tutti_service::instance().read(
      region.file_id, reads, io::tutti::staging_pool::spill, stream.get());
    flowcsd::add_bytes(flowcsd::byte_counter::spill_read, transfer.issued_bytes);
    flowcsd::add_count(flowcsd::event_counter::io_requests, transfer.requests);
  }

  // Rebuild the cudf columns from the filled buffers.
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    gpu_columns.push_back(
      build_restored_column(columns[i], std::move(column_buffers[i]), stream, mr));
  }
  stream.sync();
  flowcsd::add_count(flowcsd::event_counter::restored_batches, 1);

  // The representation records a writer event on `stream` for later cross-stream readers.
  return std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(gpu_columns)),
    *const_cast<cucascade::memory::memory_space*>(target_memory_space),
    stream);
}

}  // namespace

void register_tutti_spill_converters(cucascade::representation_converter_registry& registry)
{
  auto const& config = flowcsd::config();
  if (!config.direct_disk_spill) { return; }
  // The downgrade executor still skips HOST, but DISK writes would go through cuCascade's file
  // backend instead of Tutti, which changes what a spill measures.
  if (!io::tutti::tutti_service::configured()) {
    SIRIUS_LOG_WARN(
      "[flowcsd] direct_disk_spill is set but no Tutti runtime is configured; GPU batches spill "
      "to DISK through cuCascade's file backend");
    return;
  }

  // register_converter throws on duplicate keys, so drop the builtin GPU -> DISK converter first.
  registry.unregister_converter<cucascade::gpu_table_representation,
                                cucascade::disk_data_representation>();
  registry.register_converter<cucascade::gpu_table_representation,
                              cucascade::disk_data_representation>(convert_gpu_to_tutti_spill);
  registry.register_converter<tutti_spill_representation, cucascade::gpu_table_representation>(
    convert_tutti_spill_to_gpu);

  std::string spill_dirs;
  for (auto const& dir : config.spill_dirs) {
    if (!spill_dirs.empty()) { spill_dirs += ", "; }
    spill_dirs += dir;
  }
  SIRIUS_LOG_INFO("[flowcsd] registered direct GPU <-> DISK spill through Tutti (spill dirs: {})",
                  spill_dirs);
}

}  // namespace spill
}  // namespace sirius
