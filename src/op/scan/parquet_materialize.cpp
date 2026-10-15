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

#include "op/scan/parquet_materialize.hpp"

#include "flowcsd/flowcsd_config.hpp"
#include "flowcsd/flowcsd_stats.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"
#include "op/scan/page_survival.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace sirius::op::scan {

using hybrid_scan_reader    = cudf::io::parquet::experimental::hybrid_scan_reader;
using hybrid_scan_multifile = cudf::io::parquet::experimental::hybrid_scan_multifile;
using cudf::io::parquet::ColumnChunk;
using cudf::io::parquet::ColumnChunkMetaData;
using cudf::io::parquet::Encoding;
using cudf::io::parquet::FileMetaData;
using cudf::io::parquet::PageType;

std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& row_group_indices)
{
  if (row_group_indices.empty()) { return {}; }
  hybrid_scan_reader reader(metadata, options);
  return reader.all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size()),
    options);
}

std::vector<cudf::io::text::byte_range_info> column_chunk_range_cache::ranges(
  std::shared_ptr<cudf::io::parquet::FileMetaData const> const& metadata,
  std::shared_ptr<cudf::io::parquet_reader_options const> const& options,
  std::vector<cudf::size_type> const& row_group_indices)
{
  if (row_group_indices.empty()) { return {}; }
  if (!_reader || _metadata != metadata || _options != options) {
    _reader.reset();
    _reader   = std::make_unique<hybrid_scan_reader>(*metadata, *options);
    _metadata = metadata;
    _options  = options;
  }
  return _reader->all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size()),
    *options);
}

void column_chunk_range_cache::clear() noexcept
{
  _reader.reset();
  _metadata.reset();
  _options.reset();
}

bool prefers_bulk_materialize(std::span<parquet_source const> sources,
                              cudf::io::parquet_reader_options const& options) noexcept
{
  if (sources.empty()) { return false; }
  // Every source has to qualify: the decode consumes one flattened chunk-data
  // span covering all of them, so a single file that cannot be read this way
  // takes the whole split to the general route.
  return std::all_of(sources.begin(), sources.end(), [](parquet_source const& src) {
    return src.datasource && src.metadata && !src.row_group_indices.empty() &&
           src.datasource->prefers_bulk_io();
  });
}

namespace {

char const* physical_type_name(cudf::io::parquet::Type type)
{
  switch (type) {
    case cudf::io::parquet::Type::BOOLEAN: return "BOOLEAN";
    case cudf::io::parquet::Type::INT32: return "INT32";
    case cudf::io::parquet::Type::INT64: return "INT64";
    case cudf::io::parquet::Type::INT96: return "INT96";
    case cudf::io::parquet::Type::FLOAT: return "FLOAT";
    case cudf::io::parquet::Type::DOUBLE: return "DOUBLE";
    case cudf::io::parquet::Type::BYTE_ARRAY: return "BYTE_ARRAY";
    case cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY: return "FIXED_LEN_BYTE_ARRAY";
    default: return "group";
  }
}

std::string describe(cudf::io::parquet::SchemaElement const& element)
{
  return "'" + element.name + "' " + physical_type_name(element.type);
}

std::string path_of(parquet_source const& src)
{
  return src.datasource ? src.datasource->get_io_object().object_path() : std::string{"<unknown>"};
}

/// Per output column, what flowcsd decoded_materialized and the per-column
/// scan breakdown need.
struct column_accounting {
  std::int64_t rows_decoded       = 0;  ///< rows in the data pages cuDF decodes
  std::int64_t uncompressed_bytes = 0;  ///< encoded size of the column's chunks (width fallback)
  std::int64_t values             = 0;  ///< values in the column's chunks (width fallback)
  flowcsd::column_scan_counters scan;   ///< chunk / requested bytes and pages of this column
};

/// What one materialize call read.  Recorded into the flowcsd counters once the
/// output table exists (@ref record_scan_counters).
struct read_accounting {
  std::unordered_map<std::string, column_accounting> columns;  ///< by top-level column name

  /// Count @p chunk as selected: @p rows_decoded of its rows lie in pages cuDF
  /// decodes, @p requested_bytes of it were asked of the datasource, and
  /// @p pages_read of its @p pages_total data pages were read.
  void add_chunk(ColumnChunk const& chunk,
                 std::int64_t rows_decoded,
                 std::uint64_t requested_bytes,
                 std::uint64_t pages_total,
                 std::uint64_t pages_read)
  {
    auto const& meta = chunk.meta_data;
    auto& column     = columns[meta.path_in_schema.front()];
    column.rows_decoded += rows_decoded;
    column.uncompressed_bytes += meta.total_uncompressed_size;
    column.values += meta.num_values;
    column.scan.chunk_bytes_full += static_cast<std::uint64_t>(meta.total_compressed_size);
    column.scan.nvme_read_requested += requested_bytes;
    column.scan.pages_total += pages_total;
    column.scan.pages_read += pages_read;
  }
};

/// A column chunk the reader asked for, found in its source's footer.
struct located_chunk {
  std::size_t source;              ///< index of its source
  ColumnChunk const* chunk;        ///< its footer entry
  std::int64_t file_offset;        ///< where the chunk starts in the file
  std::size_t size;                ///< bytes of the whole chunk
  cudf::size_type first_mask_row;  ///< row-mask row of its first row
  std::int64_t num_rows;           ///< rows of its row group
};

/// Where a column chunk starts in its file: at its dictionary page when it has
/// one.  The rule the hybrid scan reader builds its byte ranges with
/// (hybrid_scan_impl.cpp:461-464), so a range's offset names its chunk.
std::int64_t chunk_start_offset(ColumnChunkMetaData const& meta)
{
  if (meta.dictionary_page_offset == 0) { return meta.data_page_offset; }
  return std::min(meta.data_page_offset, meta.dictionary_page_offset);
}

/// Find @p ranges -- the reader's byte ranges for one source's selected
/// @p row_groups -- in that source's footer.
///
/// The reader lists ranges by row group and then column without naming the
/// column; a chunk's start offset is unique within its file, so it identifies
/// the chunk.  @p first_source_row is the row-mask row of the source's first
/// selected row: the row mask spans every source's selected row groups, in
/// source order and then selection order.
std::vector<located_chunk> locate_chunks(FileMetaData const& footer,
                                         std::vector<cudf::size_type> const& row_groups,
                                         std::span<cudf::io::text::byte_range_info const> ranges,
                                         std::size_t source,
                                         cudf::size_type first_source_row)
{
  // Every selected chunk by start offset, with the mask row its row group starts at.
  std::unordered_map<std::int64_t, located_chunk> by_offset;
  auto mask_row = first_source_row;
  for (auto const rg : row_groups) {
    auto const& row_group = footer.row_groups[rg];
    for (auto const& chunk : row_group.columns) {
      auto const offset = chunk_start_offset(chunk.meta_data);
      auto const size   = static_cast<std::size_t>(chunk.meta_data.total_compressed_size);
      by_offset.emplace(offset,
                        located_chunk{source, &chunk, offset, size, mask_row, row_group.num_rows});
    }
    mask_row += static_cast<cudf::size_type>(row_group.num_rows);
  }

  std::vector<located_chunk> located;
  located.reserve(ranges.size());
  for (auto const& range : ranges) {
    auto const found = by_offset.find(range.offset());
    CUDF_EXPECTS(found != by_offset.end() && std::cmp_equal(found->second.size, range.size()),
                 "[parquet_materialize] a reader byte range matches no selected column chunk");
    located.push_back(found->second);
  }
  return located;
}

/// Data pages of a chunk, from its encoding statistics (0 when the writer
/// recorded none): the whole-chunk route has no page index to count them with.
std::uint64_t data_page_count(ColumnChunkMetaData const& meta)
{
  std::uint64_t pages = 0;
  if (!meta.encoding_stats.has_value()) { return pages; }
  for (auto const& stats : *meta.encoding_stats) {
    if (stats.page_type == PageType::DATA_PAGE || stats.page_type == PageType::DATA_PAGE_V2) {
      pages += static_cast<std::uint64_t>(stats.count);
    }
  }
  return pages;
}

/// Decoded bytes per row of output column @p column, for decoded_materialized:
/// the type width for fixed-width columns, (chars + offsets bytes) / rows for
/// strings, and otherwise -- other types, or strings with no rows to average
/// over -- the encoded bytes per value of the column's chunks.
double decoded_bytes_per_row(cudf::column_view const& column,
                             column_accounting const& chunks,
                             ::cuda::stream_ref stream)
{
  if (cudf::is_fixed_width(column.type())) {
    return static_cast<double>(cudf::size_of(column.type()));
  }
  if (column.type().id() == cudf::type_id::STRING && column.size() > 0) {
    cudf::strings_column_view const strings(column);
    auto const offsets       = strings.offsets();
    auto const offsets_bytes = static_cast<double>(offsets.size()) * cudf::size_of(offsets.type());
    auto const chars_bytes   = static_cast<double>(strings.chars_size(stream));
    return (chars_bytes + offsets_bytes) / column.size();
  }
  if (chunks.values == 0) { return 0.0; }
  return static_cast<double>(chunks.uncompressed_bytes) / static_cast<double>(chunks.values);
}

/// Names of a reader output's columns, in column order.
std::vector<std::string> column_names(cudf::io::table_metadata const& metadata)
{
  std::vector<std::string> names;
  names.reserve(metadata.schema_info.size());
  for (auto const& column : metadata.schema_info) {
    names.push_back(column.name);
  }
  return names;
}

/// Add one materialize call to the flowcsd scan counters.  @p output_names name
/// the columns of @p output, in order.
void record_scan_counters(read_accounting const& accounting,
                          std::span<parquet_source const> sources,
                          std::vector<std::string> const& output_names,
                          cudf::table_view const& output,
                          ::cuda::stream_ref stream)
{
  using flowcsd::byte_counter;
  using flowcsd::event_counter;

  std::uint64_t row_groups = 0;
  for (auto const& src : sources) {
    row_groups += src.row_group_indices.size();
  }

  // decoded_materialized: rows in decoded pages x decoded width, over the output columns.
  double decoded_bytes = 0;
  for (std::size_t c = 0; c < output_names.size(); ++c) {
    auto const found = accounting.columns.find(output_names[c]);
    if (found == accounting.columns.end()) { continue; }
    auto const& column       = output.column(static_cast<cudf::size_type>(c));
    auto const bytes_per_row = decoded_bytes_per_row(column, found->second, stream);
    decoded_bytes += static_cast<double>(found->second.rows_decoded) * bytes_per_row;
  }

  // Totals and the per-(table, column) breakdown.  The study's layout keeps one
  // table per directory ({table}/{table}-NNNNN.parquet), so the directory names it.
  auto const table = std::filesystem::path(path_of(sources.front())).parent_path().filename();
  for (auto const& [name, column] : accounting.columns) {
    flowcsd::add_bytes(byte_counter::chunk_bytes_full, column.scan.chunk_bytes_full);
    flowcsd::add_bytes(byte_counter::nvme_read_requested, column.scan.nvme_read_requested);
    flowcsd::add_count(event_counter::pages_total, column.scan.pages_total);
    flowcsd::add_count(event_counter::pages_read, column.scan.pages_read);
    flowcsd::add_column_scan(table.string() + "." + name, column.scan);
  }
  flowcsd::add_bytes(byte_counter::decoded_materialized, static_cast<std::uint64_t>(decoded_bytes));
  flowcsd::add_count(event_counter::row_groups_read, row_groups);
}

/// Run one cuDF materialize call under the flowcsd decode timer, waiting for its
/// kernels so the interval covers the decode itself.
template <typename Materialize>
cudf::io::table_with_metadata timed_decode(Materialize&& materialize, ::cuda::stream_ref stream)
{
  flowcsd::timed_scope const decode{flowcsd::time_category::decode};
  auto result = materialize();
  stream.sync();
  return result;
}

/// Column chunks resident on the device: one source's, or one page-skipping step's.
struct fetched_chunks {
  /// One buffer per column chunk.  They must outlive the decode; their
  /// stream-ordered deallocation is ordered behind it, so no synchronize is
  /// needed at the hand-off.
  std::vector<rmm::device_buffer> buffers;
  /// One span per chunk, in the order the reader enumerated the ranges -- which
  /// is the order @c materialize_all_columns expects its chunk data in.
  std::vector<cudf::device_span<std::uint8_t const>> spans;
};

/// Read every column chunk of @p src into device memory in ONE vectored
/// request, so a split costs one round trip per file rather than one per chunk.
/// That is the whole point of the bulk route against an object store, where
/// every extra request is another round trip.  Adds the read to @p accounting:
/// whole chunks, every data page.
fetched_chunks fetch_chunks(parquet_source const& src,
                            cudf::io::parquet_reader_options const& options,
                            std::span<cudf::io::text::byte_range_info const> ranges,
                            read_accounting& accounting,
                            ::cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  // Derive the ranges when the caller did not already have them.  A caller that
  // fadvised this split does, and passing them avoids re-walking the footer.
  std::vector<cudf::io::text::byte_range_info> owned;
  if (ranges.empty()) {
    owned  = column_chunk_ranges(*src.metadata, options, src.row_group_indices);
    ranges = owned;
  }

  // The row-mask fields of the located chunks are unused here: this route has no row mask.
  for (auto const& located : locate_chunks(*src.metadata, src.row_group_indices, ranges, 0, 0)) {
    auto const pages = data_page_count(located.chunk->meta_data);
    accounting.add_chunk(
      *located.chunk, located.num_rows, static_cast<std::uint64_t>(located.size), pages, pages);
  }

  fetched_chunks out;
  out.buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    out.buffers.emplace_back(static_cast<std::size_t>(range.size()), stream, mr);
  }

  std::vector<io::slice> reads;
  reads.reserve(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    reads.emplace_back(static_cast<std::size_t>(ranges[i].offset()),
                       static_cast<std::size_t>(ranges[i].size()),
                       static_cast<std::uint8_t*>(out.buffers[i].data()));
  }

  {
    std::ignore = src.datasource->device_read_ranges_async(reads, stream).get();
  }

  out.spans.reserve(out.buffers.size());
  for (auto const& buf : out.buffers) {
    out.spans.emplace_back(static_cast<std::uint8_t const*>(buf.data()), buf.size());
  }
  return out;
}

/// Ranges for source @p i, or an empty span when the caller supplied none.
std::span<cudf::io::text::byte_range_info const> ranges_for(
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges, std::size_t i)
{
  return i < ranges.size() ? std::span<cudf::io::text::byte_range_info const>(ranges[i])
                           : std::span<cudf::io::text::byte_range_info const>{};
}

/// Bulk route: read every source's column chunks, then decode straight out of
/// those device buffers.
std::unique_ptr<cudf::table> materialize_bulk(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  read_accounting accounting;
  if (sources.size() == 1) {
    // Single file: the single-source reader, whose chunk data is flat.
    auto const& src   = sources.front();
    auto const chunks = fetch_chunks(src, options, ranges_for(ranges, 0), accounting, stream, mr);

    hybrid_scan_reader reader(*src.metadata, options);
    auto result = timed_decode(
      [&] {
        return reader.materialize_all_columns(
          cudf::host_span<cudf::size_type const>(src.row_group_indices.data(),
                                                 src.row_group_indices.size()),
          cudf::host_span<cudf::device_span<std::uint8_t const> const>(chunks.spans.data(),
                                                                       chunks.spans.size()),
          options,
          stream,
          mr);
      },
      stream);
    record_scan_counters(
      accounting, sources, column_names(result.metadata), result.tbl->view(), stream);
    return std::move(result.tbl);
  }

  require_same_parquet_schema(sources);

  // Several files in one split: the multi-file reader takes the row groups per
  // source and wants its chunk data flattened in source order, with each
  // source's chunks in the same row-group / column order the single-file reader
  // produces.  Fetching per source and appending in that order is exactly that
  // layout -- and it keeps one vectored request per file, which is the unit the
  // IO backend can coalesce (ranges from different files share no offsets).
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::device_span<std::uint8_t const>> spans;
  metadatas.reserve(sources.size());
  rg_per_src.reserve(sources.size());

  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto chunks = fetch_chunks(sources[i], options, ranges_for(ranges, i), accounting, stream, mr);
    spans.insert(spans.end(), chunks.spans.begin(), chunks.spans.end());
    // Moving a device_buffer keeps its device pointer, so the spans stay valid;
    // the buffers just have to outlive the decode.
    std::move(chunks.buffers.begin(), chunks.buffers.end(), std::back_inserter(buffers));
    metadatas.push_back(*sources[i].metadata);
    rg_per_src.push_back(sources[i].row_group_indices);
  }

  cudf::io::parquet::experimental::hybrid_scan_multifile reader(
    cudf::host_span<cudf::io::parquet::FileMetaData const>(metadatas.data(), metadatas.size()),
    options);
  auto result = timed_decode(
    [&] {
      return reader.materialize_all_columns(
        cudf::host_span<std::vector<cudf::size_type> const>(rg_per_src.data(), rg_per_src.size()),
        cudf::host_span<cudf::device_span<std::uint8_t const> const>(spans.data(), spans.size()),
        options,
        stream,
        mr);
    },
    stream);
  record_scan_counters(
    accounting, sources, column_names(result.metadata), result.tbl->view(), stream);
  return std::move(result.tbl);
}

/// General route: let cudf read as it decodes, over whatever sources there are.
std::unique_ptr<cudf::table> materialize_general(std::span<parquet_source const> sources,
                                                 cudf::io::parquet_reader_options const& options,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::io::datasource>> cudf_sources;
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  cudf_sources.reserve(sources.size());
  metadatas.reserve(sources.size());
  rg_per_src.reserve(sources.size());

  for (auto const& src : sources) {
    cudf_sources.push_back(cudf::io::datasource::create(src.datasource.get()));
    metadatas.push_back(*src.metadata);
    rg_per_src.push_back(src.row_group_indices);
  }

  auto opts = options;
  opts.set_row_groups(std::move(rg_per_src));

  auto [table, _] =
    cudf::io::read_parquet(std::move(cudf_sources), std::move(metadatas), opts, stream, mr);
  return std::move(table);
}

// ---- Page-skipping route -----------------------------------------------------
//
// Leans on internals of cuDF's hybrid scan reader (cuDF v26.08.01,
// cpp/src/io/parquet/ and its experimental/ directory; file:line below):
//   * with the page index set up, cuDF parses the header of EVERY page of a
//     chunk at its OffsetIndex location inside the chunk buffer, reading only
//     PageHeader {1 type, 2 uncompressed_page_size, 3 compressed_page_size > 0,
//     5 DataPageHeader {1 num_values, 2 encoding, 3, 4 level encodings}} and
//     requiring a supported encoding (page_hdr.cu:426-507, 742-824;
//     reader_impl_preprocess_utils.cu:444-500);
//   * with use_data_page_mask::YES it neither decompresses nor decodes a page
//     whose OffsetIndex row range holds no true-or-null row of the row mask
//     (page_index_filter.cu:974-1183; reader_impl_chunking_utils.cu:121-146,
//     524-571; the decode kernels in reader_impl.cpp), and nulls those rows
//     before the filter drops them.
// So nothing past a pruned page's header is ever looked at, and a synthesized
// header is all a chunk buffer needs where the page was not read.

/// Whether every column chunk of @p footer records where its ColumnIndex and
/// OffsetIndex live, which loading the page index needs.
bool has_page_index(FileMetaData const& footer)
{
  return std::all_of(footer.row_groups.begin(), footer.row_groups.end(), [](auto const& row_group) {
    return std::all_of(
      row_group.columns.begin(), row_group.columns.end(), [](ColumnChunk const& chunk) {
        return chunk.offset_index_length > 0 && chunk.column_index_length > 0;
      });
  });
}

/// Whether @p footer has no repeated (list) fields: a synthesized header
/// counts one value per row, which holds only without repetition.
bool has_flat_schema(FileMetaData const& footer)
{
  return std::none_of(footer.schema.begin(), footer.schema.end(), [](auto const& element) {
    return element.max_repetition_level > 0;
  });
}

/// Whether the page-skipping route serves a bulk read of @p sources.
bool can_skip_pages(std::span<parquet_source const> sources,
                    cudf::io::parquet_reader_options const& options,
                    page_index_cache const* page_indexes)
{
  if (page_indexes == nullptr || !flowcsd::config().page_skipping) { return false; }
  if (!options.get_filter().has_value()) { return false; }
  return std::all_of(sources.begin(), sources.end(), [](parquet_source const& src) {
    return has_page_index(*src.metadata) && has_flat_schema(*src.metadata);
  });
}

/// @p src's footer with its page index read from the file and parsed in.
std::shared_ptr<FileMetaData const> load_page_index(parquet_source const& src,
                                                    cudf::io::parquet_reader_options const& options)
{
  hybrid_scan_reader const reader(*src.metadata, options);
  auto const range = reader.page_index_byte_range();
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(range.size()));
  auto const bytes_read =
    src.datasource->host_read(static_cast<std::size_t>(range.offset()), bytes.size(), bytes.data());
  CUDF_EXPECTS(bytes_read == bytes.size(),
               "[parquet_materialize] short page index read from " + path_of(src));

  reader.setup_page_index(cudf::host_span<std::uint8_t const>(bytes.data(), bytes.size()));
  return std::make_shared<FileMetaData const>(reader.parquet_metadata());
}

/// @p src's footer carrying the page index of its selected row groups only.
///
/// The readers copy the footer they are given, and a whole file's page index
/// dwarfs one split's share of it.  cuDF consults the page index of the
/// selected row groups only (page_index_filter_utils.cu:27-168,
/// reader_impl_helpers.cpp:703-841).
FileMetaData with_selected_page_index(parquet_source const& src, FileMetaData const& indexed)
{
  auto footer = *src.metadata;
  for (auto const rg : src.row_group_indices) {
    auto& columns               = footer.row_groups[rg].columns;
    auto const& indexed_columns = indexed.row_groups[rg].columns;
    for (std::size_t c = 0; c < columns.size(); ++c) {
      columns[c].offset_index = indexed_columns[c].offset_index;
      columns[c].column_index = indexed_columns[c].column_index;
    }
  }
  return footer;
}

/// @ref locate_chunks over every source, for byte ranges flattened in source
/// order with @p range_sources naming each range's source -- the multi-file
/// reader's layout.
std::vector<located_chunk> locate_all_chunks(
  std::vector<FileMetaData> const& footers,
  std::vector<std::vector<cudf::size_type>> const& row_groups,
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::vector<cudf::size_type> const& range_sources)
{
  std::vector<located_chunk> located;
  located.reserve(ranges.size());
  std::size_t begin                = 0;
  cudf::size_type first_source_row = 0;
  for (std::size_t source = 0; source < footers.size(); ++source) {
    auto end = begin;
    while (end < ranges.size() && std::cmp_equal(range_sources[end], source)) {
      ++end;
    }
    auto const source_ranges = std::span(ranges).subspan(begin, end - begin);
    auto const source_chunks =
      locate_chunks(footers[source], row_groups[source], source_ranges, source, first_source_row);
    located.insert(located.end(), source_chunks.begin(), source_chunks.end());
    for (auto const rg : row_groups[source]) {
      first_source_row += static_cast<cudf::size_type>(footers[source].row_groups[rg].num_rows);
    }
    begin = end;
  }
  CUDF_EXPECTS(begin == ranges.size(),
               "[parquet_materialize] reader byte ranges are not in source order");
  return located;
}

/// The encoding a synthesized header names for a pruned data page of a chunk.
///
/// cuDF never decodes a pruned page, but it rejects unsupported encodings and
/// picks a decode-kernel bit from every page header (page_hdr.cu:211-269,
/// 793-819), so the header names an encoding the chunk's real data pages use:
/// the first data-page entry of encoding_stats or, without them, the encoding
/// list minus the level encodings (RLE, BIT_PACKED) and, for a dictionary
/// chunk, minus the dictionary page's own entry.
Encoding data_page_encoding(ColumnChunkMetaData const& meta, bool has_dictionary_page)
{
  if (meta.encoding_stats.has_value()) {
    for (auto const& stats : *meta.encoding_stats) {
      if (stats.page_type == PageType::DATA_PAGE || stats.page_type == PageType::DATA_PAGE_V2) {
        return stats.encoding;
      }
    }
  }
  auto const& encodings = meta.encodings;
  if (has_dictionary_page) {
    // Dictionary-encoded data pages: RLE_DICTIONARY from format 2 writers,
    // PLAIN_DICTIONARY from format 1 writers.
    bool const rle_dictionary =
      std::find(encodings.begin(), encodings.end(), Encoding::RLE_DICTIONARY) != encodings.end();
    return rle_dictionary ? Encoding::RLE_DICTIONARY : Encoding::PLAIN_DICTIONARY;
  }
  for (auto const encoding : encodings) {
    if (encoding != Encoding::RLE && encoding != Encoding::BIT_PACKED) { return encoding; }
  }
  // Only level encodings listed: an RLE-encoded boolean column.
  return Encoding::RLE;
}

// Thrift compact protocol as cuDF's page header parser reads it
// (page_hdr.cu:78-104, 402-417): a field header byte is
// (field-id delta << 4) | type, an i32 is a zigzag varint, and a struct ends
// with a stop byte.
constexpr std::uint8_t THRIFT_TYPE_I32    = 5;
constexpr std::uint8_t THRIFT_TYPE_STRUCT = 12;
constexpr std::uint8_t THRIFT_STOP        = 0;
/// PageHeader.data_page_header is field 5, two ids after compressed_page_size.
constexpr std::uint8_t DATA_PAGE_HEADER_FIELD_DELTA = 2;
/// Bytes of a synthesized header besides its two page sizes and its row count.
constexpr std::int32_t SYNTHESIZED_HEADER_FIXED_BYTES = 14;

/// Zigzag form of a Thrift i32.
std::uint32_t zigzag(std::int32_t value)
{
  return (static_cast<std::uint32_t>(value) << 1) ^ static_cast<std::uint32_t>(value >> 31);
}

/// Bytes of the shortest varint of @p value's zigzag form.
std::int32_t i32_bytes(std::int32_t value)
{
  auto bits         = zigzag(value);
  std::int32_t size = 1;
  while (bits >= 0x80) {
    bits >>= 7;
    ++size;
  }
  return size;
}

void put_byte(page_header_write& header, std::uint8_t byte) { header.bytes[header.size++] = byte; }

/// Append an i32 field @p field_delta ids after the previous field, its varint
/// padded to @p width bytes (at least the shortest form).  cuDF keeps reading
/// varint bytes while the continuation bit is set (page_hdr.cu:78-87), so a
/// padded varint reads back the same value.
void put_i32_field(page_header_write& header,
                   std::uint8_t field_delta,
                   std::int32_t value,
                   std::int32_t width)
{
  put_byte(header, static_cast<std::uint8_t>((field_delta << 4) | THRIFT_TYPE_I32));
  auto bits = zigzag(value);
  for (std::int32_t i = 1; i < width; ++i) {
    put_byte(header, static_cast<std::uint8_t>((bits & 0x7F) | 0x80));
    bits >>= 7;
  }
  put_byte(header, static_cast<std::uint8_t>(bits));
}

/// Thrift compact-protocol PageHeader standing in for a pruned V1 data page
/// that held @p num_rows rows in @p page_bytes bytes (its OffsetIndex
/// compressed_page_size, header included), to be written at @p destination:
///
///   15 00      PageHeader.type = DATA_PAGE
///   15 <w>     PageHeader.uncompressed_page_size = body
///   15 <w>     PageHeader.compressed_page_size = body
///   2C         PageHeader.data_page_header: struct, field id +2
///   15 <r>       DataPageHeader.num_values = num_rows
///   15 <e>       DataPageHeader.encoding = encoding
///   15 06        DataPageHeader.definition_level_encoding = RLE
///   15 06        DataPageHeader.repetition_level_encoding = RLE
///   00           end of DataPageHeader
///   00         end of PageHeader
///
/// Values are zigzag varints; the two sizes are padded to w, the varint length
/// of page_bytes, so the header length (14 + 2w + r bytes) is known before
/// body = page_bytes - header length is.  Header plus body then fill the page's
/// OffsetIndex extent exactly, so cuDF's sequential header walk
/// (page_hdr.cu:577-647) also lands on the next page.
page_header_write synthesized_page_header(std::uint8_t* destination,
                                          std::int32_t page_bytes,
                                          std::int32_t num_rows,
                                          Encoding encoding)
{
  auto const size_width   = i32_bytes(page_bytes);
  auto const rows_width   = i32_bytes(num_rows);
  auto const header_bytes = SYNTHESIZED_HEADER_FIXED_BYTES + 2 * size_width + rows_width;
  auto const body_bytes   = page_bytes - header_bytes;
  // The page's real header carried these fields too, so only a page of a few
  // bytes could leave no room for a body.
  CUDF_EXPECTS(body_bytes > 0,
               "[parquet_materialize] data page too small for a synthesized header");

  page_header_write header{destination, 0, {}};
  put_i32_field(header, 1, static_cast<std::int32_t>(PageType::DATA_PAGE), 1);
  put_i32_field(header, 1, body_bytes, size_width);
  put_i32_field(header, 1, body_bytes, size_width);
  put_byte(header,
           static_cast<std::uint8_t>((DATA_PAGE_HEADER_FIELD_DELTA << 4) | THRIFT_TYPE_STRUCT));
  put_i32_field(header, 1, num_rows, rows_width);
  put_i32_field(header, 1, static_cast<std::int32_t>(encoding), 1);
  put_i32_field(header, 1, static_cast<std::int32_t>(Encoding::RLE), 1);
  put_i32_field(header, 1, static_cast<std::int32_t>(Encoding::RLE), 1);
  put_byte(header, THRIFT_STOP);
  put_byte(header, THRIFT_STOP);
  CUDF_EXPECTS(std::cmp_equal(header.size, header_bytes),
               "[parquet_materialize] synthesized header length disagrees with its layout");
  return header;
}

/// Row-mask row ranges of the data pages of @p chunks, chunk after chunk.  Page
/// k of a chunk covers rows [first_row_index_k, first_row_index_k+1) of its row
/// group, the last page ending at the row group's row count: the ranges cuDF
/// builds its data page mask from (page_index_filter_utils.cu:118-168).
std::vector<page_row_range> page_row_ranges(std::span<located_chunk const> chunks)
{
  std::vector<page_row_range> ranges;
  for (auto const& chunk : chunks) {
    auto const& locations = chunk.chunk->offset_index.value().page_locations;
    for (std::size_t k = 0; k < locations.size(); ++k) {
      auto const end_row =
        k + 1 < locations.size() ? locations[k + 1].first_row_index : chunk.num_rows;
      ranges.push_back(
        {static_cast<cudf::size_type>(chunk.first_mask_row + locations[k].first_row_index),
         static_cast<cudf::size_type>(chunk.first_mask_row + end_row)});
    }
  }
  return ranges;
}

/// Device buffers the size of whole chunks, one per entry of @p chunks.  The
/// whole size keeps file offsets inside a chunk 1:1 with buffer offsets, which
/// is how cuDF finds each page from the OffsetIndex
/// (reader_impl_preprocess_utils.cu:449-484).
fetched_chunks allocate_chunk_buffers(std::span<located_chunk const> chunks,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  fetched_chunks out;
  out.buffers.reserve(chunks.size());
  out.spans.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    auto const& buffer = out.buffers.emplace_back(chunk.size, stream, mr);
    out.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  return out;
}

/// Reads and header writes that fill one step's chunk buffers.
struct chunk_fill_plan {
  std::vector<std::vector<io::slice>> reads_per_source;  ///< one vectored read per source
  std::vector<page_header_write> headers;                ///< one per pruned page
};

/// Add to @p plan what fills @p chunk's @p buffer: runs of adjacent bytes to
/// read -- the dictionary region [chunk start, first data page), which cuDF
/// always decodes, then the surviving pages -- and a synthesized header for
/// every pruned page.  The chunk's pages are entries @p first_page onward of
/// @p pages and @p surviving.  Returns the rows in its surviving pages.
std::int64_t plan_chunk_fill(located_chunk const& chunk,
                             std::uint8_t* buffer,
                             std::vector<page_row_range> const& pages,
                             std::vector<bool> const& surviving,
                             std::size_t first_page,
                             chunk_fill_plan& plan)
{
  auto const& locations = chunk.chunk->offset_index.value().page_locations;
  CUDF_EXPECTS(!locations.empty(), "[parquet_materialize] column chunk without data pages");
  bool const has_dictionary = locations.front().offset > chunk.file_offset;
  auto const encoding       = data_page_encoding(chunk.chunk->meta_data, has_dictionary);
  auto& reads               = plan.reads_per_source[chunk.source];
  auto const add_read       = [&](std::int64_t begin, std::int64_t end) {
    if (end <= begin) { return; }
    reads.emplace_back(static_cast<std::size_t>(begin),
                       static_cast<std::size_t>(end - begin),
                       buffer + (begin - chunk.file_offset));
  };

  std::int64_t rows_decoded = 0;
  std::int64_t run_begin    = chunk.file_offset;
  std::int64_t run_end      = locations.front().offset;
  for (std::size_t k = 0; k < locations.size(); ++k) {
    auto const& location = locations[k];
    auto const& rows     = pages[first_page + k];
    auto const page_rows = rows.end_row - rows.first_row;
    auto const page_end  = location.offset + location.compressed_page_size;
    if (!surviving[first_page + k]) {
      // Pruned: end the run here and let a header stand in for the page.
      add_read(run_begin, run_end);
      run_begin = page_end;
      run_end   = page_end;
      plan.headers.push_back(synthesized_page_header(buffer + (location.offset - chunk.file_offset),
                                                     location.compressed_page_size,
                                                     page_rows,
                                                     encoding));
      continue;
    }
    // Surviving: extend the run; pages adjacent in the file share one slice.
    if (location.offset != run_end) {
      add_read(run_begin, run_end);
      run_begin = location.offset;
    }
    run_end = page_end;
    rows_decoded += page_rows;
  }
  add_read(run_begin, run_end);
  return rows_decoded;
}

/// Issue each source's reads as ONE vectored device read and wait for it,
/// source after source as on the plain bulk route.
void read_per_source(std::span<parquet_source const> sources,
                     std::vector<std::vector<io::slice>> const& reads_per_source,
                     ::cuda::stream_ref stream)
{
  for (std::size_t source = 0; source < sources.size(); ++source) {
    auto const& reads = reads_per_source[source];
    if (reads.empty()) { continue; }
    std::ignore = sources[source].datasource->device_read_ranges_async(reads, stream).get();
  }
}

/// Read one step's column chunks (the filter or the payload columns) without
/// the data pages that hold no surviving row of @p row_mask.
///
/// Correctness invariant: the pages read are a superset of the pages cuDF
/// decodes.  With use_data_page_mask::YES cuDF decodes a page iff its
/// OffsetIndex row range holds a true-or-null row of the row mask it is given
/// (page_index_filter.cu:974-1183).  The caller passes this function that same
/// row mask, and pages_with_surviving_rows applies that rule over the same
/// OffsetIndex row ranges.  Every other page keeps a parseable synthesized
/// header, since cuDF parses every page header of a chunk before masking.
fetched_chunks fetch_surviving_pages(std::span<parquet_source const> sources,
                                     std::span<located_chunk const> chunks,
                                     cudf::column_view const& row_mask,
                                     read_accounting& accounting,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  // Which data pages hold a surviving row.
  auto const pages     = page_row_ranges(chunks);
  auto const surviving = pages_with_surviving_rows(row_mask, pages, stream, mr);
  auto const pages_read = std::count(surviving.begin(), surviving.end(), true);

  // No page survives, so every row is pruned: cuDF then returns an empty table
  // without touching chunk data (hybrid_scan_impl.cpp:527-535, 565-573).
  if (pages_read == 0) {
    for (auto const& chunk : chunks) {
      auto const chunk_pages = chunk.chunk->offset_index.value().page_locations.size();
      accounting.add_chunk(*chunk.chunk, 0, 0, chunk_pages, 0);
    }
    return {};
  }

  // Plan the reads and headers that fill whole-chunk buffers.  A chunk whose
  // pages all survive is one run: a single whole-chunk slice, as on the plain
  // bulk route.
  auto out = allocate_chunk_buffers(chunks, stream, mr);
  chunk_fill_plan plan{std::vector<std::vector<io::slice>>(sources.size()), {}};
  std::size_t first_page = 0;
  for (std::size_t c = 0; c < chunks.size(); ++c) {
    auto* const buffer        = static_cast<std::uint8_t*>(out.buffers[c].data());
    auto const& source_reads  = plan.reads_per_source[chunks[c].source];
    auto const reads_before   = source_reads.size();
    auto const rows_decoded   = plan_chunk_fill(chunks[c], buffer, pages, surviving, first_page, plan);
    std::uint64_t chunk_bytes = 0;  // what this chunk asks of the datasource
    for (auto r = reads_before; r < source_reads.size(); ++r) {
      chunk_bytes += source_reads[r].size();
    }
    auto const chunk_pages = chunks[c].chunk->offset_index.value().page_locations.size();
    auto const chunk_first = surviving.begin() + static_cast<std::ptrdiff_t>(first_page);
    auto const chunk_read =
      std::count(chunk_first, chunk_first + static_cast<std::ptrdiff_t>(chunk_pages), true);
    accounting.add_chunk(*chunks[c].chunk,
                         rows_decoded,
                         chunk_bytes,
                         chunk_pages,
                         static_cast<std::uint64_t>(chunk_read));
    first_page += chunk_pages;
  }

  // Headers and reads fill disjoint bytes of the buffers, so either may go first.
  write_page_headers(plan.headers, stream, mr);
  read_per_source(sources, plan.reads_per_source, stream);
  return out;
}

/// The chunk data of @p chunks as the readers take it.
cudf::host_span<cudf::device_span<std::uint8_t const> const> chunk_data(
  fetched_chunks const& chunks)
{
  return {chunks.spans.data(), chunks.spans.size()};
}

/// The columns @p options reads, in output order: its projection, or every
/// top-level column of the schema when it projects nothing (cuDF's order then,
/// reader_impl_helpers.cpp:2075-2079).
std::vector<std::string> output_column_names(FileMetaData const& footer,
                                             cudf::io::parquet_reader_options const& options)
{
  if (options.get_column_names().has_value()) { return *options.get_column_names(); }
  std::vector<std::string> names;
  for (auto const child : footer.schema.front().children_idx) {
    names.push_back(footer.schema[child].name);
  }
  return names;
}

/// Position of the column named @p name in a reader output, if it is there.
std::optional<std::size_t> column_position(cudf::io::table_metadata const& metadata,
                                           std::string const& name)
{
  auto const& columns = metadata.schema_info;
  auto const found    = std::find_if(
    columns.begin(), columns.end(), [&](auto const& column) { return column.name == name; });
  if (found == columns.end()) { return std::nullopt; }
  return static_cast<std::size_t>(found - columns.begin());
}

/// Whether some output column is not a filter column, i.e. a payload step is needed.
bool has_payload_columns(std::vector<std::string> const& names,
                         cudf::io::table_metadata const& filter_metadata)
{
  return std::any_of(names.begin(), names.end(), [&](auto const& name) {
    return !column_position(filter_metadata, name).has_value();
  });
}

/// The output table: the columns @p names in order, each moved out of the
/// filter or the payload table.  These are the columns, rows and row order
/// materialize_all_columns returns for the same options: both steps decode the
/// same selection and keep, in row order, the rows the filter accepts.  Column
/// selection by index and the prepended index columns, which Sirius never
/// sets, are not handled.
std::unique_ptr<cudf::table> assemble_output(std::vector<std::string> const& names,
                                             cudf::io::table_with_metadata filter,
                                             std::optional<cudf::io::table_with_metadata> payload)
{
  auto filter_columns = filter.tbl->release();
  std::vector<std::unique_ptr<cudf::column>> payload_columns;
  if (payload.has_value()) { payload_columns = payload->tbl->release(); }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(names.size());
  for (auto const& name : names) {
    if (auto const position = column_position(filter.metadata, name); position.has_value()) {
      columns.push_back(std::move(filter_columns[*position]));
      continue;
    }
    auto const position =
      payload.has_value() ? column_position(payload->metadata, name) : std::nullopt;
    CUDF_EXPECTS(position.has_value(),
                 "[parquet_materialize] output column '" + name +
                   "' is in neither the filter nor the payload table");
    columns.push_back(std::move(payload_columns[*position]));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Page-skipping route (see materialize_parquet): filter columns read without
/// the pages the page statistics rule out, decoded and filtered; then payload
/// columns read without the pages holding no row the filter kept.
std::unique_ptr<cudf::table> materialize_page_skipping(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  page_index_cache& page_indexes,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  using cudf::io::parquet::experimental::use_data_page_mask;
  require_same_parquet_schema(sources);

  // One reader over every source; its footers carry the selected row groups' page index.
  std::vector<FileMetaData> footers;
  std::vector<std::vector<cudf::size_type>> row_groups;
  footers.reserve(sources.size());
  row_groups.reserve(sources.size());
  for (auto const& src : sources) {
    auto const indexed = page_indexes.footer_with_page_index(src, options);
    footers.push_back(with_selected_page_index(src, *indexed));
    row_groups.push_back(src.row_group_indices);
  }
  hybrid_scan_multifile const reader(
    cudf::host_span<FileMetaData const>(footers.data(), footers.size()), options);
  cudf::host_span<std::vector<cudf::size_type> const> const selection(row_groups.data(),
                                                                      row_groups.size());
  read_accounting accounting;

  // Rows the page statistics cannot rule out (null where they cannot decide).
  auto row_mask = reader.build_row_mask_with_page_index_stats(selection, options, stream, mr);

  // Filter step: read and decode the filter columns.  The reader narrows the
  // row mask to the rows the filter keeps.
  auto const [filter_ranges, filter_sources] =
    reader.filter_column_chunks_byte_ranges(selection, options);
  auto const filter_chunks = locate_all_chunks(footers, row_groups, filter_ranges, filter_sources);
  auto const filter_data =
    fetch_surviving_pages(sources, filter_chunks, row_mask->view(), accounting, stream, mr);
  auto row_mask_view = row_mask->mutable_view();
  auto filter_table  = timed_decode(
    [&] {
      return reader.materialize_filter_columns(selection,
                                               chunk_data(filter_data),
                                               row_mask_view,
                                               use_data_page_mask::YES,
                                               options,
                                               stream,
                                               mr);
    },
    stream);
  // The reader left every row of the mask valid (hybrid_scan_preprocess.cu:376-386)
  // but recorded the new null count only in its own copy of the view.
  row_mask->set_null_count(0);

  // Payload step: read and decode the other output columns for the rows the filter kept.
  auto const names = output_column_names(footers.front(), options);
  std::optional<cudf::io::table_with_metadata> payload_table;
  if (has_payload_columns(names, filter_table.metadata)) {
    auto const [payload_ranges, payload_sources] =
      reader.payload_column_chunks_byte_ranges(selection, options);
    auto const payload_chunks =
      locate_all_chunks(footers, row_groups, payload_ranges, payload_sources);
    auto const payload_data =
      fetch_surviving_pages(sources, payload_chunks, row_mask->view(), accounting, stream, mr);
    payload_table = timed_decode(
      [&] {
        return reader.materialize_payload_columns(selection,
                                                  chunk_data(payload_data),
                                                  row_mask->view(),
                                                  use_data_page_mask::YES,
                                                  options,
                                                  stream,
                                                  mr);
      },
      stream);
  }

  auto output = assemble_output(names, std::move(filter_table), std::move(payload_table));
  record_scan_counters(accounting, sources, names, output->view(), stream);
  return output;
}

}  // namespace

std::shared_ptr<FileMetaData const> page_index_cache::footer_with_page_index(
  parquet_source const& src, cudf::io::parquet_reader_options const& options)
{
  std::shared_ptr<entry> file_entry;
  {
    std::lock_guard const lock(_mutex);
    auto& slot = _by_path[path_of(src)];
    if (!slot) { slot = std::make_shared<entry>(); }
    file_entry = slot;
  }
  // Loaded outside the map lock so different files load in parallel; call_once
  // makes concurrent first uses of one file share a single load.
  std::call_once(file_entry->loaded,
                 [&] { file_entry->footer = load_page_index(src, options); });
  return file_entry->footer;
}

void require_same_parquet_schema(std::span<parquet_source const> sources)
{
  if (sources.size() < 2) { return; }
  auto const& first = sources.front().metadata->schema;
  for (auto const& src : sources.subspan(1)) {
    auto const& schema = src.metadata->schema;
    if (schema == first) { continue; }
    std::string const prefix = "[parquet_materialize] All sources must have the same schema: '" +
                               path_of(sources.front()) + "' and '" + path_of(src) + "' ";
    if (schema.size() != first.size()) {
      throw std::runtime_error(prefix + "have " + std::to_string(first.size()) + " and " +
                               std::to_string(schema.size()) +
                               " schema elements (including the root)");
    }
    auto const [lhs, rhs] = std::mismatch(first.begin(), first.end(), schema.begin(), schema.end());
    std::string detail    = describe(*lhs) + " vs " + describe(*rhs);
    if (lhs->name == rhs->name && lhs->type == rhs->type) {
      detail += "; they differ in annotation, width or nesting";
    }
    throw std::runtime_error(prefix + "differ at schema element " +
                             std::to_string(lhs - first.begin()) + " (" + detail + ")");
  }
}

std::unique_ptr<cudf::table> materialize_parquet(
  std::span<parquet_source const> sources,
  cudf::io::parquet_reader_options const& options,
  std::span<std::vector<cudf::io::text::byte_range_info> const> ranges,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  page_index_cache* page_indexes)
{
  if (sources.empty()) { return nullptr; }
  if (prefers_bulk_materialize(sources, options)) {
    if (can_skip_pages(sources, options, page_indexes)) {
      return materialize_page_skipping(sources, options, *page_indexes, stream, mr);
    }
    return materialize_bulk(sources, options, ranges, stream, mr);
  }
  return materialize_general(sources, options, stream, mr);
}

}  // namespace sirius::op::scan
