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

// Device side of the page-skipping Parquet scan route (parquet_materialize.cpp):
// deciding which data pages hold a surviving row of a row mask, and writing the
// synthesized headers that stand in for the pages left unread.

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sirius::op::scan {

/// Rows [first_row, end_row) of one data page, in row-mask coordinates.
struct page_row_range {
  cudf::size_type first_row;
  cudf::size_type end_row;
};

/// For each entry of @p pages, whether @p row_mask has a surviving row inside it.
///
/// A row survives when it is true or null.  This is the rule cuDF's
/// compute_data_page_mask applies to the same mask: it turns null rows true
/// before asking whether any row of a page's range is true.  @p row_mask is a
/// BOOL8 column.  Costs one prefix count over the mask, one lookup per page
/// boundary and one device-to-host copy; synchronizes @p stream.
[[nodiscard]] std::vector<bool> pages_with_surviving_rows(cudf::column_view const& row_mask,
                                                          std::span<page_row_range const> pages,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr);

/// One synthesized page header and the device address it is written to.
struct page_header_write {
  /// Room for the longest header parquet_materialize.cpp builds (29 bytes).
  static constexpr std::size_t max_bytes = 32;

  std::uint8_t* destination;
  std::uint32_t size;
  std::uint8_t bytes[max_bytes];
};

/// Write every header of @p headers to its destination: one host-to-device copy
/// of the whole batch, then one kernel that scatters the bytes.  Stream-ordered
/// on @p stream; @p headers may be released as soon as this returns.
void write_page_headers(std::span<page_header_write const> headers,
                        ::cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr);

}  // namespace sirius::op::scan
