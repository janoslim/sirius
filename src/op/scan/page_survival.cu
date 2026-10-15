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

#include "op/scan/page_survival.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>

namespace sirius::op::scan {

namespace {

/// 1 when a row of the mask survives (null, or valid and true), else 0.
struct row_survives_fn {
  cudf::column_device_view row_mask;

  __device__ cudf::size_type operator()(cudf::size_type row) const
  {
    bool const survives = row_mask.is_null(row) || row_mask.element<bool>(row);
    return survives ? 1 : 0;
  }
};

/// 1 when a page's row range holds a surviving row, i.e. the running count of
/// surviving rows grows across the range.
struct page_survives_fn {
  cudf::size_type const* surviving_rows_before;  ///< [r] = surviving rows in [0, r)
  page_row_range const* pages;

  __device__ std::uint8_t operator()(std::size_t page) const
  {
    auto const& range = pages[page];
    return surviving_rows_before[range.end_row] > surviving_rows_before[range.first_row] ? 1 : 0;
  }
};

/// Copy one synthesized header to its destination (headers are a few dozen bytes).
struct write_header_fn {
  page_header_write const* headers;

  __device__ void operator()(std::size_t index) const
  {
    auto const& header = headers[index];
    for (std::uint32_t byte = 0; byte < header.size; ++byte) {
      header.destination[byte] = header.bytes[byte];
    }
  }
};

}  // namespace

std::vector<bool> pages_with_surviving_rows(cudf::column_view const& row_mask,
                                            std::span<page_row_range const> pages,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  if (pages.empty()) { return {}; }
  auto const num_rows = row_mask.size();
  // The per-page lookups below are unchecked device reads, so check the ranges here.
  bool const ranges_inside_mask =
    std::all_of(pages.begin(), pages.end(), [&](page_row_range const& page) {
      return 0 <= page.first_row && page.first_row <= page.end_row && page.end_row <= num_rows;
    });
  CUDF_EXPECTS(ranges_inside_mask, "[page_survival] page row range outside the row mask");

  // Running count of surviving rows: surviving_rows_before[r] counts rows [0, r).
  rmm::device_uvector<cudf::size_type> surviving_rows_before(static_cast<std::size_t>(num_rows) + 1,
                                                             stream,
                                                             mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(surviving_rows_before.data(), 0, sizeof(cudf::size_type), stream.get()));
  auto const mask     = cudf::column_device_view::create(row_mask, stream);
  auto const survives = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), row_survives_fn{*mask});
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr),
                         survives,
                         survives + num_rows,
                         surviving_rows_before.begin() + 1);

  // A page survives when the count grows between its two boundaries.
  rmm::device_uvector<page_row_range> device_pages(pages.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_pages.data(),
                                pages.data(),
                                pages.size_bytes(),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  rmm::device_uvector<std::uint8_t> device_survives(pages.size(), stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    thrust::make_counting_iterator<std::size_t>(0),
                    thrust::make_counting_iterator<std::size_t>(pages.size()),
                    device_survives.begin(),
                    page_survives_fn{surviving_rows_before.data(), device_pages.data()});

  // One copy back to the host.
  std::vector<std::uint8_t> host_survives(pages.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_survives.data(),
                                device_survives.data(),
                                host_survives.size(),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();
  return std::vector<bool>(host_survives.begin(), host_survives.end());
}

void write_page_headers(std::span<page_header_write const> headers,
                        ::cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr)
{
  if (headers.empty()) { return; }

  // One copy up; the device copy is freed in stream order, after the kernel reads it.
  rmm::device_uvector<page_header_write> device_headers(headers.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_headers.data(),
                                headers.data(),
                                headers.size_bytes(),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  thrust::for_each(rmm::exec_policy_nosync(stream, mr),
                   thrust::make_counting_iterator<std::size_t>(0),
                   thrust::make_counting_iterator<std::size_t>(headers.size()),
                   write_header_fn{device_headers.data()});
}

}  // namespace sirius::op::scan
