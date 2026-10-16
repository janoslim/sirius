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

#include "flowcsd/flowcsd_config.hpp"

#include "yaml_reader.hpp"

#include <utility>

namespace sirius::flowcsd {

namespace {

// Written once during context initialization, before any query thread starts,
// and only read afterwards, so no lock is needed.
flowcsd_config installed_config;

}  // namespace

void flowcsd_config::from_yaml(YAML::Node const& node, flowcsd_config& out)
{
  yaml::reader r(node, "sirius.flowcsd");
  r.optional("tutti_runtime_config", out.tutti_runtime_config);
  if (r.has_value("tutti_path_prefixes")) {
    // A YAML list replaces the default prefixes instead of appending to them.
    out.tutti_path_prefixes.clear();
    r.optional("tutti_path_prefixes", out.tutti_path_prefixes);
  }
  r.optional("staging_slab_bytes", yaml::bytes(out.staging_slab_bytes));
  r.optional("read_staging_slabs", out.read_staging_slabs);
  r.optional("spill_staging_slabs", out.spill_staging_slabs);
  r.optional("max_request_bytes", yaml::bytes(out.max_request_bytes));
  r.optional("page_skipping", out.page_skipping);
  r.optional("direct_disk_spill", out.direct_disk_spill);
  r.optional("spill_dirs", out.spill_dirs);
  r.optional("stats_jsonl", out.stats_jsonl);
  r.reject_unknown();
}

void install_config(flowcsd_config config) { installed_config = std::move(config); }

flowcsd_config const& config() { return installed_config; }

int tutti_device_index(std::string const& path) noexcept
{
  auto const& prefixes = installed_config.tutti_path_prefixes;
  for (std::size_t index = 0; index < prefixes.size(); ++index) {
    if (path.starts_with(prefixes[index])) { return static_cast<int>(index); }
  }
  return -1;
}

}  // namespace sirius::flowcsd
