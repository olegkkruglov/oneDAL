/*******************************************************************************
* Copyright 2021 Intel Corporation
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
*******************************************************************************/

#pragma once

#include <cstdint>

#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

template <typename Index = std::int32_t>
struct csr_topology_gpu_view {
    const std::int64_t* rows = nullptr;
    const Index* cols = nullptr;
    std::int64_t vertex_count = 0;
    std::int64_t edge_count = 0;
    std::int64_t cols_count = 0;
};

/// Creates a GPU view from a device_csr_topology (zero-copy, just extracts pointers).
template <typename Index = std::int32_t>
csr_topology_gpu_view<Index> make_gpu_view(
        const dal::preview::detail::device_csr_topology<Index>& device_topo) {
    csr_topology_gpu_view<Index> view;
    view.rows = device_topo.get_rows();
    view.cols = device_topo.get_cols();
    view.vertex_count = device_topo.get_vertex_count();
    view.edge_count = device_topo.get_edge_count();
    view.cols_count = device_topo.get_cols_array().get_count();
    return view;
}

template <typename Float, typename Task, typename EdgeValue, typename Index = std::int32_t>
traverse_result<Task> run_shortest_paths_gpu(const dal::backend::context_gpu& ctx,
                                             const detail::descriptor_base<Task>& desc,
                                             const csr_topology_gpu_view<Index>& topology,
                                             const EdgeValue* edge_values);

} // namespace oneapi::dal::preview::shortest_paths::backend
