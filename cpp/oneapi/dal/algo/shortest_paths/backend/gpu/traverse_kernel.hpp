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

/// Lightweight GPU view of a CSR topology with edge weights.
/// Holds raw device pointers extracted from device_csr_topology.
template <typename Index = std::int32_t, typename Weight = std::int32_t>
struct csr_topology_gpu_view {
    const std::int64_t* rows = nullptr;
    const Index* cols = nullptr;
    const Weight* weights = nullptr;
    std::int64_t vertex_count = 0;
    std::int64_t edge_count = 0;
};

/// Creates a GPU view from a weighted device_csr_topology (zero-copy).
template <typename Index = std::int32_t, typename Weight = std::int32_t>
csr_topology_gpu_view<Index, Weight> make_gpu_view(
        const dal::preview::detail::device_csr_topology<Index, Weight>& device_topo) {
    csr_topology_gpu_view<Index, Weight> view;
    view.rows = device_topo.get_rows();
    view.cols = device_topo.get_cols();
    view.weights = device_topo.get_weights();
    view.vertex_count = device_topo.get_vertex_count();
    view.edge_count = device_topo.get_edge_count();
    return view;
}

template <typename Float, typename Task, typename Index, typename Weight>
traverse_result<Task> run_shortest_paths_gpu(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const csr_topology_gpu_view<Index, Weight>& topology);

} // namespace oneapi::dal::preview::shortest_paths::backend
