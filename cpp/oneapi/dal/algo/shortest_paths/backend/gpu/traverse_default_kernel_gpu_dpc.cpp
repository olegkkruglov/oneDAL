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

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/shortest_paths/backend/gpu/shortest_paths.hpp"
#include "oneapi/dal/algo/shortest_paths/backend/gpu/traverse_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

namespace detail_gpu {

/// Computes single-source shortest paths on the GPU using a Bellman-Ford-style
/// iterative relaxation approach.
///
/// @tparam Index      The index type used for vertex/edge indexing
/// @tparam EdgeValue  The edge weight type (int32 or double)
/// @param[in]  queue         SYCL queue for device execution
/// @param[in]  rows          CSR row-offset array (vertex_count + 1 entries)
/// @param[in]  cols          CSR column-index array
/// @param[in]  edge_values   Edge weight array (same order as cols)
/// @param[in]  vertex_count  Number of vertices in the graph
/// @param[in]  edge_count    Number of edges
/// @param[in]  source        Source vertex
/// @param[in]  compute_predecessors  Whether to compute predecessor array
/// @return     A pair of (distances, predecessors) USM arrays
template <typename Index, typename EdgeValue>
std::pair<EdgeValue*, std::int32_t*> compute_shortest_paths_gpu(
        sycl::queue& queue,
        const std::int64_t* rows,
        const Index* cols,
        const EdgeValue* edge_values,
        std::int64_t vertex_count,
        std::int64_t edge_count,
        std::int64_t source,
        bool compute_predecessors) {
    constexpr EdgeValue inf_dist = std::numeric_limits<EdgeValue>::max();

    auto* dist = sycl::malloc_shared<EdgeValue>(vertex_count, queue);
    auto* changed = sycl::malloc_shared<std::int32_t>(1, queue);
    std::int32_t* pred = nullptr;

    if (compute_predecessors) {
        pred = sycl::malloc_shared<std::int32_t>(vertex_count, queue);
    }

    // Initialize distances to infinity and predecessors to -1
    queue
        .submit([&](sycl::handler& cgh) {
            EdgeValue* d_dist = dist;
            std::int32_t* d_pred = pred;
            const std::int64_t vc = vertex_count;
            const std::int64_t src = source;
            const EdgeValue inf_val = inf_dist;
            const bool do_pred = compute_predecessors;

            cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                const std::int64_t v = idx[0];
                d_dist[v] = (v == src) ? EdgeValue(0) : inf_val;
                if (do_pred) {
                    d_pred[v] = -1;
                }
            });
        })
        .wait_and_throw();

    // Iterative relaxation until convergence
    bool has_changed = true;
    while (has_changed) {
        changed[0] = 0;

        queue
            .submit([&](sycl::handler& cgh) {
                const std::int64_t* d_rows = rows;
                const Index* d_cols = cols;
                const EdgeValue* d_vals = edge_values;
                EdgeValue* d_dist = dist;
                std::int32_t* d_pred = pred;
                std::int32_t* d_changed = changed;
                const std::int64_t vc = vertex_count;
                const EdgeValue inf_val = inf_dist;
                const bool do_pred = compute_predecessors;

                cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                    const std::int64_t u = idx[0];
                    const EdgeValue dist_u = d_dist[u];

                    // Skip unreachable vertices
                    if (dist_u >= inf_val)
                        return;

                    const std::int64_t row_begin = d_rows[u];
                    const std::int64_t row_end = d_rows[u + 1];

                    for (std::int64_t e = row_begin; e < row_end; ++e) {
                        const std::int64_t v = d_cols[e];
                        const EdgeValue w = d_vals[e];
                        const EdgeValue new_dist = dist_u + w;

                        // Atomically update distance if shorter path found
                        sycl::atomic_ref<EdgeValue,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            dist_ref(d_dist[v]);

                        EdgeValue old_dist = dist_ref.load();
                        while (new_dist < old_dist) {
                            if (dist_ref.compare_exchange_weak(old_dist, new_dist)) {
                                if (do_pred) {
                                    d_pred[v] = static_cast<std::int32_t>(u);
                                }
                                d_changed[0] = 1;
                                break;
                            }
                        }
                    }
                });
            })
            .wait_and_throw();

        has_changed = (changed[0] != 0);
    }

    sycl::free(changed, queue);

    return { dist, pred };
}

} // namespace detail_gpu

template <typename Float, typename Task, typename EdgeValue, typename Index>
traverse_result<Task> run_shortest_paths_gpu(const dal::backend::context_gpu& ctx,
                                             const detail::descriptor_base<Task>& desc,
                                             const csr_topology_gpu_view<Index>& topology,
                                             const EdgeValue* edge_values) {
    auto& queue = ctx.get_queue();

    const auto vertex_count = topology.vertex_count;
    const auto edge_count = topology.edge_count;
    const auto source = desc.get_source();

    if (vertex_count == 0) {
        return traverse_result<Task>();
    }

    // Transfer edge values to device
    const std::int64_t cols_count = topology.cols_count;
    auto* device_edge_values = sycl::malloc_device<EdgeValue>(cols_count > 0 ? cols_count : 1, queue);
    if (cols_count > 0) {
        queue.memcpy(device_edge_values, edge_values, cols_count * sizeof(EdgeValue))
            .wait_and_throw();
    }

    const bool compute_predecessors =
        static_cast<bool>(desc.get_optional_results() & optional_results::predecessors);

    auto [dist, pred] = detail_gpu::compute_shortest_paths_gpu(queue,
                                                                topology.rows,
                                                                topology.cols,
                                                                device_edge_values,
                                                                vertex_count,
                                                                edge_count,
                                                                source,
                                                                compute_predecessors);

    sycl::free(device_edge_values, queue);

    traverse_result<Task> result;

    if (desc.get_optional_results() & optional_results::distances) {
        auto dist_arr =
            array<EdgeValue>(queue, dist, vertex_count, [queue](EdgeValue* ptr) mutable {
                sycl::free(ptr, queue);
            });
        result.set_distances(dal::detail::homogen_table_builder{}
                                 .reset(dist_arr, vertex_count, 1)
                                 .build());
    }
    else {
        sycl::free(dist, queue);
    }

    if (compute_predecessors && pred != nullptr) {
        auto pred_arr =
            array<std::int32_t>(queue, pred, vertex_count, [queue](std::int32_t* ptr) mutable {
                sycl::free(ptr, queue);
            });
        result.set_predecessors(dal::detail::homogen_table_builder{}
                                    .reset(pred_arr, vertex_count, 1)
                                    .build());
    }

    return result;
}

template <typename Float, typename Task, typename Topology>
traverse_result<Task> traverse_kernel_gpu<Float, Task, Topology>::operator()(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const Topology& topology) const {
    // This overload is not directly called; the Graph-aware overload in
    // select_kernel_dpc.cpp handles extracting topology and edge values.
    // Provided for template completeness.
    return traverse_result<Task>();
}

// Explicit template instantiations
template struct traverse_kernel_gpu<float,
                                    task::one_to_all,
                                    dal::preview::detail::topology<std::int32_t>>;

template traverse_result<task::one_to_all>
run_shortest_paths_gpu<float, task::one_to_all, std::int32_t, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::one_to_all>&,
    const csr_topology_gpu_view<std::int32_t>&,
    const std::int32_t*);

template traverse_result<task::one_to_all>
run_shortest_paths_gpu<float, task::one_to_all, double, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::one_to_all>&,
    const csr_topology_gpu_view<std::int32_t>&,
    const double*);

} // namespace oneapi::dal::preview::shortest_paths::backend
