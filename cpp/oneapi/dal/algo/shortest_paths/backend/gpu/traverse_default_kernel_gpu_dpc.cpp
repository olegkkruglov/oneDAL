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
#include "oneapi/dal/backend/primitives/frontier.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

namespace fp = oneapi::dal::preview::backend::primitives;

namespace detail_gpu {

/// Computes single-source shortest paths on the GPU using a frontier-driven
/// label-correcting approach with the oneDAL frontier advance primitive
/// (inspired by SYgraph, University of Salerno).
///
/// Each active vertex propagates its distance + edge weight to neighbors via
/// atomic fetch_min. A two-layer bitmap frontier tracks which vertices had
/// their distance improved, and the advance kernel provides workload-balanced
/// edge processing across workgroup, subgroup, and work-item granularities.
///
/// @tparam Index   The index type for column indices (e.g., int32_t)
/// @tparam Weight  The weight type for edge values (e.g., int32_t or double)
/// @param[in]  queue         SYCL queue for device execution
/// @param[in]  rows          CSR row-offset array (vertex_count + 1 entries)
/// @param[in]  cols          CSR column-index array
/// @param[in]  weights       Edge weight array (same size as cols)
/// @param[in]  vertex_count  Number of vertices in the graph
/// @param[in]  edge_count    Number of edges
/// @param[in]  source        Source vertex for SSSP
/// @param[in]  compute_predecessors  Whether to track predecessor vertices
/// @param[out] distances     USM-allocated distance array (caller frees)
/// @param[out] predecessors  USM-allocated predecessor array (caller frees, may be nullptr)
template <typename Index, typename Weight>
void compute_sssp_gpu(sycl::queue& queue,
                      const std::int64_t* rows,
                      const Index* cols,
                      const Weight* weights,
                      std::int64_t vertex_count,
                      std::int64_t edge_count,
                      std::int64_t source,
                      bool compute_predecessors,
                      Weight* distances,
                      std::int32_t* predecessors) {
    constexpr Weight inf = std::numeric_limits<Weight>::max();

    // Initialize distances to infinity, source to 0
    queue
        .submit([&](sycl::handler& cgh) {
            Weight* d_dist = distances;
            const std::int64_t vc = vertex_count;
            const std::int64_t src = source;
            cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                d_dist[idx[0]] = (static_cast<std::int64_t>(idx[0]) == src)
                                     ? Weight{ 0 }
                                     : inf;
            });
        })
        .wait_and_throw();

    // Initialize predecessors to -1 if requested
    if (compute_predecessors && predecessors != nullptr) {
        queue.memset(predecessors, 0xFF, vertex_count * sizeof(std::int32_t)).wait_and_throw();
    }

    // Fast path: if no edges, only source has distance 0
    if (edge_count == 0) {
        return;
    }

    // Create graph wrapper referencing the existing device CSR topology
    // VertexT = Index (column indices), WeightT = Weight, OffsetT = int64_t
    fp::csr_graph_external<Index, std::uint32_t, Weight, std::int64_t> graph(
        queue,
        static_cast<std::uint64_t>(vertex_count),
        rows,
        cols,
        weights);

    // Create two-layer bitmap frontiers for double-buffering
    fp::frontier<std::uint32_t> in_frontier(queue, vertex_count, sycl::usm::alloc::device);
    fp::frontier<std::uint32_t> out_frontier(queue, vertex_count, sycl::usm::alloc::device);

    // Initialize: only source vertex is active
    {
        auto fv = in_frontier.get_device_view();
        const std::uint32_t src32 = static_cast<std::uint32_t>(source);
        queue
            .submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                    fv.insert(src32);
                });
            })
            .wait_and_throw();
    }

    // Label-correcting SSSP with workload-balanced frontier advance
    Weight* d_dist = distances;
    std::int32_t* d_pred = predecessors;
    const bool track_pred = compute_predecessors && (predecessors != nullptr);

    while (!in_frontier.empty()) {
        auto e = fp::advance(
            graph,
            in_frontier,
            out_frontier,
            [=](auto src, auto dst, auto edge, auto weight) -> bool {
                const Weight dist_src = d_dist[src];

                // Skip if source is unreachable or if addition would overflow
                // (for integer weight types, dist_src + weight can wrap to
                // negative, causing an infinite relaxation loop)
                if (dist_src == inf || weight >= inf - dist_src) {
                    return false;
                }

                const Weight new_dist = dist_src + weight;
                const Weight old_dist =
                    sycl::atomic_ref<Weight,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(d_dist[dst])
                        .fetch_min(new_dist);

                // If we improved the distance, add dst to output frontier
                if (new_dist < old_dist) {
                    if (track_pred) {
                        // Store predecessor (last writer wins, which is fine
                        // because the distance is also atomically minimized)
                        d_pred[dst] = static_cast<std::int32_t>(src);
                    }
                    return true;
                }
                return false;
            });
        e.wait_and_throw();

        fp::swap_frontiers(in_frontier, out_frontier);
        out_frontier.clear();
    }
}

} // namespace detail_gpu

template <typename Float, typename Task, typename Index, typename Weight>
traverse_result<Task> run_shortest_paths_gpu(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const csr_topology_gpu_view<Index, Weight>& topology) {
    auto& queue = ctx.get_queue();

    const auto vertex_count = topology.vertex_count;
    const auto edge_count = topology.edge_count;
    const auto source = desc.get_source();

    if (vertex_count == 0) {
        return traverse_result<Task>();
    }

    const bool need_predecessors =
        static_cast<bool>(desc.get_optional_results() & optional_results::predecessors);
    const bool need_distances =
        static_cast<bool>(desc.get_optional_results() & optional_results::distances);

    // Allocate output arrays
    auto* distances = sycl::malloc_shared<Weight>(vertex_count, queue);
    std::int32_t* predecessors = nullptr;
    if (need_predecessors) {
        predecessors = sycl::malloc_shared<std::int32_t>(vertex_count, queue);
    }

    detail_gpu::compute_sssp_gpu(queue,
                                 topology.rows,
                                 topology.cols,
                                 topology.weights,
                                 vertex_count,
                                 edge_count,
                                 source,
                                 need_predecessors,
                                 distances,
                                 predecessors);

    traverse_result<Task> result;

    if (need_distances) {
        auto dist_arr =
            array<Weight>(queue, distances, vertex_count, [queue](Weight* ptr) mutable {
                sycl::free(ptr, queue);
            });
        result.set_distances(
            dal::detail::homogen_table_builder{}
                .reset(dist_arr, vertex_count, 1)
                .build());
    }
    else {
        sycl::free(distances, queue);
    }

    if (need_predecessors && predecessors != nullptr) {
        auto pred_arr = array<std::int32_t>(
            queue,
            predecessors,
            vertex_count,
            [queue](std::int32_t* ptr) mutable { sycl::free(ptr, queue); });
        result.set_predecessors(
            dal::detail::homogen_table_builder{}
                .reset(pred_arr, vertex_count, 1)
                .build());
    }

    return result;
}

template <typename Float, typename Task, typename Topology, typename EdgeValue>
traverse_result<Task> traverse_kernel_gpu<Float, Task, Topology, EdgeValue>::operator()(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const Topology& t,
    const EdgeValue* edge_values) const {
    using index_type = typename Topology::vertex_type;
    auto& queue = ctx.get_queue();

    const auto vertex_count = t.get_vertex_count();
    const auto edge_count = t.get_edge_count();

    if (vertex_count == 0) {
        return traverse_result<Task>();
    }

    // Transfer topology to device
    auto device_topo =
        dal::preview::detail::topology_to_device<index_type>(queue, t);

    // Transfer edge weights to device
    const std::int64_t weights_count = t._cols.get_count();
    dal::array<EdgeValue> device_weights;
    if (weights_count > 0) {
        device_weights =
            dal::array<EdgeValue>::empty(queue, weights_count, sycl::usm::alloc::device);
        queue.memcpy(device_weights.get_mutable_data(),
                     edge_values,
                     weights_count * sizeof(EdgeValue))
            .wait_and_throw();
    }

    csr_topology_gpu_view<index_type, EdgeValue> gpu_view;
    gpu_view.rows = device_topo.get_rows();
    gpu_view.cols = device_topo.get_cols();
    gpu_view.weights = device_weights.get_count() > 0 ? device_weights.get_data() : nullptr;
    gpu_view.vertex_count = vertex_count;
    gpu_view.edge_count = edge_count;

    return run_shortest_paths_gpu<Float, Task>(ctx, desc, gpu_view);
}

// Explicit instantiations
template struct traverse_kernel_gpu<float,
                                    task::one_to_all,
                                    dal::preview::detail::topology<std::int32_t>,
                                    std::int32_t>;

template struct traverse_kernel_gpu<float,
                                    task::one_to_all,
                                    dal::preview::detail::topology<std::int32_t>,
                                    double>;

template traverse_result<task::one_to_all>
run_shortest_paths_gpu<float, task::one_to_all, std::int32_t, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::one_to_all>&,
    const csr_topology_gpu_view<std::int32_t, std::int32_t>&);

template traverse_result<task::one_to_all>
run_shortest_paths_gpu<float, task::one_to_all, std::int32_t, double>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::one_to_all>&,
    const csr_topology_gpu_view<std::int32_t, double>&);

} // namespace oneapi::dal::preview::shortest_paths::backend
