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

#include "oneapi/dal/algo/connected_components/backend/gpu/connected_components.hpp"
#include "oneapi/dal/algo/connected_components/backend/gpu/vertex_partitioning_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::connected_components::backend {

namespace detail_gpu {

/// Computes connected components on the GPU using a Shiloach-Vishkin-style
/// hook-and-compress approach.
///
/// @tparam Index  The index type used for vertex/edge indexing
/// @param[in]  queue         SYCL queue for device execution
/// @param[in]  rows          CSR row-offset array (vertex_count + 1 entries)
/// @param[in]  cols          CSR column-index array (2 * edge_count entries)
/// @param[in]  vertex_count  Number of vertices in the graph
/// @param[in]  edge_count    Number of undirected edges
/// @param[out] component_count  The number of connected components found
/// @return     A USM-allocated array of per-vertex component labels
template <typename Index>
std::int32_t* compute_components_gpu(sycl::queue& queue,
                                     const Index* rows,
                                     const Index* cols,
                                     std::int64_t vertex_count,
                                     std::int64_t edge_count,
                                     std::int64_t& component_count) {
    auto* components = sycl::malloc_shared<std::int32_t>(vertex_count, queue);
    auto* changed = sycl::malloc_shared<std::int32_t>(1, queue);

    // Initialize each vertex as its own component
    queue
        .submit([&](sycl::handler& cgh) {
            std::int32_t* d_components = components;
            const std::int64_t vc = vertex_count;
            cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                d_components[idx[0]] = static_cast<std::int32_t>(idx[0]);
            });
        })
        .wait_and_throw();

    // Iteratively hook and compress until convergence
    bool has_changed = true;
    while (has_changed) {
        changed[0] = 0;

        // Hook: for each edge (u, v), point the higher component root
        // toward the lower component root
        queue
            .submit([&](sycl::handler& cgh) {
                const Index* d_rows = rows;
                const Index* d_cols = cols;
                std::int32_t* d_components = components;
                std::int32_t* d_changed = changed;
                const std::int64_t vc = vertex_count;

                cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                    const std::int64_t u = idx[0];
                    const std::int64_t u_start = d_rows[u];
                    const std::int64_t u_end = d_rows[u + 1];

                    std::int32_t comp_u = d_components[u];

                    for (std::int64_t i = u_start; i < u_end; ++i) {
                        const std::int64_t v = d_cols[i];
                        std::int32_t comp_v = d_components[v];

                        // Link higher-labeled component root to lower one
                        if (comp_u < comp_v) {
                            sycl::atomic_ref<std::int32_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(
                                d_components[comp_v])
                                .fetch_min(comp_u);
                            d_changed[0] = 1;
                        }
                        else if (comp_v < comp_u) {
                            sycl::atomic_ref<std::int32_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(
                                d_components[comp_u])
                                .fetch_min(comp_v);
                            d_changed[0] = 1;
                        }
                    }
                });
            })
            .wait_and_throw();

        // Compress: path compression so each vertex points directly to root
        queue
            .submit([&](sycl::handler& cgh) {
                std::int32_t* d_components = components;
                const std::int64_t vc = vertex_count;

                cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                    std::int32_t v = static_cast<std::int32_t>(idx[0]);
                    while (d_components[v] != d_components[d_components[v]]) {
                        d_components[v] = d_components[d_components[v]];
                    }
                });
            })
            .wait_and_throw();

        has_changed = (changed[0] != 0);
    }

    // Count components and reorder labels
    auto* ordered_labels = sycl::malloc_shared<std::int32_t>(vertex_count, queue);
    auto* comp_count_ptr = sycl::malloc_shared<std::int32_t>(1, queue);
    comp_count_ptr[0] = 0;

    // Initialize ordered_labels to -1
    queue.memset(ordered_labels, 0xFF, vertex_count * sizeof(std::int32_t)).wait_and_throw();

    // Sequential pass to assign ordered component IDs
    for (std::int64_t u = 0; u < vertex_count; ++u) {
        std::int32_t root = components[u];
        if (ordered_labels[root] < 0) {
            ordered_labels[root] = comp_count_ptr[0];
            comp_count_ptr[0]++;
        }
        components[u] = ordered_labels[root];
    }

    component_count = comp_count_ptr[0];

    sycl::free(changed, queue);
    sycl::free(ordered_labels, queue);
    sycl::free(comp_count_ptr, queue);

    return components;
}

} // namespace detail_gpu

template <typename Float, typename Task, typename Index>
vertex_partitioning_result<Task> run_vertex_partitioning_gpu(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const csr_topology_gpu_view<Index>& topology) {
    auto& queue = ctx.get_queue();

    const auto vertex_count = topology.vertex_count;
    const auto edge_count = topology.edge_count;

    if (vertex_count == 0) {
        return vertex_partitioning_result<Task>();
    }

    std::int64_t component_count = 0;
    auto* labels = detail_gpu::compute_components_gpu(queue,
                                                      topology.rows,
                                                      topology.cols,
                                                      vertex_count,
                                                      edge_count,
                                                      component_count);

    auto labels_arr =
        array<std::int32_t>(queue, labels, vertex_count, [queue](std::int32_t* ptr) mutable {
            sycl::free(ptr, queue);
        });

    vertex_partitioning_result<Task> result;
    result.set_labels(dal::detail::homogen_table_builder{}
                          .reset(labels_arr, vertex_count, 1)
                          .build());
    result.set_component_count(component_count);

    return result;
}

template <typename Float, typename Task, typename Topology>
vertex_partitioning_result<Task> vertex_partitioning_kernel_gpu<Float, Task, Topology>::operator()(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const Topology& topology) const {
    auto& queue = ctx.get_queue();
    const auto vertex_count = topology.get_vertex_count();

    if (vertex_count == 0) {
        return vertex_partitioning_result<Task>();
    }

    // Transfer the host topology to device memory using the graph transfer utility.
    // This copies row offsets (int64 -> int32 conversion) and column indices to device.
    auto device_topo =
        dal::preview::detail::topology_to_device<std::int32_t>(queue, topology);

    // Create a lightweight GPU view from the device topology (zero-copy)
    const auto gpu_view = make_gpu_view(device_topo);

    return run_vertex_partitioning_gpu<Float, Task, std::int32_t>(ctx, desc, gpu_view);
}

// Explicit template instantiations
template struct vertex_partitioning_kernel_gpu<float,
                                               task::vertex_partitioning,
                                               dal::preview::detail::topology<std::int32_t>>;

template vertex_partitioning_result<task::vertex_partitioning>
run_vertex_partitioning_gpu<float, task::vertex_partitioning, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::vertex_partitioning>&,
    const csr_topology_gpu_view<std::int32_t>&);

} // namespace oneapi::dal::preview::connected_components::backend
