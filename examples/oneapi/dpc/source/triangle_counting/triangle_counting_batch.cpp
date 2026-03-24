/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <iomanip>
#include <iostream>
#include <memory>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/triangle_counting.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::triangle_counting;

void run(sycl::queue& q) {
    const auto filename = get_data_path("data/graph.csv");

    // Read the graph from CSV into host memory
    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename });

    // Explicitly transfer the graph topology from host to device.
    // This copies CSR row offsets and column indices to device USM memory.
    const auto& host_topo = dal::detail::get_impl(graph).get_topology();
    auto device_topo =
        dal::preview::detail::topology_to_device<std::int32_t>(q, host_topo);

    std::cout << "Graph transferred to device: "
              << device_topo.get_vertex_count() << " vertices, "
              << device_topo.get_edge_count() << " edges" << std::endl;

    // Set algorithm parameters: compute both local and global triangle counts
    const auto tc_desc = descriptor<float, method::ordered_count, task::local_and_global>();

    // Run triangle counting on the SYCL device (CPU or GPU).
    // The kernel internally transfers the host graph to device for computation.
    const auto result = dal::preview::vertex_ranking(q, tc_desc, graph);

    // Extract and print the results
    std::cout << "Global triangles: " << result.get_global_rank() << std::endl;
    std::cout << "Local triangles:" << std::endl;

    auto local_triangles_table = result.get_ranks();
    const auto& local_triangles = static_cast<const dal::homogen_table&>(local_triangles_table);
    const auto local_triangles_data = local_triangles.get_data<std::int64_t>();
    for (auto i = 0; i < local_triangles_table.get_row_count(); i++) {
        std::cout << i << ":\t" << local_triangles_data[i] << std::endl;
    }

    // Demonstrate round-trip: transfer back to host and verify
    auto roundtrip_topo = dal::preview::detail::topology_to_host(device_topo);
    std::cout << "\nRound-trip verification: "
              << roundtrip_topo.get_vertex_count() << " vertices, "
              << roundtrip_topo.get_edge_count() << " edges" << std::endl;
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
