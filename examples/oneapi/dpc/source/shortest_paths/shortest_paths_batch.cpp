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
#include <iomanip>
#include <iostream>
#include <memory>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/shortest_paths.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& q) {
    const auto filename = get_data_path("data/weighted_edge_list.csv");

    // Read the weighted directed graph from CSV
    using vertex_type = std::int32_t;
    using weight_type = double;
    using graph_t = dal::preview::directed_adjacency_vector_graph<vertex_type, weight_type>;

    const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename },
                                          dal::preview::read_mode::weighted_edge_list);

    // Set algorithm parameters: source vertex 0, delta 0.85,
    // compute both distances and predecessors
    const auto sp_desc = dal::preview::shortest_paths::descriptor<
        float,
        dal::preview::shortest_paths::method::delta_stepping,
        dal::preview::shortest_paths::task::one_to_all>(
        0,
        0.85,
        dal::preview::shortest_paths::optional_results::distances |
            dal::preview::shortest_paths::optional_results::predecessors);

    // Compute shortest paths on the SYCL device (CPU or GPU)
    const auto result = dal::preview::traverse(q, sp_desc, graph);

    // Extract and print the results
    std::cout << "Distances:" << std::endl;
    std::cout << result.get_distances() << std::endl;
    std::cout << "Predecessors:" << std::endl;
    std::cout << result.get_predecessors() << std::endl;
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
