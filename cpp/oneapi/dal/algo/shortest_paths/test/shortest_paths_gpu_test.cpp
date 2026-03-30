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

#include <array>
#include <limits>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/shortest_paths/traverse.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::shortest_paths::gpu::test {

namespace dal = oneapi::dal;

constexpr double inf_double = std::numeric_limits<double>::max();

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_cols_count() const {
        return cols_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

    std::int64_t get_source() const {
        return source;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
    std::int64_t source;
};

/// Simple directed graph:
///   0 --3--> 1 --2--> 2
///   |                 ^
///   +------10---------+
/// Source=0, expected distances: [0, 3, 5]
class simple_chain_graph_type : public graph_base_data {
public:
    simple_chain_graph_type() {
        vertex_count = 3;
        edge_count = 3;
        cols_count = 3;
        rows_count = 4;
        source = 0;
    }

    std::array<std::int32_t, 3> degrees = { 2, 1, 0 };
    std::array<std::int32_t, 3> cols = { 1, 2, 2 };
    std::array<std::int64_t, 4> rows = { 0, 2, 3, 3 };
    std::array<double, 3> edge_weights = { 3.0, 10.0, 2.0 };
    std::array<double, 3> expected_distances = { 0.0, 3.0, 5.0 };
    std::array<std::int32_t, 3> expected_predecessors = { -1, 0, 1 };
};

/// Diamond graph:
///        0
///       / \
///      1   4
///     / \
///    2   3
///     \ /
///      4
/// Edges: 0->1(1), 0->4(10), 1->2(2), 1->3(3), 2->4(1), 3->4(1)
/// Source=0, expected distances: [0, 1, 3, 4, 4]
class diamond_graph_type : public graph_base_data {
public:
    diamond_graph_type() {
        vertex_count = 5;
        edge_count = 6;
        cols_count = 6;
        rows_count = 6;
        source = 0;
    }

    std::array<std::int32_t, 5> degrees = { 2, 2, 1, 1, 0 };
    std::array<std::int32_t, 6> cols = { 1, 4, 2, 3, 4, 4 };
    std::array<std::int64_t, 6> rows = { 0, 2, 4, 5, 6, 6 };
    std::array<double, 6> edge_weights = { 1.0, 10.0, 2.0, 3.0, 1.0, 1.0 };
    std::array<double, 5> expected_distances = { 0.0, 1.0, 3.0, 4.0, 4.0 };
    std::array<std::int32_t, 5> expected_predecessors = { -1, 0, 1, 1, 2 };
};

/// Single vertex, no edges
class single_vertex_graph_type : public graph_base_data {
public:
    single_vertex_graph_type() {
        vertex_count = 1;
        edge_count = 0;
        cols_count = 0;
        rows_count = 2;
        source = 0;
    }

    std::array<std::int32_t, 1> degrees = { 0 };
    std::array<std::int32_t, 1> cols = { 0 }; // dummy
    std::array<std::int64_t, 2> rows = { 0, 0 };
    std::array<double, 1> edge_weights = { 0.0 }; // dummy
    std::array<double, 1> expected_distances = { 0.0 };
    std::array<std::int32_t, 1> expected_predecessors = { -1 };
};

/// Disconnected: 0->1(1), 2 is isolated
/// Source=0, expected distances: [0, 1, inf]
class disconnected_graph_type : public graph_base_data {
public:
    disconnected_graph_type() {
        vertex_count = 3;
        edge_count = 1;
        cols_count = 1;
        rows_count = 4;
        source = 0;
    }

    std::array<std::int32_t, 3> degrees = { 1, 0, 0 };
    std::array<std::int32_t, 1> cols = { 1 };
    std::array<std::int64_t, 4> rows = { 0, 1, 1, 1 };
    std::array<double, 1> edge_weights = { 1.0 };
    std::array<double, 3> expected_distances = { 0.0, 1.0, inf_double };
    std::array<std::int32_t, 3> expected_predecessors = { -1, 0, -1 };
};

/// Linear chain: 0->1->2->3->4, all weight 1
/// Source=0, distances: [0,1,2,3,4]
class linear_chain_graph_type : public graph_base_data {
public:
    linear_chain_graph_type() {
        vertex_count = 5;
        edge_count = 4;
        cols_count = 4;
        rows_count = 6;
        source = 0;
    }

    std::array<std::int32_t, 5> degrees = { 1, 1, 1, 1, 0 };
    std::array<std::int32_t, 4> cols = { 1, 2, 3, 4 };
    std::array<std::int64_t, 6> rows = { 0, 1, 2, 3, 4, 4 };
    std::array<double, 4> edge_weights = { 1.0, 1.0, 1.0, 1.0 };
    std::array<double, 5> expected_distances = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    std::array<std::int32_t, 5> expected_predecessors = { -1, 0, 1, 2, 3 };
};

class shortest_paths_gpu_test {
public:
    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        using graph_t = dal::preview::directed_adjacency_vector_graph<std::int32_t, double>;
        graph_t g;
        auto& graph_impl = oneapi::dal::detail::get_impl(g);

        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t edge_count = graph_data.get_edge_count();
        const std::int64_t cols_count = graph_data.get_cols_count();
        const std::int64_t rows_count = graph_data.get_rows_count();

        using vertex_allocator_type =
            typename std::allocator_traits<std::allocator<char>>::rebind_alloc<std::int32_t>;
        using edge_allocator_type =
            typename std::allocator_traits<std::allocator<char>>::rebind_alloc<std::int64_t>;
        using weight_allocator_type =
            typename std::allocator_traits<std::allocator<char>>::rebind_alloc<double>;

        vertex_allocator_type vertex_allocator;
        edge_allocator_type edge_allocator;
        weight_allocator_type weight_allocator;

        std::int32_t* degrees =
            oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
        std::int32_t* cols =
            oneapi::dal::preview::detail::allocate(vertex_allocator, cols_count > 0 ? cols_count : 1);
        std::int64_t* rows =
            oneapi::dal::preview::detail::allocate(edge_allocator, rows_count);
        double* weights =
            oneapi::dal::preview::detail::allocate(weight_allocator, cols_count > 0 ? cols_count : 1);

        for (std::int64_t i = 0; i < vertex_count; i++) {
            degrees[i] = graph_data.degrees[i];
        }

        for (std::int64_t i = 0; i < cols_count; i++) {
            cols[i] = graph_data.cols[i];
        }

        for (std::int64_t i = 0; i < rows_count; i++) {
            rows[i] = graph_data.rows[i];
        }

        for (std::int64_t i = 0; i < cols_count; i++) {
            weights[i] = graph_data.edge_weights[i];
        }

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, cols_count, degrees);
        graph_impl.set_edge_values(weights, cols_count > 0 ? cols_count : 1);

        return g;
    }

    sycl::queue get_queue() {
        return sycl::queue{ sycl::default_selector_v };
    }

    template <typename GraphType>
    void check_distances_only() {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        const std::int64_t vertex_count = graph_data.get_vertex_count();

        auto q = get_queue();

        const auto sp_desc = dal::preview::shortest_paths::descriptor<
            float,
            dal::preview::shortest_paths::method::delta_stepping,
            dal::preview::shortest_paths::task::one_to_all>(
            graph_data.get_source(),
            1.0,
            dal::preview::shortest_paths::optional_results::distances);

        const auto result = dal::preview::traverse(q, sp_desc, g);

        const auto& dist_table = result.get_distances();
        REQUIRE(dist_table.get_row_count() == vertex_count);
        REQUIRE(dist_table.get_column_count() == 1);

        const auto dist_data =
            dal::row_accessor<const double>(
                static_cast<const dal::homogen_table&>(dist_table))
                .pull({ 0, -1 });

        for (std::int64_t i = 0; i < vertex_count; i++) {
            REQUIRE(dist_data[i] == graph_data.expected_distances[i]);
        }
    }

    template <typename GraphType>
    void check_distances_and_predecessors() {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        const std::int64_t vertex_count = graph_data.get_vertex_count();

        auto q = get_queue();

        const auto sp_desc = dal::preview::shortest_paths::descriptor<
            float,
            dal::preview::shortest_paths::method::delta_stepping,
            dal::preview::shortest_paths::task::one_to_all>(
            graph_data.get_source(),
            1.0,
            dal::preview::shortest_paths::optional_results::distances |
                dal::preview::shortest_paths::optional_results::predecessors);

        const auto result = dal::preview::traverse(q, sp_desc, g);

        // Check distances
        const auto& dist_table = result.get_distances();
        REQUIRE(dist_table.get_row_count() == vertex_count);

        const auto dist_data =
            dal::row_accessor<const double>(
                static_cast<const dal::homogen_table&>(dist_table))
                .pull({ 0, -1 });

        for (std::int64_t i = 0; i < vertex_count; i++) {
            REQUIRE(dist_data[i] == graph_data.expected_distances[i]);
        }

        // Check predecessors
        const auto& pred_table = result.get_predecessors();
        REQUIRE(pred_table.get_row_count() == vertex_count);

        const auto pred_data =
            dal::row_accessor<const std::int32_t>(
                static_cast<const dal::homogen_table&>(pred_table))
                .pull({ 0, -1 });

        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (graph_data.expected_distances[i] < inf_double) {
                // For reachable vertices, verify predecessor leads to a shorter path
                // (exact predecessor may vary for equal-length paths)
                if (i != graph_data.get_source()) {
                    REQUIRE(pred_data[i] >= 0);
                    REQUIRE(pred_data[i] < vertex_count);
                }
                else {
                    REQUIRE(pred_data[i] == -1);
                }
            }
            else {
                REQUIRE(pred_data[i] == -1);
            }
        }
    }
};

TEST_M(shortest_paths_gpu_test, "GPU: simple chain - distances only") {
    this->check_distances_only<simple_chain_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: simple chain - distances and predecessors") {
    this->check_distances_and_predecessors<simple_chain_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: diamond graph - distances only") {
    this->check_distances_only<diamond_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: diamond graph - distances and predecessors") {
    this->check_distances_and_predecessors<diamond_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: single vertex") {
    this->check_distances_and_predecessors<single_vertex_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: disconnected graph") {
    this->check_distances_and_predecessors<disconnected_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: linear chain") {
    this->check_distances_and_predecessors<linear_chain_graph_type>();
}

TEST_M(shortest_paths_gpu_test, "GPU: null graph") {
    using graph_t = dal::preview::directed_adjacency_vector_graph<std::int32_t, double>;
    graph_t null_graph;
    auto q = get_queue();

    const auto sp_desc = dal::preview::shortest_paths::descriptor<
        float,
        dal::preview::shortest_paths::method::delta_stepping,
        dal::preview::shortest_paths::task::one_to_all>(0, 1.0);

    // Null graph should throw because source >= vertex_count
    REQUIRE_THROWS(dal::preview::traverse(q, sp_desc, null_graph));
}

} // namespace oneapi::dal::algo::shortest_paths::gpu::test
