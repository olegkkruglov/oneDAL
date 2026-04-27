/*******************************************************************************
* Copyright contributors to the oneDAL project
* Copyright 2025 University of Salerno
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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/graph/detail/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::preview::backend::primitives {

/// Device-copyable CSR graph view for use inside SYCL kernels.
///
/// @tparam VertexT   Type of column indices (neighbor IDs)
/// @tparam EdgeT     Type used for edge indexing (iterator distance)
/// @tparam WeightT   Type of edge weights (ignored when weights are null)
/// @tparam OffsetT   Type of row offsets (e.g., int64_t for large graphs)
template <typename VertexT = std::uint32_t,
          typename EdgeT = std::uint32_t,
          typename WeightT = std::uint32_t,
          typename OffsetT = VertexT>
class csr_graph_view {
    using vertex_t = VertexT;
    using edge_t = EdgeT;
    using weight_t = WeightT;
    using offset_t = OffsetT;

    struct neighbor_iterator_t {
        neighbor_iterator_t(const vertex_t* start_ptr, const vertex_t* ptr)
                : _ptr(ptr),
                  _start_ptr(start_ptr) {}

        SYCL_EXTERNAL bool operator==(const neighbor_iterator_t& other) const {
            return _ptr == other._ptr;
        }

        SYCL_EXTERNAL bool operator!=(const neighbor_iterator_t& other) const {
            return _ptr != other._ptr;
        }

        SYCL_EXTERNAL neighbor_iterator_t& operator++() {
            ++_ptr;
            return *this;
        }

        SYCL_EXTERNAL neighbor_iterator_t operator+(int n) const {
            return neighbor_iterator_t(_start_ptr, _ptr + n);
        }

        SYCL_EXTERNAL vertex_t operator*() const {
            return *_ptr;
        }

        SYCL_EXTERNAL edge_t get_index() const {
            return static_cast<edge_t>(_ptr - _start_ptr);
        }

        const vertex_t* _ptr;
        const vertex_t* _start_ptr;
    };

public:
    /// Constructs a graph view. Weights may be nullptr for unweighted graphs.
    csr_graph_view(std::uint64_t num_nodes,
                   const offset_t* row_ptr,
                   const vertex_t* col_indices,
                   const weight_t* weights = nullptr)
            : _num_nodes(num_nodes),
              _row_ptr(row_ptr),
              _col_indices(col_indices),
              _weights(weights) {
        ONEDAL_ASSERT(_row_ptr != nullptr, "Row pointer must not be null");
        ONEDAL_ASSERT(_col_indices != nullptr, "Column indices must not be null");
    }

    SYCL_EXTERNAL inline std::uint32_t get_degree(const vertex_t vertex) const {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(vertex) < _num_nodes, "Vertex index out of bounds");
        return static_cast<std::uint32_t>(_row_ptr[vertex + 1] - _row_ptr[vertex]);
    }

    SYCL_EXTERNAL inline weight_t get_weight(const edge_t edge) const {
        if (_weights) {
            return _weights[edge];
        }
        return weight_t{ 0 };
    }

    SYCL_EXTERNAL inline neighbor_iterator_t begin(vertex_t vertex) const {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(vertex) < _num_nodes, "Vertex index out of bounds");
        return neighbor_iterator_t(_col_indices, _col_indices + _row_ptr[vertex]);
    }

    SYCL_EXTERNAL inline neighbor_iterator_t end(vertex_t vertex) const {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(vertex) < _num_nodes, "Vertex index out of bounds");
        return neighbor_iterator_t(_col_indices, _col_indices + _row_ptr[vertex + 1]);
    }

private:
    std::uint64_t _num_nodes;
    const offset_t* _row_ptr;
    const vertex_t* _col_indices;
    const weight_t* _weights;
};

/// Host-side CSR graph that owns device USM arrays and provides a device view.
///
/// For algorithms that already have device-resident CSR data (e.g., from
/// device_csr_topology), use csr_graph_external instead of copying again.
///
/// @tparam VertexT   Type of column indices
/// @tparam EdgeT     Type for edge indexing
/// @tparam WeightT   Type of edge weights
/// @tparam OffsetT   Type of row offsets
template <typename VertexT = std::uint32_t,
          typename EdgeT = std::uint32_t,
          typename WeightT = std::uint32_t,
          typename OffsetT = VertexT>
class csr_graph {
    using vertex_t = VertexT;
    using edge_t = EdgeT;
    using weight_t = WeightT;
    using offset_t = OffsetT;
    using graph_view_t = csr_graph_view<vertex_t, edge_t, weight_t, offset_t>;

public:
    csr_graph(sycl::queue& queue,
              std::vector<OffsetT> row_ptr,
              std::vector<VertexT> col_indices,
              std::vector<WeightT> weights = {},
              sycl::usm::alloc alloc = sycl::usm::alloc::shared)
            : _queue(queue),
              _num_nodes(row_ptr.size() - 1),
              _has_weights(!weights.empty()) {
        std::int64_t row_ptr_size = static_cast<std::int64_t>(row_ptr.size());
        std::int64_t col_indices_size = static_cast<std::int64_t>(col_indices.size());

        _row_ptr = pr::ndarray<offset_t, 1>::empty(_queue, { row_ptr_size }, alloc);
        _col_indices = pr::ndarray<vertex_t, 1>::empty(_queue, { col_indices_size }, alloc);

        auto e1 = dal::backend::copy_host2usm(_queue,
                                              _row_ptr.get_mutable_data(),
                                              row_ptr.data(),
                                              row_ptr.size());

        auto e2 = dal::backend::copy_host2usm(_queue,
                                              _col_indices.get_mutable_data(),
                                              col_indices.data(),
                                              col_indices.size());

        e1.wait_and_throw();
        e2.wait_and_throw();

        if (_has_weights) {
            std::int64_t weights_size = static_cast<std::int64_t>(weights.size());
            _weights = pr::ndarray<weight_t, 1>::empty(_queue, { weights_size }, alloc);
            auto e3 = dal::backend::copy_host2usm(_queue,
                                                  _weights.get_mutable_data(),
                                                  weights.data(),
                                                  weights.size());
            e3.wait_and_throw();
        }
    }

    graph_view_t get_device_view() const {
        return { _num_nodes,
                 _row_ptr.get_mutable_data(),
                 _col_indices.get_mutable_data(),
                 _has_weights ? _weights.get_mutable_data() : nullptr };
    }

    sycl::queue& get_queue() const {
        return _queue;
    }

    std::uint64_t get_vertex_count() const {
        return _num_nodes;
    }

private:
    sycl::queue& _queue;
    std::uint64_t _num_nodes;
    bool _has_weights;
    pr::ndarray<offset_t, 1> _row_ptr;
    pr::ndarray<vertex_t, 1> _col_indices;
    pr::ndarray<weight_t, 1> _weights;
};

/// Lightweight host-side wrapper that binds existing device pointers + queue
/// into the interface expected by advance(). No data copies — just references.
///
/// Use this when CSR data is already on the device (e.g., from device_csr_topology).
///
/// @tparam VertexT   Type of column indices
/// @tparam EdgeT     Type for edge indexing
/// @tparam WeightT   Type of edge weights
/// @tparam OffsetT   Type of row offsets
template <typename VertexT = std::uint32_t,
          typename EdgeT = std::uint32_t,
          typename WeightT = std::uint32_t,
          typename OffsetT = VertexT>
class csr_graph_external {
    using graph_view_t = csr_graph_view<VertexT, EdgeT, WeightT, OffsetT>;

public:
    csr_graph_external(sycl::queue& queue,
                       std::uint64_t vertex_count,
                       const OffsetT* row_ptr,
                       const VertexT* col_indices,
                       const WeightT* weights = nullptr)
            : _queue(queue),
              _vertex_count(vertex_count),
              _row_ptr(row_ptr),
              _col_indices(col_indices),
              _weights(weights) {}

    graph_view_t get_device_view() const {
        return { _vertex_count, _row_ptr, _col_indices, _weights };
    }

    sycl::queue& get_queue() const {
        return _queue;
    }

    std::uint64_t get_vertex_count() const {
        return _vertex_count;
    }

private:
    sycl::queue& _queue;
    std::uint64_t _vertex_count;
    const OffsetT* _row_ptr;
    const VertexT* _col_indices;
    const WeightT* _weights;
};

} // namespace oneapi::dal::preview::backend::primitives
