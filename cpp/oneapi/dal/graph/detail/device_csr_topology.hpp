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

#pragma once

#ifdef ONEDAL_DATA_PARALLEL

#include <sycl/sycl.hpp>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/graph/detail/csr_topology.hpp"

namespace oneapi::dal::preview::detail {

/// A device-resident CSR topology that holds row offsets and column indices
/// as dal::array objects in device USM memory.
///
/// Unlike the host \ref topology class which stores row offsets as int64 and
/// column indices as IndexType, this class normalizes both arrays to IndexType
/// (typically int32) to avoid type mismatches in GPU kernels.
///
/// Construct via \ref topology_to_device() or \ref topology_to_host().
template <typename IndexType = std::int32_t>
class device_csr_topology {
public:
    using index_type = IndexType;

    device_csr_topology() = default;

    device_csr_topology(dal::array<IndexType> device_rows,
                        dal::array<IndexType> device_cols,
                        std::int64_t vertex_count,
                        std::int64_t edge_count)
            : rows_(std::move(device_rows)),
              cols_(std::move(device_cols)),
              vertex_count_(vertex_count),
              edge_count_(edge_count) {}

    /// Pointer to device-resident row offsets (vertex_count + 1 entries).
    const IndexType* get_rows() const {
        return rows_.get_data();
    }

    /// Pointer to device-resident column indices (2 * edge_count entries).
    const IndexType* get_cols() const {
        return cols_.get_data();
    }

    std::int64_t get_vertex_count() const {
        return vertex_count_;
    }

    std::int64_t get_edge_count() const {
        return edge_count_;
    }

    /// Returns the underlying device array for row offsets.
    const dal::array<IndexType>& get_rows_array() const {
        return rows_;
    }

    /// Returns the underlying device array for column indices.
    const dal::array<IndexType>& get_cols_array() const {
        return cols_;
    }

private:
    dal::array<IndexType> rows_;
    dal::array<IndexType> cols_;
    std::int64_t vertex_count_ = 0;
    std::int64_t edge_count_ = 0;
};

/// Transfers a host topology to device memory, converting row offsets from
/// int64 to IndexType (int32) on the device.
///
/// @tparam IndexType  The index type for the device topology (default int32)
/// @param queue       SYCL queue identifying the target device
/// @param host_topo   Host-resident CSR topology
/// @return A device_csr_topology with arrays in device USM memory
template <typename IndexType = std::int32_t>
device_csr_topology<IndexType> topology_to_device(sycl::queue& queue,
                                                  const topology<IndexType>& host_topo) {
    const auto vertex_count = host_topo.get_vertex_count();
    const auto edge_count = host_topo.get_edge_count();

    if (vertex_count == 0) {
        return device_csr_topology<IndexType>();
    }

    const std::int64_t rows_count = vertex_count + 1;

    // Transfer column indices (IndexType -> IndexType) directly to device
    auto device_cols = dal::backend::to_device_sync(queue, host_topo._cols);

    // Row offsets are stored as int64 in host topology but we need IndexType
    // on device. Copy int64 to device, then convert with a parallel kernel.
    auto device_rows_i64 = dal::backend::to_device_sync(queue, host_topo._rows);

    auto device_rows = dal::array<IndexType>::empty(queue, rows_count, sycl::usm::alloc::device);
    queue
        .submit([&](sycl::handler& cgh) {
            const auto* src = device_rows_i64.get_data();
            auto* dst = device_rows.get_mutable_data();
            const auto n = rows_count;
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                dst[idx[0]] = static_cast<IndexType>(src[idx[0]]);
            });
        })
        .wait_and_throw();

    return device_csr_topology<IndexType>(std::move(device_rows),
                                          std::move(device_cols),
                                          vertex_count,
                                          edge_count);
}

/// Transfers a device_csr_topology back to a host topology.
///
/// @tparam IndexType  The index type
/// @param device_topo Device-resident CSR topology
/// @return A host topology with arrays in host memory
template <typename IndexType = std::int32_t>
topology<IndexType> topology_to_host(const device_csr_topology<IndexType>& device_topo) {
    const auto vertex_count = device_topo.get_vertex_count();
    const auto edge_count = device_topo.get_edge_count();

    if (vertex_count == 0) {
        return topology<IndexType>();
    }

    // Transfer device arrays back to host
    auto host_cols = dal::backend::to_host_sync(device_topo.get_cols_array());
    auto host_rows_i32 = dal::backend::to_host_sync(device_topo.get_rows_array());

    // Convert IndexType row offsets back to int64 for the host topology
    const std::int64_t rows_count = vertex_count + 1;
    auto host_rows_i64 = dal::array<std::int64_t>::empty(rows_count);
    auto* dst = host_rows_i64.get_mutable_data();
    const auto* src = host_rows_i32.get_data();
    for (std::int64_t i = 0; i < rows_count; ++i) {
        dst[i] = static_cast<std::int64_t>(src[i]);
    }

    // Build degrees from row offsets
    auto host_degrees = dal::array<IndexType>::empty(vertex_count);
    auto* deg = host_degrees.get_mutable_data();
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        deg[i] = static_cast<IndexType>(dst[i + 1] - dst[i]);
    }

    topology<IndexType> result;
    result._vertex_count = vertex_count;
    result._edge_count = edge_count;
    result._rows = std::move(host_rows_i64);
    result._cols = std::move(host_cols);
    result._degrees = std::move(host_degrees);
    result._rows_ptr = result._rows.get_data();
    result._cols_ptr = result._cols.get_data();
    result._degrees_ptr = result._degrees.get_data();
    return result;
}

} // namespace oneapi::dal::preview::detail

#endif // ONEDAL_DATA_PARALLEL
