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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/algo/triangle_counting/backend/gpu/triangle_counting.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#endif

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Policy, typename Descriptor, typename Topology>
struct backend_base {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(const Policy& ctx,
                                                     const Descriptor& descriptor,
                                                     const Topology& t) = 0;
    virtual ~backend_base() = default;
};

template <typename Policy, typename Descriptor, typename Topology>
struct backend_default : public backend_base<Policy, Descriptor, Topology> {
    static_assert(dal::detail::is_one_of_v<Policy, dal::detail::host_policy>,
                  "Host policy only is supported.");

    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(const Policy& ctx,
                                                     const Descriptor& descriptor,
                                                     const Topology& t) {
        return vertex_ranking_kernel_cpu<method_t, task_t, allocator_t, Topology>()(
            ctx,
            descriptor,
            descriptor.get_allocator(),
            t);
    }
};

#ifdef ONEDAL_DATA_PARALLEL
template <typename Descriptor, typename Topology>
struct backend_default<dal::detail::data_parallel_policy, Descriptor, Topology>
        : public backend_base<dal::detail::data_parallel_policy, Descriptor, Topology> {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    virtual vertex_ranking_result<task_t> operator()(
        const dal::detail::data_parallel_policy& ctx,
        const Descriptor& descriptor,
        const Topology& t) {
        return dal::backend::dispatch_by_device(
            ctx,
            [&]() {
                // CPU path: delegate to host policy kernel
                return vertex_ranking_kernel_cpu<method_t, task_t, allocator_t, Topology>()(
                    dal::detail::host_policy::get_default(),
                    descriptor,
                    descriptor.get_allocator(),
                    t);
            },
            [&]() {
                // GPU path: delegate to GPU kernel
                dal::backend::context_gpu gpu_ctx{ ctx };
                return backend::vertex_ranking_kernel_gpu<float_t, task_t, Topology>()(
                    gpu_ctx,
                    descriptor,
                    t);
            });
    }
};
#endif

template <typename Policy, typename Descriptor, typename Topology>
dal::detail::shared<backend_base<Policy, Descriptor, Topology>> get_backend(const Descriptor& desc,
                                                                            const Topology& t) {
    return std::make_shared<backend_default<Policy, Descriptor, Topology>>();
}

} // namespace oneapi::dal::preview::triangle_counting::detail
