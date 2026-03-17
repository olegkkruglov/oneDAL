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

#include "oneapi/dal/algo/triangle_counting/detail/select_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/gpu/triangle_counting.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Descriptor, typename Topology>
vertex_ranking_result<typename Descriptor::task_t>
backend_default<dal::detail::data_parallel_policy, Descriptor, Topology>::operator()(
    const dal::detail::data_parallel_policy& ctx,
    const Descriptor& descriptor,
    const Topology& t) {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

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

// Explicit instantiations for supported descriptor/topology combinations
using default_descriptor_local = descriptor<
    float,
    method::ordered_count,
    task::local,
    std::allocator<char>>;

using default_descriptor_global = descriptor<
    float,
    method::ordered_count,
    task::global,
    std::allocator<char>>;

using default_descriptor_local_and_global = descriptor<
    float,
    method::ordered_count,
    task::local_and_global,
    std::allocator<char>>;

using default_topology = dal::preview::detail::topology<std::int32_t>;

template struct ONEDAL_EXPORT backend_default<dal::detail::data_parallel_policy,
                                              default_descriptor_local,
                                              default_topology>;
template struct ONEDAL_EXPORT backend_default<dal::detail::data_parallel_policy,
                                              default_descriptor_global,
                                              default_topology>;
template struct ONEDAL_EXPORT backend_default<dal::detail::data_parallel_policy,
                                              default_descriptor_local_and_global,
                                              default_topology>;

} // namespace oneapi::dal::preview::triangle_counting::detail
