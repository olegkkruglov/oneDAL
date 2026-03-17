#pragma once

#include <cstdint>

#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template <typename Index = std::int32_t>
struct csr_topology_gpu_view {
    const Index* rows = nullptr;
    const Index* cols = nullptr;
    std::int64_t vertex_count = 0;
    std::int64_t edge_count = 0;
};

template <typename Float, typename Task, typename Index = std::int32_t>
vertex_ranking_result<Task> run_vertex_ranking_gpu(const dal::backend::context_gpu& ctx,
                                                   const detail::descriptor_base<Task>& desc,
                                                   const csr_topology_gpu_view<Index>& topology);

} // namespace oneapi::dal::preview::triangle_counting::backend
