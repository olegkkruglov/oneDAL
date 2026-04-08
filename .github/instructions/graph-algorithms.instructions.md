---
applyTo: ["**/algo/triangle_counting/**", "**/algo/connected_components/**", "**/algo/shortest_paths/**", "**/algo/subgraph_isomorphism/**", "**/algo/jaccard/**", "**/algo/louvain/**", "**/graph/**", "**/vertex_ranking*", "**/vertex_partitioning*", "**/graph_matching*", "**/traverse*"]
---

# Graph Algorithms Instructions for GitHub Copilot

## Architecture Overview

Graph algorithms in oneDAL live under the `oneapi::dal::preview` namespace (not `oneapi::dal`).
They operate on graph data structures (CSR topology), not tabular data.

### Graph Algorithms

| Algorithm | Operation | Graph Type | Header |
|---|---|---|---|
| Triangle Counting | `vertex_ranking()` | Undirected | `triangle_counting.hpp` |
| Connected Components | `vertex_partitioning()` | Undirected | `connected_components.hpp` |
| Shortest Paths | `traverse()` | Directed (weighted) | `shortest_paths.hpp` |
| Subgraph Isomorphism | `graph_matching()` | Undirected | `subgraph_isomorphism.hpp` |
| Jaccard Similarity | `vertex_similarity()` | Undirected | `jaccard.hpp` |
| Louvain | `vertex_partitioning()` | Undirected (weighted) | `louvain.hpp` |

### Graph Types

- **`undirected_adjacency_vector_graph<>`** — CSR undirected graph (most algorithms)
- **`directed_adjacency_vector_graph<>`** — CSR directed graph (shortest paths)

Both graph types enforce `IndexType = std::int32_t` via `static_assert`. This is
a deliberate design constraint — see `cpp/oneapi/dal/graph/detail/common.hpp`:
```cpp
constexpr bool is_valid_index_v = dal::detail::is_one_of_v<IndexType, std::int32_t>;
```

## CSR Topology Data Structure

The internal graph representation is `topology<std::int32_t>` defined in
`cpp/oneapi/dal/graph/detail/csr_topology.hpp`:

| Field | Type | Size | Purpose |
|---|---|---|---|
| `_rows` | `int64[]` | vertex_count + 1 | CSR row offsets (cumulative edge counts) |
| `_cols` | `int32[]` | 2 × edge_count (undirected) | Column indices (neighbor vertex IDs) |
| `_degrees` | `int32[]` | vertex_count | Per-vertex degree |

**Why mixed types**: Row offsets are int64 because total edges can exceed 2^31.
Column indices and degrees are int32 because vertex count is bounded by int32.

### Host ↔ Device Transfer

`device_csr_topology.hpp` provides GPU transfer utilities:
- `topology_to_device(queue, host_topology)` → `device_csr_topology` (USM device memory)
- `topology_to_host(device_topology)` → `topology` (host memory)

**Critical**: Rows remain int64 on device — no narrowing conversion. Column indices
stay int32 (IndexType). Device pointers cannot be dereferenced on host.

## GPU Dispatch Pattern

### File Structure for GPU-Enabled Algorithm

```
algo/{algorithm}/
├── common.hpp                          # Descriptor, method/task tags, enums
├── {operation}_types.hpp               # Input/Result types (public)
├── {operation}.hpp                     # Public operation function
├── detail/
│   ├── select_kernel.hpp               # Backend dispatch (host + data_parallel_policy)
│   ├── select_kernel_dpc.cpp           # GPU dispatch implementation
│   ├── {operation}_ops.hpp             # Operation dispatch helpers
│   └── {operation}_types.hpp           # Internal type details
├── backend/
│   ├── cpu/
│   │   └── {operation}_default_kernel.hpp  # CPU kernel
│   └── gpu/
│       ├── {algorithm}.hpp                 # Forward declarations
│       ├── {operation}_kernel.hpp          # GPU view struct + kernel declaration
│       └── {operation}_default_kernel_gpu_dpc.cpp  # SYCL GPU kernel
├── test/
│   ├── {algorithm}_test.cpp            # Host (CPU) tests
│   └── {algorithm}_gpu_test.cpp        # GPU tests (DPC++)
└── BUILD
```

### select_kernel.hpp — Backend Dispatch

Two specializations: host_policy (CPU) and data_parallel_policy (GPU):

```cpp
// CPU specialization (inline, calls kernel directly)
template <typename Descriptor, typename Topology>
struct backend_default<dal::detail::host_policy, Descriptor, Topology> {
    vertex_ranking_result<task_t> operator()(const host_policy& ctx,
                                             const Descriptor& desc,
                                             const Topology& t) {
        return vertex_ranking_kernel_cpu<...>()(ctx, desc, desc.get_allocator(), t);
    }
};

// GPU specialization (declaration only — resolved at link time via DPC++ library)
#ifdef ONEDAL_DATA_PARALLEL
template <typename Descriptor, typename Topology>
struct backend_default<dal::detail::data_parallel_policy, Descriptor, Topology> {
    vertex_ranking_result<task_t> operator()(const data_parallel_policy& ctx,
                                             const Descriptor& desc,
                                             const Topology& t);
};
#endif
```

### select_kernel_dpc.cpp — GPU Dispatch Implementation

Uses `dispatch_by_device` to route CPU vs GPU execution:

```cpp
return dal::backend::dispatch_by_device(
    ctx,
    [&]() {
        // CPU fallback: delegate to host kernel
        return vertex_ranking_kernel_cpu<...>()(host_policy::get_default(), ...);
    },
    [&]() {
        // GPU path: transfer topology to device, run GPU kernel
        dal::backend::context_gpu gpu_ctx{ ctx };
        auto device_topo = topology_to_device<std::int32_t>(gpu_ctx.get_queue(), t);
        auto gpu_view = make_gpu_view(device_topo);
        return run_vertex_ranking_gpu(gpu_ctx, desc, gpu_view);
    });
```

**Must include explicit template instantiations** at the bottom for each
descriptor/topology combination.

### GPU View Struct

Lightweight pointer wrapper passed to SYCL kernels (no `dal::array` overhead):

```cpp
template <typename Index = std::int32_t>
struct csr_topology_gpu_view {
    const std::int64_t* rows = nullptr;  // Device pointer (int64)
    const Index* cols = nullptr;          // Device pointer (int32)
    std::int64_t vertex_count = 0;
    std::int64_t edge_count = 0;
};
```

**Critical**: Never dereference `rows` or `cols` on the host — they are device pointers.
If you need a value on host (e.g., `cols_count`), store it as a scalar field.

## Build System Patterns

### Bazel BUILD for Graph Algorithm with GPU

```python
dal_module(
    name = "algorithm_name",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
)

# Host tests — exclude GPU test files
dal_test_suite(
    name = "tests",
    framework = "catch2",
    srcs = glob(["test/*.cpp"], exclude = ["test/*_gpu_test.cpp"]),
    dal_deps = [":algorithm_name"],
)

# GPU tests — DPC++ compilation, GPU test files only
dal_test_suite(
    name = "gpu_tests",
    framework = "catch2",
    compile_as = ["dpc++"],
    srcs = glob(["test/*_gpu_test.cpp"]),
    dal_deps = [":algorithm_name"],
)
```

### DPC++ Example in examples/oneapi/dpc/BUILD

Add algorithm name to the `dal_algo_example_suite` `algos` list:
```python
dal_algo_example_suite(
    algos = [
        ...
        "triangle_counting",
    ],
    compile_as = ["dpc++"],
    ...
)
```

### Build Commands

```bash
# Bazel: host tests
bazel test //cpp/oneapi/dal/algo/{algorithm}:tests --test_output=all

# Bazel: GPU tests
bazel test //cpp/oneapi/dal/algo/{algorithm}:gpu_tests --test_output=all

# Make: build DPC++ library (needed for CMake examples)
make onedal_dpc -j$(nproc)

# CMake: build and run examples
cd __release_lnx/daal/latest/examples/oneapi/dpc/
cmake -B build -DCMAKE_PREFIX_PATH=$DALROOT
cmake --build build -j$(nproc)
```

## Testing Patterns

### GPU Test Structure (Catch2)

```cpp
#include <sycl/sycl.hpp>
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif
#include "oneapi/dal/algo/triangle_counting.hpp"
#include "catch2/catch.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::triangle_counting;

// Build graph inline (no CSV dependency)
auto create_test_graph() {
    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    auto builder = dal::preview::detail::undirected_adjacency_vector_graph_builder<>(4, 6);
    builder.set_edge(0, 1); builder.set_edge(0, 2); builder.set_edge(1, 2); ...
    return builder.build();
}

TEST_CASE("triangle counting on GPU") {
    sycl::queue q{ sycl::gpu_selector_v };
    const auto graph = create_test_graph();
    const auto desc = descriptor<float, method::ordered_count, task::global>();
    const auto result = dal::preview::vertex_ranking(q, desc, graph);
    REQUIRE(result.get_global_rank() == expected_value);
}
```

### Key Testing Rules
- Build graphs inline using graph builders — no external data file dependencies
- Test both result correctness and edge cases (empty graph, single vertex, disconnected)
- GPU tests go in `*_gpu_test.cpp`, host tests in `*_test.cpp`
- Use `sycl::gpu_selector_v` for GPU queue

## Example Patterns

### CPU Example (`examples/oneapi/cpp/source/{algorithm}/`)

```cpp
#include "oneapi/dal/algo/triangle_counting.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"

const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename });
const auto desc = descriptor<float, method::ordered_count, task::local_and_global>();
const auto result = dal::preview::vertex_ranking(desc, graph);
```

### DPC++ Example (`examples/oneapi/dpc/source/{algorithm}/`)

```cpp
#include <sycl/sycl.hpp>
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif
#include "oneapi/dal/algo/triangle_counting.hpp"

void run(sycl::queue& q) {
    const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename });
    const auto desc = descriptor<float, method::ordered_count, task::local_and_global>();
    // Pass sycl::queue as first argument for GPU execution
    const auto result = dal::preview::vertex_ranking(q, desc, graph);
}

int main() {
    for (auto d : list_devices()) {
        sycl::queue q{ d };
        run(q);
    }
}
```

### Key Differences CPU vs DPC++
- DPC++ requires `#define ONEDAL_DATA_PARALLEL` before oneDAL headers
- DPC++ passes `sycl::queue` as the first argument to the operation
- DPC++ examples iterate over available devices (`list_devices()`)

## PR Review Checklist for Graph Algorithms

### New GPU Interface
- [ ] `select_kernel.hpp` has `data_parallel_policy` specialization
- [ ] `select_kernel_dpc.cpp` uses `dispatch_by_device` with CPU fallback + GPU path
- [ ] GPU view struct uses `int64*` for rows, `int32*` for cols (no narrowing)
- [ ] Device pointers never dereferenced on host
- [ ] Explicit template instantiations cover all descriptor/topology combos
- [ ] BUILD file has `gpu_tests` target with `compile_as = ["dpc++"]`
- [ ] Host tests exclude `*_gpu_test.cpp` files
- [ ] GPU tests build graphs inline (no data file dependencies)
- [ ] DPC++ example added to `dal_algo_example_suite` in `examples/oneapi/dpc/BUILD`

### Operation Plumbing (for new algorithms)
- [ ] Operation dispatch header in `cpp/oneapi/dal/detail/` (e.g., `vertex_ranking_ops.hpp`)
- [ ] `sycl::queue` overload added to operation dispatch
- [ ] Public header (e.g., `vertex_ranking.hpp`) has `sycl::queue` overload

## Cross-Reference
- **[general.instructions.md](general.instructions.md)** - Repository overview
- **[cpp-coding-guidelines.instructions.md](cpp-coding-guidelines.instructions.md)** - C++ standards
- **[build-systems.instructions.md](build-systems.instructions.md)** - Build system details
- **[examples.instructions.md](examples.instructions.md)** - Example code patterns
