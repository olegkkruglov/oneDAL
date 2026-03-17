load("@onedal//dev/bazel:dal.bzl", "dal_module")

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "triangle_counting_backend_cpu",
    auto_deps = False,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.hpp"]),
    deps = [
        "@onedal//cpp/oneapi/dal/algo/triangle_counting/backend:triangle_counting_backend_common",
    ],
)
