#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

load("@onedal//dev/bazel:repos.bzl", "repos")

micromkl_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
        "%{os}/include",
    ],
    libs = [
        "%{os}/lib/libdaal_mkl_thread.a",
        "%{os}/lib/libdaal_vmlipp_core.a",
    ],
    build_template = "@onedal//dev/bazel/deps:micromkl.tpl.BUILD",
    download_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # In this case, files from `lib/*` will be copied to `lib/intel64/*`.
    "lib/": "lib/intel64/",
    },
    local_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # In this case, files from `lib/*` will be copied to `lib/intel64/*`.
    "lib/": "lib/intel64/",
    },
)

micromkl_dpc_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/libdaal_sycl.a",
    ],
    build_template = "@onedal//dev/bazel/deps:micromkldpc.tpl.BUILD",
    download_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # In this case, files from `lib/*` will be copied to `lib/intel64/*`.
    "lib/": "lib/intel64/",
    },
    local_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # In this case, files from `lib/*` will be copied to `lib/intel64/*`.
    "lib/": "lib/intel64/",
    },
)
