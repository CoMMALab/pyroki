#!/usr/bin/env bash
# Build all CUDA kernels.
#
# Usage (from repo root):
#   bash build_kernels/build_all.sh
#   bash build_kernels/build_all.sh --debug
#
# Override GPU arch for all kernels:
#   GPU_ARCH=-arch=sm_80 bash build_kernels/build_all.sh

set -euo pipefail

# Parse + validate ONCE here, then forward the RESOLVED values to every kernel so
# the whole .so set is guaranteed to share one (MAX_JOINTS, MAX_ACT) pair. Mixing
# capacities across libraries is the failure this forwarding exists to prevent:
# nothing at runtime would catch, say, an ls_ik built at MAX_ACT=24 loaded next to
# an sqp_ik built at 16.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_build_params.sh"
parse_build_params "$@"
BUILD_ARGS=("${BUILD_PARAM_ARGS[@]}")

echo "Building all kernels with MAX_JOINTS=${PYROFFI_MAX_JOINTS} MAX_ACT=${PYROFFI_MAX_ACT}"

bash "${SCRIPT_DIR}/build_fk_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_collision_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_collision_binary_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_fused_self_collision_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_hjcd_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_ls_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_analytic_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_canonical_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_ik_jacobian_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_brownian_motion_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_hit_and_run_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_svgd_region_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_mppi_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_sqp_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_sco_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_ls_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_chomp_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_stomp_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_robogpu_collision.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_cricket_jit.sh"

echo "All CUDA kernels built successfully."
