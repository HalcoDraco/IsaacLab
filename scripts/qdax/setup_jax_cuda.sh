#!/bin/bash
# Helper to set up LD_LIBRARY_PATH for JAX CUDA support inside the IsaacLab container.
# Source this before running scripts that use both IsaacLab (torch) and QDax (jax).
#
# Usage:  source scripts/qdax/setup_jax_cuda.sh

# NVIDIA CUDA libs installed by pip (jax[cuda12]) into site-packages
SITE_NV="${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/kit/python/lib/python3.11/site-packages/nvidia"

export LD_LIBRARY_PATH="\
${SITE_NV}/cudnn/lib:\
${SITE_NV}/cuda_runtime/lib:\
${SITE_NV}/cublas/lib:\
${SITE_NV}/cufft/lib:\
${SITE_NV}/cusolver/lib:\
${SITE_NV}/cusparse/lib:\
${SITE_NV}/cuda_nvrtc/lib:\
${SITE_NV}/cuda_cupti/lib:\
${SITE_NV}/nvjitlink/lib:\
${LD_LIBRARY_PATH}"

echo "[INFO] LD_LIBRARY_PATH updated for JAX CUDA support."
