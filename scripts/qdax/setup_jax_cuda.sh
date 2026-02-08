#!/bin/bash
# Helper to set up LD_LIBRARY_PATH for JAX CUDA support inside the IsaacLab container.
# Source this before running scripts that use both IsaacLab (torch) and QDax (jax).
#
# Usage:  source scripts/qdax/setup_jax_cuda.sh

NVIDIA_LIBS="${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia"
NEW_CUDNN="/tmp/cudnn_new/nvidia/cudnn/lib"

export LD_LIBRARY_PATH="${NEW_CUDNN}:${NVIDIA_LIBS}/cuda_runtime/lib:${NVIDIA_LIBS}/cublas/lib:${NVIDIA_LIBS}/cufft/lib:${NVIDIA_LIBS}/cusolver/lib:${NVIDIA_LIBS}/cusparse/lib:${NVIDIA_LIBS}/cuda_nvrtc/lib:${NVIDIA_LIBS}/cuda_cupti/lib:${NVIDIA_LIBS}/nvjitlink/lib:${LD_LIBRARY_PATH}"

echo "[INFO] LD_LIBRARY_PATH updated for JAX CUDA support."
