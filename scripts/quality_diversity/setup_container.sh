#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Quality-Diversity container setup for IsaacLab
# ─────────────────────────────────────────────────────────────────────────────
# Run this inside the IsaacLab Docker container to install the dependencies
# required by the QD training scripts.
#
#   docker exec isaac-lab-base bash /workspace/isaaclab/scripts/quality_diversity/setup_container.sh
#
# Prerequisites:
#   - The container must be running and named "isaac-lab-base".
#   - Isaac Sim's Python is at /workspace/isaaclab/_isaac_sim/python.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PIP="/workspace/isaaclab/_isaac_sim/python.sh -m pip"

echo "=== [1/3] Upgrading coverage (fixes numba/coverage incompatibility) ==="
# numba (pulled in by pyribs → numpy_groupies) breaks when coverage < 7.6
# due to a missing 'Tracer' attribute in coverage.types.
$PIP install --quiet "coverage>=7.6"

echo "=== [2/3] Installing pyribs (QD library) ==="
# The PyPI package name is "ribs", not "pyribs".
$PIP install --quiet ribs

echo "=== [3/3] Verifying imports ==="
/workspace/isaaclab/_isaac_sim/python.sh -c "
import ribs, stable_baselines3 as sb3, torch, gymnasium
print(f'  pyribs          {ribs.__version__}')
print(f'  stable-baselines3 {sb3.__version__}')
print(f'  torch            {torch.__version__}')
print(f'  gymnasium        {gymnasium.__version__}')
print(f'  CUDA available   {torch.cuda.is_available()}')
print('All imports OK.')
"

echo ""
echo "=== Setup complete ==="
echo "Run QD training with:"
echo "  isaaclab -p scripts/quality_diversity/train.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --headless --algo map_elites --generations 50"
