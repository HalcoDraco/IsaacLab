# Enter container
docker exec -it isaac-lab-base bash

# Install packages
PYTHON=/workspace/isaaclab/_isaac_sim/kit/python/bin/python3
$PYTHON -m pip install --no-cache-dir \
  "jax[cuda12]==0.9.0.1" "jaxlib==0.9.0.1" "jax-cuda12-plugin==0.9.0.1" \
  "flax==0.10.3" "optax==0.2.4" "chex==0.1.88" "coverage"

# Copy QDax source into container (from host)
# docker cp /path/to/QDax isaac-lab-base:/workspace/QDax
$PYTHON -m pip install --no-deps -e /workspace/QDax

# Run training
cd /workspace/isaaclab
source scripts/qdax/setup_jax_cuda.sh
./isaaclab.sh -p scripts/qdax/train_me_based.py --algorithm pgame --task Isaac-Ant-Direct-v0 --num_envs 32 --headless