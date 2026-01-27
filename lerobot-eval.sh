export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

lerobot-eval \
  --policy.path="Robot-Learning-Collective/VLA-0-Smol" \
  --policy.n_action_steps=1 \
  --policy.chunk_size=1 \
  --policy.ensemble_size=0 \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=25
