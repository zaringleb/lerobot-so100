lerobot-eval \
  --policy.path="outputs/train/2025-12-09/00-07-30_pusht_vla0/checkpoints/030000/pretrained_model" \
  --env.type=pusht \
  --policy.ensemble_size=0 \
  --policy.n_action_steps=5 \
  --policy.chunk_size=5 \
  --env.task=PushT-v0 \
  --eval.batch_size=1 \
  --eval.n_episodes=1

#   --policy.use_vllm_client=true \
