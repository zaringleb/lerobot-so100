vllm serve vla-0-smol \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"full_cuda_graph": true}'