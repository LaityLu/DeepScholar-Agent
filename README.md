CUDA_VISIBLE_DEVICES=0  \
VLLM_USE_FLASHINFER_SAMPLER=0  \
uv run vllm serve ./model/Qwen3.5-4B  \
--served-model-name Qwen3.5-4B  \
--max-model-len 4096  \
--gpu-memory-utilization 0.30  \
--enforce-eager  \
--reasoning-parser qwen3  \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--host 0.0.0.0  \
--port 8000  \
--api-key 123