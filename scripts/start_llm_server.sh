#!/usr/bin/env bash
# Step 3: serve the scheduler LLM with vLLM's OpenAI-compatible server.
# Keep it running (e.g. in tmux) while the experiment runs; stop it with Ctrl-C.
#
#   bash scripts/start_llm_server.sh
#
# It is ready when it prints "Application startup complete"; then
#   curl http://127.0.0.1:8000/v1/models   lists the model.
#
# Environment (all optional):
#   VLLM_VENV, MODEL_DIR      as in setup_llm.sh
#   SERVED_NAME               model name the API reports  (default deepseek-r1-distill-qwen-32b-awq)
#   CUDA_VISIBLE_DEVICES      which GPU to use            (default 0)
#   HOST, PORT                where to listen             (default 127.0.0.1:8000)
#   MAX_MODEL_LEN             context length              (default 16384)
#   GPU_MEM_UTIL              share of GPU memory vLLM may take (default 0.90)
# Extra arguments are passed to "vllm serve".
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/deepseek-r1-distill-qwen-32b-awq}"
SERVED_NAME="${SERVED_NAME:-deepseek-r1-distill-qwen-32b-awq}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[ -f "$VLLM_VENV/bin/activate" ] || { echo "no vLLM venv at $VLLM_VENV: run scripts/setup_llm.sh" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "no model at $MODEL_DIR: run scripts/setup_llm.sh" >&2; exit 1; }
set +u; . "$VLLM_VENV/bin/activate"; set -u

# The prompt is under ~2k tokens and the driver asks for at most 8192 answer
# tokens, so 16384 leaves ample room. vLLM detects the AWQ quantization itself.
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --seed 0 \
  "$@"
