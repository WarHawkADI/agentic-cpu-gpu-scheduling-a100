#!/usr/bin/env bash
# Step 2: install vLLM and download the paper's scheduler LLM,
# DeepSeek-R1-Distill-Qwen-32B, 4-bit AWQ build (about 18 GB).
#
#   bash scripts/setup_llm.sh
#
# Environment (all optional):
#   VLLM_VENV     Python venv for vLLM                 (default ~/venvs/vllm)
#   VLLM_VERSION  vLLM version to install, or "latest" (default 0.11.0)
#   MODEL_REPO    Hugging Face repo of the model       (default casperhansen/deepseek-r1-distill-qwen-32b-awq)
#   MODEL_DIR     where the model is stored            (default ~/models/deepseek-r1-distill-qwen-32b-awq)
#   HF_TOKEN      only needed if Hugging Face rate-limits anonymous downloads
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
MODEL_REPO="${MODEL_REPO:-casperhansen/deepseek-r1-distill-qwen-32b-awq}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/deepseek-r1-distill-qwen-32b-awq}"
step() { echo; echo "=== $*"; }

step "1/3 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
  echo "nvidia-smi not found: install the NVIDIA driver first" >&2
  exit 1
fi

step "2/3 vLLM $VLLM_VERSION in $VLLM_VENV"
[ -f "$VLLM_VENV/bin/activate" ] || python3 -m venv "$VLLM_VENV"
set +u; . "$VLLM_VENV/bin/activate"; set -u
python3 -m pip install -q -U pip
if [ "$VLLM_VERSION" = latest ]; then
  python3 -m pip install -U vllm
else
  python3 -m pip install "vllm==$VLLM_VERSION"
fi
python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, \
'| CUDA available:', torch.cuda.is_available())"

step "3/3 Model $MODEL_REPO -> $MODEL_DIR"
mkdir -p "$MODEL_DIR"
MODEL_REPO="$MODEL_REPO" MODEL_DIR="$MODEL_DIR" python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ["MODEL_REPO"], local_dir=os.environ["MODEL_DIR"])
print("model files in", path)
EOF
du -sh "$MODEL_DIR"

echo
echo "LLM ready. Next, in a tmux window: bash scripts/start_llm_server.sh   (see README, section 4)"
