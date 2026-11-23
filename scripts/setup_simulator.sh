#!/usr/bin/env bash
# Step 1: build the simulator (LLMServingSim's ASTRA-Sim) and install this
# project's Chakra converter into it. Needs no GPU. Takes about 5-15 minutes.
#
#   bash scripts/setup_simulator.sh
#
# Environment (all optional):
#   SIM         where LLMServingSim is cloned       (default ~/LLMServingSim)
#   SIM_VENV    Python venv for the simulator side  (default ~/venvs/sim)
#   SIM_COMMIT  LLMServingSim commit to build       (default: the one this project was tested with)
#   SKIP_APT=1  do not install system packages (install them yourself, see README)
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM="${SIM:-$HOME/LLMServingSim}"
SIM_VENV="${SIM_VENV:-$HOME/venvs/sim}"
SIM_COMMIT="${SIM_COMMIT:-560ce8210f8a1ab0be67a7ae66cce085ea990051}"
SIM_REPO="https://github.com/casys-kaist/LLMServingSim.git"
step() { echo; echo "=== $*"; }

step "1/5 System packages"
if [ "${SKIP_APT:-0}" != 1 ] && command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git python3-venv python3-dev python3-pip \
    libprotobuf-dev protobuf-compiler tmux
else
  echo "skipped (SKIP_APT=1 or no apt-get); checking the tools are there"
fi
for tool in git cmake make g++ protoc python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing: $tool (see README, section 2)" >&2; exit 1; }
done

step "2/5 Python venv $SIM_VENV"
[ -f "$SIM_VENV/bin/activate" ] || python3 -m venv "$SIM_VENV"
set +u; . "$SIM_VENV/bin/activate"; set -u
# Old setuptools (e.g. Ubuntu 22.04's 59) cannot build Chakra.
python3 -m pip install -q -U pip setuptools wheel

step "3/5 LLMServingSim at commit ${SIM_COMMIT:0:7} -> $SIM"
if [ ! -d "$SIM/.git" ]; then
  git clone "$SIM_REPO" "$SIM"
fi
git -C "$SIM" fetch -q origin || echo "(fetch failed; using the commits already cloned)"
git -C "$SIM" checkout -q "$SIM_COMMIT"
git -C "$SIM" submodule update --init --recursive
echo "LLMServingSim: $(git -C "$SIM" rev-parse HEAD)"
echo "ASTRA-Sim:     $(git -C "$SIM/astra-sim" rev-parse HEAD)"

step "4/5 Install the converter, then build (LLMServingSim's scripts/compile.sh)"
cp "$PROJ/llm_converter_cpu_gpu.py" "$SIM/astra-sim/extern/graph_frontend/chakra/src/converter/"
(cd "$SIM" && bash ./scripts/compile.sh)

step "5/5 Check"
BIN="$SIM/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
[ -x "$BIN" ] || { echo "ASTRA-Sim binary not found: $BIN" >&2; exit 1; }
echo "ASTRA-Sim binary: $(readlink -f "$BIN")"
python3 -c "import chakra.src.converter.llm_converter_cpu_gpu as m; print('converter:', m.__file__)"
python3 "$PROJ/agentic_sched_experiment.py" selftest | tail -1
bash "$PROJ/simulate.sh" S13 GN,GQ,GN,CPU | tail -3

echo
echo "Simulator ready. Next: bash scripts/setup_llm.sh   (see README, section 3)"
