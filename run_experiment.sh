#!/usr/bin/env bash
# Run one command of the experiment driver on LLMServingSim's ASTRA-Sim.
#
#   bash run_experiment.sh <command> [driver options]
#   e.g.  bash run_experiment.sh selftest
#         bash run_experiment.sh crosscheck
#         bash run_experiment.sh llmcheck --llm-base-url http://127.0.0.1:8000/v1
#         bash run_experiment.sh all --llm-base-url http://127.0.0.1:8000/v1 --with-ablation --out results_real_llm
#
# Environment: SIM (LLMServingSim checkout, default ~/LLMServingSim),
#              SIM_VENV (Python venv made by scripts/setup_simulator.sh, default ~/venvs/sim).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${SIM:-$HOME/LLMServingSim}"
SIM_VENV="${SIM_VENV:-$HOME/venvs/sim}"
BIN="$SIM/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
CHAKRA="$SIM/astra-sim/extern/graph_frontend/chakra"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$SIM_VENV/bin/activate" ]; then
  set +u; . "$SIM_VENV/bin/activate"; set -u
fi
[ -x "$BIN" ] || { echo "ASTRA-Sim is not built at $SIM: run  bash scripts/setup_simulator.sh" >&2; exit 1; }

# The converter is imported from the installed chakra package, so reinstall it
# whenever the project copy of llm_converter_cpu_gpu.py changes.
if ! cmp -s "$PROJ/llm_converter_cpu_gpu.py" "$CHAKRA/src/converter/llm_converter_cpu_gpu.py"; then
  echo "Installing the updated converter into Chakra ..." >&2
  cp "$PROJ/llm_converter_cpu_gpu.py" "$CHAKRA/src/converter/"
  python3 -m pip install -q "$CHAKRA"
fi

# ASTRA-Sim's configs and its log/log.log live in a private work folder: the
# ASTRA-Sim command line is split on spaces (so no spaces in these paths), and
# runs started side by side must not share log/log.log.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/tooldag.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cp "$PROJ"/astra_configs/* "$WORK/"
export ASTRA_SIM_CMD="$BIN --workload-configuration={workload} \
--system-configuration=$WORK/system.json --network-configuration=$WORK/network.yml \
--memory-configuration=$WORK/memory.json --start-npu-ids=0 --end-npu-ids=0"

cmd="${1:-all}"
shift || true
python3 "$PROJ/agentic_sched_experiment.py" "$cmd" --backend astra --num-npus 1 \
  --astra-cwd "$WORK" --astra-log-file "$WORK/log/log.log" "$@"
