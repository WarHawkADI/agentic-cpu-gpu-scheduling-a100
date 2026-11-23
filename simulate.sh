#!/usr/bin/env bash
# Simulate ONE scenario + device mapping on ASTRA-Sim and show what happens:
# the trace, ASTRA-Sim's own output, the per-tool timeline and the latency.
#
#   bash simulate.sh                       S13 with the paper's oracle mapping
#   bash simulate.sh S5 gpu,gpu,cpu,gpu    any scenario + mapping (tools A,B,C,D)
#   bash simulate.sh S13 GN,GN,GN,CPU      an out-of-memory mapping (refused)
#
# Mappings: S1-S10 use gpu/cpu; S11-S13 use GN (gpu_now) / GQ (gpu_queue) / CPU.
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${SIM:-$HOME/LLMServingSim}"
SIM_VENV="${SIM_VENV:-$HOME/venvs/sim}"
BIN="$SIM/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
CHAKRA="$SIM/astra-sim/extern/graph_frontend/chakra"
DEMO="${DEMO:-$HOME/tooldag_demo}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$SIM_VENV/bin/activate" ]; then
  set +u; . "$SIM_VENV/bin/activate"; set -u
fi
[ -x "$BIN" ] || { echo "ASTRA-Sim is not built at $SIM: run  bash scripts/setup_simulator.sh" >&2; exit 1; }
if ! cmp -s "$PROJ/llm_converter_cpu_gpu.py" "$CHAKRA/src/converter/llm_converter_cpu_gpu.py"; then
  echo "Installing the updated converter into Chakra ..."
  cp "$PROJ/llm_converter_cpu_gpu.py" "$CHAKRA/src/converter/" && python3 -m pip install -q "$CHAKRA"
fi

SCEN="${1:-S13}"
MAP="${2:-GN,GQ,GN,CPU}"
mkdir -p "$DEMO" && cd "$DEMO" || exit 1
cp "$PROJ"/astra_configs/* .
rm -rf log
name="$(echo "${SCEN}_${MAP}" | tr ',' '_')"
rm -f "$name".* astra_output.txt

echo "=================== 1. Workload: $SCEN, mapping A,B,C,D = $MAP"
python3 "$PROJ/agentic_sched_experiment.py" export --scenario "$SCEN" --mapping "$MAP" \
  --trace "$name.txt" --convert "$name" 2>&1 | grep -v '^wrote '
echo
[ -f "$name.txt" ] && cat "$name.txt"
if [ ! -f "$name.0.et" ]; then
  echo
  echo "No ET written: the mapping was refused (see the message above), so there is nothing to simulate."
  exit 1
fi

echo
echo "=================== 2. ASTRA-Sim output"
echo "Running ASTRA-Sim: $(readlink -f "$BIN")"
echo "  (LLMServingSim's ASTRA-Sim, analytical network backend; 1 cycle = 1 ns)"
echo
echo exit | "$BIN" --workload-configuration="$name" --system-configuration=system.json \
  --network-configuration=network.yml --memory-configuration=memory.json \
  --start-npu-ids=0 --end-npu-ids=0 | tee astra_output.txt
sleep 0.2  # the debug log is written asynchronously

echo
echo "=================== 3. Timeline (each tool node: start / finish)"
grep -E "issue,|callback," log/log.log \
  | sed -E 's/.*(issue|callback),sys->id=[0-9]+, tick=([0-9]+), node->id=[0-9]+, node->name=COMP_NODE_TOOL_([^,]+),.*/\1 \2 \3/' \
  | awk '{ printf "  %-7s %9.1f ms   %s\n", ($1 == "issue" ? "start" : "finish"), $2 / 1e6, $3 }'

cycles="$(grep -oE 'finished, [0-9]+ cycles' astra_output.txt | grep -oE '[0-9]+' | head -1)"
echo
echo "=================== 4. Result"
if [ -n "$cycles" ]; then
  awk -v c="$cycles" -v s="$SCEN" -v m="$MAP" \
    'BEGIN { printf "  %s with %s: end-to-end latency = %d ns = %.1f ms\n", s, m, c, c / 1e6 }'
else
  echo "  ASTRA-Sim did not report a finish time (see its output above)."
fi
echo "  Files kept in $DEMO: $name.txt (trace), $name.0.et (workload), astra_output.txt, log/log.log"
