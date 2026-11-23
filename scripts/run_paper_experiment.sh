#!/usr/bin/env bash
# Step 4: the whole experiment of the paper, with the real scheduler LLM.
# Start the LLM server first (scripts/start_llm_server.sh), then:
#
#   bash scripts/run_paper_experiment.sh
#
# What it does, all logged into $OUT:
#   1. waits until the LLM server answers
#   2. records the environment                       -> environment.txt
#   3. crosscheck: ASTRA-Sim vs the reference model  -> crosscheck.txt  (until it has passed once)
#   4. llmcheck: one real scheduling call            -> llmcheck.txt
#   5. oracle + baselines + agentic scheduler (cold and warm start, 20 rounds, S1-S13)
#      + the S13 monitor ablation                    -> summary.md, rounds.csv, llm_log.jsonl, ...
#   6. (full run only) S13 once more with the tool profiles of the paper's own S13 run
#      (Fig. 7: 627/463/417/58 MB), for a direct comparison with Figs. 7 and 9
#                                                    -> fig7_s13/summary.md, ...
#
# If the run stops (e.g. the LLM server stopped answering), it waits for the
# server and continues by itself, up to AUTO_RESUME times. Running this script
# again later also continues where it stopped. To start from scratch, delete $OUT.
#
# Environment (all optional):
#   LLM_BASE_URL     the server            (default http://127.0.0.1:8000/v1)
#   OUT              results folder        (default results_real_llm in this project)
#   PARALLEL=3       run 3 parts side by side against the same server, then merge them
#                    (about 2-3x faster; per-call LLM times are then measured under load)
#   SCENARIOS        (default S1-S13; PARALLEL=3 needs the default)
#   ROUNDS           (default 20, as in the paper)
#   WAIT_SERVER      seconds to wait for the server (default 1800)
#   AUTO_RESUME      automatic continuations after a failure (default 3)
#   SKIP_CROSSCHECK=1
#   SKIP_FIG7=1      leave out step 6 (it runs only when SCENARIOS is S1-S13 anyway)
# Extra arguments go to the driver's "all" command (see: python3 agentic_sched_experiment.py all -h).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
OUT="${OUT:-$PROJ/results_real_llm}"
PARALLEL="${PARALLEL:-1}"
SCENARIOS="${SCENARIOS:-S1-S13}"
ROUNDS="${ROUNDS:-20}"
AUTO_RESUME="${AUTO_RESUME:-3}"
FIG7=1
if [ "${SKIP_FIG7:-0}" = 1 ] || [ "$SCENARIOS" != S1-S13 ]; then FIG7=0; fi
SIM="${SIM:-$HOME/LLMServingSim}"
VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
RUN="$PROJ/run_experiment.sh"
case "$PARALLEL" in 1|3) ;; *) echo "PARALLEL must be 1 or 3" >&2; exit 1 ;; esac
if [ "$PARALLEL" = 3 ] && [ "$SCENARIOS" != S1-S13 ]; then
  echo "PARALLEL=3 splits S1-S13; leave SCENARIOS unset" >&2; exit 1
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
step() { echo; echo "=== $(date '+%F %T')  $*"; }

server_model() {  # prints the served model's name; fails if the server does not answer
  python3 - "$URL" <<'EOF'
import json, os, sys, urllib.request
req = urllib.request.Request(sys.argv[1].rstrip("/") + "/models",
                             headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(json.load(resp)["data"][0]["id"])
except Exception:
    sys.exit(1)
EOF
}

wait_server() {  # waits up to WAIT_SERVER seconds for the server
  local waited=0
  until MODEL="$(server_model)"; do
    if [ "$waited" -ge "${WAIT_SERVER:-1800}" ]; then
      echo "no answer from $URL/models: is scripts/start_llm_server.sh running?" >&2
      return 1
    fi
    [ "$waited" -eq 0 ] && echo "waiting for the LLM server at $URL ..."
    sleep 15
    waited=$((waited + 15))
  done
}

run_part() {  # run_part <folder> <scenarios> [extra driver args]; continues by itself after a failure
  local dir="$1" scen="$2"; shift 2
  local try=0
  while true; do
    local resume=()
    if [ -f "$dir/rounds.csv" ]; then
      resume=(--resume)
      echo "continuing the run in $dir (--resume)"
    fi
    if bash "$RUN" all --llm-base-url "$URL" --scenarios "$scen" --rounds "$ROUNDS" \
         --out "$dir" "${resume[@]}" "$@"; then
      return 0
    fi
    try=$((try + 1))
    if [ "$try" -gt "$AUTO_RESUME" ]; then
      echo "the run in $dir stopped $try times; giving up (run this script again to continue)" >&2
      return 1
    fi
    echo "the run in $dir stopped; continuing in 60 s once the LLM server answers ($try/$AUTO_RESUME)"
    sleep 60
    wait_server || return 1
  done
}

step "1. LLM server at $URL"
wait_server || exit 1
echo "serving: $MODEL"

step "2. Environment -> $OUT/environment.txt"
{
  echo "## $(date '+%F %T %Z')"
  echo "LLM endpoint: $URL, model: $MODEL"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/GPU: /'
  fi
  "$VLLM_VENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__)" 2>/dev/null \
    || echo "vLLM: not found in $VLLM_VENV (another server?)"
  echo "this project: $(git -C "$PROJ" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
  echo "LLMServingSim: $(git -C "$SIM" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "python: $(python3 --version 2>&1)"
  echo "settings: SCENARIOS=$SCENARIOS ROUNDS=$ROUNDS PARALLEL=$PARALLEL FIG7_S13_RUN=$FIG7 extra args: $*"
  echo
} >> "$OUT/environment.txt"
sed -n '/^## /h; /^## /!H; ${x;p}' "$OUT/environment.txt"

step "3. crosscheck: ASTRA-Sim vs the reference model ($SCENARIOS)"
PASSED="CROSSCHECK PASSED for $SCENARIOS"
if [ "${SKIP_CROSSCHECK:-0}" = 1 ]; then
  echo "skipped (SKIP_CROSSCHECK=1)"
elif [ -f "$OUT/crosscheck.txt" ] && grep -qxF "$PASSED" "$OUT/crosscheck.txt"; then
  echo "already passed (see crosscheck.txt)"
elif bash "$RUN" crosscheck --scenarios "$SCENARIOS" > "$OUT/crosscheck.txt" 2>&1; then
  echo "$PASSED" >> "$OUT/crosscheck.txt"
  tail -n 6 "$OUT/crosscheck.txt"
else
  tail -n 30 "$OUT/crosscheck.txt"
  echo "crosscheck FAILED: see $OUT/crosscheck.txt and README section 8" >&2
  exit 1
fi

step "4. llmcheck: one real scheduling call (S13, warm start)"
if bash "$RUN" llmcheck --llm-base-url "$URL" > "$OUT/llmcheck.txt" 2>&1; then
  grep -E "^(endpoint|sampling|attempt|PASS)" "$OUT/llmcheck.txt"
else
  tail -n 30 "$OUT/llmcheck.txt"
  echo "llmcheck FAILED: see $OUT/llmcheck.txt and README section 8" >&2
  exit 1
fi

started=$(date +%s)
if [ "$PARALLEL" = 1 ]; then
  step "5. Run: $SCENARIOS, $ROUNDS rounds, cold + warm start, + ablation (the long step)"
  echo "progress: tail -f $OUT/run.log"
  if ! run_part "$OUT" "$SCENARIOS" --with-ablation "$@" 2>&1 | tee -a "$OUT/run.log"; then
    exit 1
  fi
  if [ "$FIG7" = 1 ]; then
    step "6. S13 with the tool profiles of the paper's own S13 run (Fig. 7) -> $OUT/fig7_s13"
    if ! run_part "$OUT/fig7_s13" S13 --profile-overrides fig7-s13 "$@" 2>&1 \
         | tee -a "$OUT/run_fig7_s13.log"; then
      exit 1
    fi
  fi
else
  step "5. Full run in parallel parts -> $OUT/parts/p1, p2, p3$([ "$FIG7" = 1 ] && echo " (+ step 6 -> $OUT/fig7_s13)")"
  echo "progress: tail -f $OUT/parts/p*.log"
  mkdir -p "$OUT/parts"
  run_part "$OUT/parts/p1" S1-S6 "$@" >> "$OUT/parts/p1.log" 2>&1 & p1=$!
  run_part "$OUT/parts/p2" S7-S11 "$@" >> "$OUT/parts/p2.log" 2>&1 & p2=$!
  run_part "$OUT/parts/p3" S12,S13 --with-ablation "$@" >> "$OUT/parts/p3.log" 2>&1 & p3=$!
  pids="$p1 $p2 $p3"
  if [ "$FIG7" = 1 ]; then
    run_part "$OUT/fig7_s13" S13 --profile-overrides fig7-s13 "$@" >> "$OUT/parts/p4_fig7_s13.log" 2>&1 &
    pids="$pids $!"
  fi
  failed=0
  for pid in $pids; do wait "$pid" || failed=1; done
  tail -n 3 "$OUT"/parts/p*.log
  if [ "$failed" = 1 ]; then
    echo "a part did not finish (see $OUT/parts/p*.log); run this script again to continue" >&2
    exit 1
  fi
  step "5b. Merge the three parts into $OUT"
  bash "$RUN" merge --rounds "$ROUNDS" --from "$OUT/parts/p1,$OUT/parts/p2,$OUT/parts/p3" \
    --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
fi
echo "run time of this session: $(( ($(date +%s) - started) / 60 )) min" | tee -a "$OUT/environment.txt"

step "Done"
echo "Results:  $OUT/summary.md  (all tables; the other files are described in README section 6)"
[ "$FIG7" = 1 ] && echo "          $OUT/fig7_s13/summary.md  (S13 with the paper's Fig. 7 profiles)"
echo "To send them back:"
echo "  tar czf $(basename "$OUT").tgz -C \"$(dirname "$OUT")\" \"$(basename "$OUT")\""
