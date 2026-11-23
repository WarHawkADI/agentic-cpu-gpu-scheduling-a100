# Agentic CPU-GPU Scheduling: reproduction with the real scheduler LLM on an A100

This repository reproduces the experiment of **"Agentic CPU-GPU Scheduling for Heterogeneous AI Workloads"** (Tianxi Lu, Sherief Reda; [arXiv:2607.22242](https://arxiv.org/abs/2607.22242)). It is set up to run on **one Linux server with an NVIDIA A100**.

**What is real and what is simulated:**

| Part of the paper | In this reproduction |
|---|---|
| The scheduler LLM: DeepSeek-R1-Distill-Qwen-32B, 4-bit AWQ, served by vLLM | **Real.** It runs on the A100 and makes every scheduling decision. |
| The AI tools' CPU and GPU (the paper's RTX A5500 + AMD EPYC 7713) | **Simulated** by ASTRA-Sim (the version inside [LLMServingSim](https://github.com/casys-kaist/LLMServingSim)), timed from the paper's Table 1 tool profiles |
| GPU memory (VRAM) budget, out-of-memory | Checked before each simulation; a mapping that does not fit fails, like a real out-of-memory |
| Runtime monitor (running averages, what-if re-runs, swaps, exploration hints), baselines, brute-force oracle | Implemented as described in the paper |

The paper also keeps the scheduler LLM on its own GPU, separate from the tools (Sec. 4.1). So the LLM's own time is **not** part of any tool latency. It is measured and reported separately.

---

## Contents

1. [Requirements](#1-requirements)
2. [Quick start](#2-quick-start)
3. [Step by step](#3-step-by-step)
4. [How long it takes](#4-how-long-it-takes)
5. [If the run is interrupted](#5-if-the-run-is-interrupted)
6. [Output files](#6-output-files)
7. [Experiment settings](#7-experiment-settings)
8. [Troubleshooting](#8-troubleshooting)
9. [Other ways to run](#9-other-ways-to-run)
10. [Repository contents](#10-repository-contents)
11. [What the paper reports, for comparison](#11-what-the-paper-reports-for-comparison)
12. [Differences from the paper](#12-differences-from-the-paper)

---

## 1. Requirements

| Item | Needed |
|---|---|
| OS | **Ubuntu 22.04** (tested). Other recent Linux distributions should work but are untested. |
| GPU | One **NVIDIA A100** (40 GB or 80 GB). The model weights take about 18 GB, and vLLM uses the rest for its KV cache (about 4 GB per 16k-token request). Other NVIDIA GPUs with 40 GB or more also work. A 24 GB GPU is tight: use `MAX_MODEL_LEN=12288` and the default `PARALLEL=1`. |
| NVIDIA driver | Installed, so `nvidia-smi` works. The default vLLM (0.11.0) comes with a PyTorch built for **CUDA 12.8**. A driver for which `nvidia-smi` shows "CUDA Version: 12.8" or higher is safe. `setup_llm.sh` checks the driver (§3.3); see §8 if the check fails. |
| Disk | About **40 GB** free: 18 GB model, about 10 GB for vLLM and PyTorch, about 2 GB for the simulator |
| RAM | 32 GB or more |
| Network | Access to GitHub, PyPI and Hugging Face (for the setup only) |
| Rights | `sudo`, for `apt-get` in the first step. Without it, see `SKIP_APT=1` in §3.2. |
| Tools | `git`, `python3` (3.10 to 3.12), `tmux` (or `screen`) to keep the runs alive after you log out |

The scripts never install anything system-wide besides the `apt` packages in §3.2. Everything else goes into two Python virtual environments in your home folder (`~/venvs/sim`, `~/venvs/vllm`) and two folders (`~/LLMServingSim`, `~/models`).

---

## 2. Quick start

```bash
git clone https://github.com/WarHawkADI/agentic-cpu-gpu-scheduling-a100.git
cd agentic-cpu-gpu-scheduling-a100

bash scripts/setup_simulator.sh        # 1. build the simulator          (~10 min, no GPU)
bash scripts/setup_llm.sh              # 2. install vLLM + download model (~15-30 min)

tmux new -s llm                        # 3. LLM server, in its own tmux window
bash scripts/start_llm_server.sh       #    wait for "Application startup complete", then Ctrl-b d

tmux new -s exp                        # 4. the experiment, in another tmux window
bash scripts/run_paper_experiment.sh   #    several hours; Ctrl-b d to detach, tmux attach -t exp to return
```

Results are written to `results_real_llm/`. The main file is `results_real_llm/summary.md`. To send everything back:

```bash
tar czf results_real_llm.tgz results_real_llm
```

**Recommended:** before the full run, do a short trial of about 40 LLM calls. It checks the whole chain and measures how long one LLM call takes on your server (see §4):

```bash
SCENARIOS=S5,S13 ROUNDS=4 OUT=results_test bash scripts/run_paper_experiment.sh
```

---

## 3. Step by step

### 3.1 Get the code

```bash
git clone https://github.com/WarHawkADI/agentic-cpu-gpu-scheduling-a100.git
cd agentic-cpu-gpu-scheduling-a100
```

All commands below are run from this folder.

### 3.2 Build the simulator

```bash
bash scripts/setup_simulator.sh
```

It runs five steps and prints each one:

1. **System packages:** `apt-get install` of build-essential, cmake, git, python3-venv, python3-dev, python3-pip, libprotobuf-dev, protobuf-compiler and tmux. It uses `sudo` when not run as root.
2. **Python venv** `~/venvs/sim`, with up-to-date pip, setuptools and wheel. Ubuntu 22.04's own setuptools is too old to build Chakra.
3. **LLMServingSim:** clones it to `~/LLMServingSim` and checks out commit `560ce82`, the version this project was tested with, with its ASTRA-Sim and Chakra submodules.
4. **Converter and build:**
   - copies this project's `llm_converter_cpu_gpu.py` into LLMServingSim's Chakra;
   - runs LLMServingSim's own `scripts/compile.sh`, which installs Chakra and builds ASTRA-Sim's analytical backend.
5. **Check:**
   - the ASTRA-Sim binary exists;
   - the converter imports;
   - the 51 self-tests pass;
   - one scenario is simulated end to end.

The last lines should look like this:

```
selftest: all checks passed
=================== 4. Result
  S13 with GN,GQ,GN,CPU: end-to-end latency = 917000000 ns = 917.0 ms
...
Simulator ready. Next: bash scripts/setup_llm.sh
```

**Options**, set as environment variables in front of the command:

| Variable | Meaning |
|---|---|
| `SIM=/path` | Where to clone LLMServingSim. The default is `~/LLMServingSim`. |
| `SIM_VENV=/path` | Where to put the venv. The default is `~/venvs/sim`. |
| `SKIP_APT=1` | Skip `apt-get`, for example without sudo. The packages listed above must then already be installed; the script checks for git, cmake, make, g++, protoc and python3. |

If you change `SIM` or `SIM_VENV`, export the same value for all later steps.

### 3.3 Install vLLM and download the model

```bash
bash scripts/setup_llm.sh
```

1. Shows the GPU (`nvidia-smi`).
2. Creates the venv `~/venvs/vllm` and installs **vLLM 0.11.0**. It then prints the vLLM and PyTorch versions and whether CUDA is available; this must say `CUDA available: True`.
3. Downloads [`casperhansen/deepseek-r1-distill-qwen-32b-awq`](https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-32b-awq) (18 GB) to `~/models/deepseek-r1-distill-qwen-32b-awq`. This is the 4-bit AWQ build of `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, the model and quantization the paper uses. It is public, so no token is needed.

| Variable | Default |
|---|---|
| `VLLM_VERSION` | `0.11.0`. Set `latest` for the newest release. |
| `VLLM_VENV` | `~/venvs/vllm` |
| `MODEL_DIR` | `~/models/deepseek-r1-distill-qwen-32b-awq` |
| `HF_TOKEN` | Unset. Only needed if Hugging Face rate-limits the download. |

### 3.4 Start the LLM server

The server must keep running during the whole experiment, so start it in **tmux**:

```bash
tmux new -s llm
bash scripts/start_llm_server.sh
```

Loading takes a few minutes. It is ready when the log shows `Application startup complete`. Then detach with `Ctrl-b d`; the server keeps running. To return to it later, use `tmux attach -t llm`.

It runs:

```
vllm serve ~/models/deepseek-r1-distill-qwen-32b-awq --served-model-name deepseek-r1-distill-qwen-32b-awq \
  --host 127.0.0.1 --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90 --seed 0
```

- vLLM detects the AWQ quantization itself.
- The scheduler prompts are under 2,000 tokens and the answers at most 8,192, so a 16,384-token context is enough.
- The server listens only on `127.0.0.1`.

To check it from another shell:

```bash
curl http://127.0.0.1:8000/v1/models     # lists deepseek-r1-distill-qwen-32b-awq
nvidia-smi                               # vLLM holds about 90% of the GPU memory
```

| Variable | Default |
|---|---|
| `CUDA_VISIBLE_DEVICES` | `0`, the first GPU. Use e.g. `2` for another GPU. |
| `PORT` | `8000` |
| `GPU_MEM_UTIL` | `0.90`. Lower it if other jobs share the GPU. |
| `MAX_MODEL_LEN` | `16384` |

Extra arguments are passed to `vllm serve`.

### 3.5 Run the experiment

In a second tmux window:

```bash
tmux new -s exp
bash scripts/run_paper_experiment.sh
```

The script does everything in order, and logs it all in `results_real_llm/`:

| Step | What happens | Output |
|---|---|---|
| 1 | Waits until the LLM server answers (up to 30 minutes, so it can be started right after the server) | |
| 2 | Records GPU, driver, vLLM version, model and commits | `environment.txt` |
| 3 | **crosscheck**: every feasible tool mapping (391) is simulated by ASTRA-Sim and by the driver's reference model, and they must agree. Once it has passed, it is not repeated. | `crosscheck.txt` |
| 4 | **llmcheck**: one real scheduling call (S13, warm start). The answer must contain a valid mapping. It prints the time and tokens the call took. | `llmcheck.txt` |
| 5 | **The experiment**: brute-force oracle, baselines, and the agentic scheduler (LLM + runtime monitor) cold and warm start, 20 rounds, S1–S13, then the monitor ablation on S13. If it stops, for example because the LLM server stopped answering, it waits for the server and continues by itself (§5). | `summary.md` and the other files in §6 |
| 6 | **S13 once more, with the tool profiles of the paper's own S13 run.** The paper's S13 run used older VRAM values (Fig. 7: 627/463/417/58 MB) than its Table 1. This step uses those values, for a direct comparison with the paper's Figs. 7 and 9. About 40 extra LLM calls. Full run only. | `fig7_s13/` |

To watch progress, attach with `tmux attach -t exp`, or run `tail -f results_real_llm/run.log`. Each finished run prints a line such as:

```
[LLM-Warm] S13: best A:GN B:GQ C:GN D:CPU E2E=917.0 ms; oracle-equivalent: yes
```

| Variable | Default |
|---|---|
| `OUT` | `results_real_llm` |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `PARALLEL` | `1`. `3` runs three parts at once (§4). |
| `SCENARIOS` | `S1-S13` |
| `ROUNDS` | `20` |
| `WAIT_SERVER` | `1800` (seconds) |
| `AUTO_RESUME` | `3`. How many times a stopped run continues by itself. |
| `SKIP_CROSSCHECK` | Unset. Set `1` to skip step 3. |
| `SKIP_FIG7` | Unset. Set `1` to skip step 6. Step 6 runs only when `SCENARIOS` is `S1-S13`, so it is skipped in the trial anyway. |

Extra arguments go to the driver's `all` command. `python3 agentic_sched_experiment.py all -h` lists them, e.g. `--temperature`, `--max-tokens`, `--llm-timeout`.

### 3.6 Send the results back

```bash
tar czf results_real_llm.tgz results_real_llm
```

The archive is small: a few MB, mostly the LLM's answers in `llm_log.jsonl`.

### 3.7 Afterwards

- Stop the server: `tmux attach -t llm`, then `Ctrl-c`.
- To remove everything the setup created:
  ```bash
  rm -rf ~/LLMServingSim ~/venvs/sim ~/venvs/vllm ~/models/deepseek-r1-distill-qwen-32b-awq
  ```

---

## 4. How long it takes

| Part | Time |
|---|---|
| `setup_simulator.sh` | 5–15 min |
| `setup_llm.sh` | 15–30 min, mostly the download |
| crosscheck | a few minutes |
| Oracle and baselines | minutes. They use only the simulator. |
| **Agentic scheduler + ablation + step 6** | **About 660 LLM calls**, plus a few re-asks. This is almost all of the time. The calls are:<br>• 13 scenarios × 2 modes (cold, warm) × 20 rounds = 520<br>• 5 ablation configurations × 20 rounds = 100<br>• step 6: 2 modes × 20 rounds = 40 |

A reasoning model writes its chain of thought before answering. So one call takes from about half a minute to a few minutes on an A100, depending on how long it reasons. **Expect roughly 5–15 hours for the full run.** For a better estimate, run the trial from §2 first. Its `summary.md` ends with a table "Scheduler LLM calls (measured on the LLM server)". Multiply its *Mean s / decision* by 660.

**Faster: `PARALLEL=3`.**

```bash
PARALLEL=3 bash scripts/run_paper_experiment.sh
```

- It runs three independent parts against the same server: S1–S6, S7–S11, and S12–S13 + the ablation. Step 6 runs as a fourth part at the same time.
- Their logs go to `results_real_llm/parts/p1.log` … `p3.log` and `p4_fig7_s13.log`.
- vLLM batches the concurrent requests, so this is typically 2–3× faster overall.
- The parts are then merged into one `results_real_llm/summary.md`.
- **Results are the same in kind:** each run is still sequential, round by round.
- **The per-call LLM times in the summary are then measured under load.** If those times matter, use the default `PARALLEL=1`.

---

## 5. If the run is interrupted

**Short outages are handled automatically:**
1. A failed LLM request is retried 3 more times, after 10, 20 and 40 seconds.
2. If it still fails, the run stops. Everything finished so far is already saved.
3. `run_paper_experiment.sh` then waits 60 seconds, waits for the server to answer again (up to `WAIT_SERVER`), and continues the run. It does this up to `AUTO_RESUME` (3) times.

**Otherwise, run the same command again:**

```bash
bash scripts/run_paper_experiment.sh          # or PARALLEL=3 ... if that is what you used
```

- Every strategy run that had finished all its rounds is kept.
- Runs that were cut off are redone from round 0, so no run mixes two sessions.
- The LLM log is cleaned the same way, so it holds no duplicates.
- The crosscheck is not repeated once it has passed.

Common causes of an interruption:
- the LLM server stopped (restart it as in §3.4);
- the SSH session closed without tmux;
- the server was rebooted.

**To start from scratch instead,** delete the folder first: `rm -rf results_real_llm`.

---

## 6. Output files

All files are in `results_real_llm/`:

| File | Content | Paper |
|---|---|---|
| `summary.md` | **All result tables** (list below) | Tables 3–7, Fig. 5 |
| `rounds.csv` | One row per strategy, scenario and round: mapping, status (ok / oom), E2E, peak VRAM, match with the oracle, number of what-if re-runs | Data for Figs. 6 and 9 |
| `oracle.csv` | Every mapping of every scenario with its E2E and per-tool latencies | Table 3 |
| `ablation.csv` | Monitor ablation rows | Table 6 |
| `llm_log.jsonl` | One line per LLM decision (details below) | |
| `run_config.json` | All options used, number of simulator runs | |
| `environment.txt` | GPU, driver, vLLM version, model, commits, run time | |
| `crosscheck.txt` | ASTRA-Sim vs reference model on all 391 feasible mappings. It contains `VERDICT: PASS` and ends with `CROSSCHECK PASSED for S1-S13`. | |
| `llmcheck.txt` | The first real LLM call: its answer, time and tokens | |
| `run.log` | Console output of the run | |
| `fig7_s13/` | Step 6: S13 with the Fig. 7 tool profiles. It has the same kinds of files: `summary.md`, `rounds.csv`, `llm_log.jsonl`, … Its console output is in `run_fig7_s13.log`, or `parts/p4_fig7_s13.log` with `PARALLEL=3`. | Figs. 7 and 9, Table 5 (S13) |
| `parts/` | With `PARALLEL=3` only: the three parts before merging, and their logs | |

Each line of `llm_log.jsonl` holds:
- the full prompt;
- every attempt: the answer text, reasoning, token usage, `finish_reason` and seconds;
- the chosen mapping, and the measured result.

**Tables in `summary.md`:**

| Table | Paper | What it shows |
|---|---|---|
| Brute-force oracle vs the paper's Table 3 | Table 3 | The best mapping per scenario, and whether it equals the paper's |
| Oracle-mapping accuracy, S1–S10 | Table 4 | Per scenario, how many of the 4 tools the best round placed as the oracle does (a mapping within 1% of the oracle's E2E counts as 4/4). The last column counts the scenarios reaching 4/4. |
| Best E2E / oracle E2E | Fig. 5 | 1.00 means as good as the oracle |
| Three-way placement under VRAM limits, S11–S13 | Table 5 | Budget, All-GPU, Greedy-Mem and oracle E2E, and per LLM mode: the best E2E, the match, and the first round with 4/4 |
| Steady-state E2E, S5–S10 | Table 7 | Mean ± std of the E2E once a strategy keeps its best-found mapping |
| Monitor ablation, S13 warm | Table 6 | Full system, without the cpu→queue re-probe, without swaps, without hints, and LLM only |
| Scheduler LLM calls | — | Decisions, attempts, mean / median / max seconds per decision, mean answer tokens, answers cut off by `max_tokens`, fallbacks |

Strategies are labelled `LLM-Cold` and `LLM-Warm` for the real LLM, next to `All-GPU`, `HEFT-CF`, `UCB1`, `StarPU` and `Greedy-Mem`.

**Absolute milliseconds are lower than the paper's.** The simulator has no kernel interference or framework overhead; for example, S13's oracle is 917 ms here vs 1577 ms measured in the paper. Compare accuracy, E2E ÷ oracle, and the rounds needed to converge.

---

## 7. Experiment settings

| Setting | Value | Where to change it |
|---|---|---|
| Scheduler LLM | DeepSeek-R1-Distill-Qwen-32B, AWQ 4-bit, on vLLM | `MODEL_REPO` / `MODEL_DIR` in §3.3 |
| Sampling | temperature 0.6, top-p 0.95. **The paper does not state its sampling settings**, so this uses DeepSeek's own recommendation for the R1 distill models. The whole prompt goes in one user message, with no system prompt, also as DeepSeek recommends. | `--temperature`, `--top-p` |
| Answer budget | 8,192 tokens per call, including reasoning | `--max-tokens` |
| Re-asks | Up to 2 when an answer holds no valid JSON mapping, or was cut off by `max_tokens` | `--llm-retries` |
| Timeout | 900 s per call | `--llm-timeout` |
| Rounds per run | 20 | `ROUNDS` |
| Start modes | cold (workload types only) and warm (Table 1 profiles in the prompt) | `--modes` |
| Scenarios | S1–S4 diamond, S5–S10 fan-out, S11–S13 fan-out with VRAM budgets 1700 / 1700 / 1100 MB | `SCENARIOS` |
| Monitor (Sec. 3.2, Alg. 1) | Switched on in stages, as in the paper. S1–S4: round history only. S5–S10 (Factor A): adds running averages (sliding window of 5) and single-tool what-if re-runs, as in the paper's Fig. 4. S11–S13 (Factor B): adds swap re-runs, cpu→queue re-probes, and an exploration hint after 2 rounds without a change or an improvement. Re-runs happen only in rounds where the mapping changed (Alg. 1, line 3). The window size and hint delay are not given in the paper; these are our choices. | `--window`, `--hint-after`, `--monitor-level` |
| Oracle ties | Mappings within 1% of the best E2E count as oracle-equivalent | `--tie-tol` |
| Tool profiles | Paper's Table 1; step 6 uses Fig. 7's values for S13's four tools | `--profiles`, `--profile-overrides fig7-s13` |
| Prompt | Rebuilt from the paper (§10): DAG and device semantics, tool cards (warm) or workload-type tags (cold), soft guidelines, history, running averages, what-ifs, hints, then "reason briefly, end with one JSON line" | `build_prompt` in the driver |

What happens with a failed answer:
- Each re-ask sends the original prompt plus a short reminder to end with one JSON line. It never sends the previous answer, so the prompt does not grow.
- An answer cut off by `max_tokens` before the model finished reasoning is **not** parsed, because any JSON inside it would be a draft.
- If no attempt gives a valid mapping, the round keeps the previous mapping (All-GPU in round 0). It is counted under *Fallbacks* in the summary.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `setup_simulator.sh` fails in Chakra's `pip install` with a setuptools error | The script upgrades setuptools in the venv. Make sure no other venv is active (`deactivate`), then run it again. |
| `missing: protoc` / `cmake` with `SKIP_APT=1` | Install `build-essential cmake git python3-venv python3-dev libprotobuf-dev protobuf-compiler`, or ask the admin to. |
| `setup_llm.sh` prints `CUDA available: False`, or vLLM says the driver is too old | The driver is older than vLLM's CUDA build needs. Either update the NVIDIA driver, or install a vLLM built for an older CUDA: `rm -rf ~/venvs/vllm` and rerun with `VLLM_VERSION=<version>` that matches your driver (see vLLM's installation docs). You can also use vLLM's Docker image (§9). |
| vLLM: `CUDA out of memory` at start-up | Another process uses the GPU (`nvidia-smi`). Pick a free GPU with `CUDA_VISIBLE_DEVICES=1`, or lower `GPU_MEM_UTIL=0.80` or `MAX_MODEL_LEN=12288`. |
| `no answer from http://127.0.0.1:8000/v1/models` | The server is not running, or still loading. Check its tmux window (`tmux attach -t llm`). If you changed `PORT`, set `LLM_BASE_URL=http://127.0.0.1:<port>/v1`. |
| `llmcheck FAILED` | Read `results_real_llm/llmcheck.txt`; it shows the model's answer. With `finish_reason=length`, the answer was cut off: raise `--max-tokens` (and `MAX_MODEL_LEN` on the server). |
| `crosscheck FAILED` | Read the end of `crosscheck.txt`. The verdict names the cause and the option that fixes it. This should not happen with the pinned LLMServingSim commit; please send the file back. |
| Many *Fallbacks* or *Cut off by max_tokens* in the trial's summary | The model often reasons past 8,192 tokens. Give it more room, into a new folder: `OUT=results_real_llm_12k bash scripts/run_paper_experiment.sh --max-tokens 12288`. Prompt + answer still fit in the server's 16,384 tokens. |
| The run stopped and did not continue by itself | It gave up after `AUTO_RESUME` tries, or the server did not answer within `WAIT_SERVER`. Fix the server, then run the same command again (§5). |
| `ASTRA-Sim is not built at …` | Run `bash scripts/setup_simulator.sh`, or set `SIM=` to where LLMServingSim is. |

---

## 9. Other ways to run

**vLLM in Docker (untested).** Use this instead of §3.3–3.4 if the server has Docker and the NVIDIA Container Toolkit. Download the model as in §3.3, or let vLLM download it, then:

```bash
docker run --rm --gpus '"device=0"' -p 127.0.0.1:8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.11.0 \
  --model casperhansen/deepseek-r1-distill-qwen-32b-awq \
  --served-model-name deepseek-r1-distill-qwen-32b-awq \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --seed 0
```

Then run `bash scripts/run_paper_experiment.sh` as usual. The simulator side does not need Docker.

**Another OpenAI-compatible endpoint.** The driver works with any server that offers `/v1/chat/completions`:

```bash
LLM_BASE_URL=http://host:port/v1 OPENAI_API_KEY=... bash scripts/run_paper_experiment.sh
```

With several models on one server, add `--llm-model NAME`.

**Single commands.** `run_experiment.sh` runs any driver command on ASTRA-Sim:

```bash
bash run_experiment.sh selftest                       # converter + reference-model checks
bash run_experiment.sh crosscheck --scenarios S13     # ASTRA-Sim vs reference model
bash run_experiment.sh oracle --out results_oracle    # brute-force oracle only
bash run_experiment.sh llmcheck --llm-base-url http://127.0.0.1:8000/v1
bash run_experiment.sh agentic --llm-base-url http://127.0.0.1:8000/v1 --scenarios S13 --modes warm --out results_s13
bash run_experiment.sh merge --from dirA,dirB --out results_merged   # combine finished runs
```

**One simulation, shown step by step.** This prints the trace, ASTRA-Sim's own output, the per-tool timeline and the end-to-end latency:

```bash
bash simulate.sh S13 GN,GQ,GN,CPU     # the paper's S13 oracle mapping
bash simulate.sh S5 gpu,gpu,cpu,gpu
```

---

## 10. Repository contents

| Path | What it is |
|---|---|
| `agentic_sched_experiment.py` | The experiment driver. It contains the tool library (Table 1), the scenarios (Table 3), the simulator back-end, the runtime monitor (Algorithm 1), the prompt, the LLM client, the baselines, the oracle, the reports, resume and merge. |
| `llm_converter_cpu_gpu.py` | LLMServingSim's Chakra converter, extended with a `TOOLDAG` trace type that turns a tool DAG + device mapping into an ASTRA-Sim workload. Each tool becomes a CPU or GPU compute node, and `gpu_queue` tools get load + compute nodes that wait for the `gpu_now` phase. |
| `astra_configs/` | ASTRA-Sim system, network and memory configuration: 1 NPU, with per-node trace logging on |
| `run_experiment.sh` | Runs one driver command on ASTRA-Sim |
| `simulate.sh` | Simulates one scenario + mapping and shows each step |
| `scripts/setup_simulator.sh` | Step 1: builds LLMServingSim's ASTRA-Sim and installs the converter |
| `scripts/setup_llm.sh` | Step 2: installs vLLM and downloads the model |
| `scripts/start_llm_server.sh` | Step 3: starts the vLLM server |
| `scripts/run_paper_experiment.sh` | Step 4: checks, then the whole experiment, with resume |
| `NOTICE.md` | Licence of the converter, and third-party notes |

**How a scheduling round works:**
1. The driver builds the prompt: the DAG, the tool profiles (warm) or workload types (cold), the history, and the monitor's evidence.
2. The LLM answers with reasoning and then one JSON mapping, e.g. `{"A": "gpu_now", "B": "gpu_queue", "C": "gpu_now", "D": "cpu"}`.
3. The converter turns the mapping into a Chakra execution trace, or rejects it as out-of-memory.
4. ASTRA-Sim executes the trace. Its finish time is the E2E, and its per-node log gives each tool's latency.
5. The monitor updates its running averages and runs its what-if, swap and re-probe simulations for the next prompt.

---

## 11. What the paper reports, for comparison

These are the paper's numbers, measured on real hardware. Compare **oracle matches and convergence rounds** with `summary.md`. Do not compare milliseconds: the simulator's absolute times are lower (§12), and each `summary.md` lists its own oracle E2E.

**Table 4: best oracle match over 20 rounds (S1–S10).** The last column counts the scenarios with 4/4.

| Strategy | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | 4/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HEFT-CF | 3/4 | 3/4 | 3/4 | 4/4 | 3/4 | 3/4 | 4/4 | 3/4 | 3/4 | 2/4 | 2/10 |
| UCB1 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 10/10 |
| StarPU | 4/4 | 3/4 | 4/4 | 4/4 | 3/4 | 3/4 | 4/4 | 3/4 | 4/4 | 4/4 | 6/10 |
| LLM | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 10/10 |

The paper has one "LLM" row; `summary.md` shows cold and warm start separately (`LLM-Cold`, `LLM-Warm`). The paper also reports two S5 examples:
- **warm start** reaches the oracle in round 0, from the profile cards alone (Fig. 3);
- **cold start** reaches it after two to three rounds of monitor feedback (Fig. 4 says round 2, the text says round 3).

**Table 5: three-way placement (S11–S13).**

| Scenario | Budget MB | Oracle mapping | LLM match | First 4/4 |
|---|---|---|---|---|
| S11 | 1700 | GN, GN, GQ, GN | 4/4 | round 2 |
| S12 | 1700 | GN, GN, GN, CPU | 4/4 | round 1 |
| S13 | 1100 | GN, GQ, GN, CPU | 4/4 | round 3 |

In the paper's S13 run (Figs. 7–9):
- round 0 is out of memory;
- round 3 takes the C↔B swap that the monitor measured;
- the mapping is then kept.

Step 6 (`fig7_s13/`) repeats this run with the same tool profiles.

**Table 6: S13 ablation.**

| Configuration | Best match | Best E2E |
|---|---|---|
| Full system | 4/4 | 1529 ms |
| w/o cpu→queue reprobe | 1/4 | 4853 ms |
| w/o swap reprobe | 1/4 | 2174 ms |
| w/o exploration hints | 1/4 | 5060 ms |
| LLM only (no monitor) | 0/4 | OOM |

**Oracle mappings (Table 3).** `summary.md` checks each against the simulator's own oracle (table "Brute-force oracle vs the paper's Table 3"). With the default settings, the simulator's oracle is the paper's mapping in 9 of 13 scenarios. In two more, S1 and S3, the paper's mapping is within 1% of the simulator's best, so it counts as oracle-equivalent. The other two differ for the reasons in §12:
- **S2:** the paper's oracle is GPU, CPU, CPU, GPU;
- **S9:** the paper's oracle is all GPU.

---

## 12. Differences from the paper

- **Tool hardware is simulated.** ASTRA-Sim runs one GPU compute operation and one CPU operation at a time per device. On the real GPU, concurrent kernels overlap partly and interfere unevenly; the paper measured +43% to +250% per tool in Table 2. So absolute times are lower than the paper's: S13's oracle is 917 ms here vs 1577 ms in the paper's Table 5.
- **One CPU lane.** Two CPU tools run one after another here, while the paper's 64-core CPU ran them side by side. This changes S2's best mapping to GPU, CPU, GPU, GPU.
- **S9:** offloading `image_resize_x30` gains 4.9% in the simulator (548 → 521 ms). That is below the 5% stability threshold, and the paper's measured oracle keeps all four tools on the GPU.
- **Deterministic simulation.** The paper executes the workflow several times per round and reports the median (Sec. 4.1). Here, one simulation per mapping gives the same result every time.
- **The paper's exact prompt is not published.** `build_prompt` rebuilds it from the paper's description (Sec. 3.3, App. B) and its prompt excerpts (Figs. 3, 4 and 7):
  - DAG and execution semantics;
  - tool cards (warm start) or workload-type tags (cold start);
  - the ratio guideline with thresholds 1 and 7 for S1–S10, or the structural memory, queue and timeline rules for S11–S13;
  - the 5% stability threshold;
  - history with the best-ever round, running averages, what-if and swap blocks, and exploration hints;
  - per-tool reasoning, then one JSON line.
- **Sampling settings** are not given in the paper; see §7.
- **The paper's profiles are inconsistent in places.** Its S13 run used older VRAM values than its Table 1 (Fig. 7 caption). The main run uses Table 1, and step 6 uses Fig. 7's values; the oracle mapping is the same either way.
- **Answers are sampled** (temperature 0.6), so individual rounds differ from run to run, as they would in the paper.

The results with a rule-based *emulated* LLM (no model run) are in the companion repository [agentic-cpu-gpu-scheduling-llmservingsim](https://github.com/WarHawkADI/agentic-cpu-gpu-scheduling-llmservingsim). That repository also has a detailed comparison with the paper and the simulator facts verified from its source.
