#!/usr/bin/env python3
"""Run the experiment of "Agentic CPU-GPU Scheduling for Heterogeneous AI
Workloads" (Tianxi Lu, Sherief Reda; arXiv:2607.22242) on ASTRA-Sim.

The simulator stands in for the paper's testbed (an RTX A5500 for the tools,
an AMD EPYC 7713 host). Every device mapping a strategy asks for is turned
into Chakra execution traces by the TOOLDAG path of llm_converter_cpu_gpu.py
and executed by ASTRA-Sim. Strategies only ever see what a deployment would
see: end-to-end latency, per-tool latency, out-of-memory.

Subcommands (``<command> -h`` lists the options):

  selftest    converter + reference-model checks; needs neither ASTRA-Sim nor an LLM
  crosscheck  every mapping on ASTRA-Sim and on the reference model; diagnoses mismatches
  oracle      brute force over all 2^4 / 3^4 mappings (Table 3, Fig. 5 normaliser)
  baselines   All-GPU, HEFT-CF, UCB1, StarPU (S1-S10); All-GPU, memory-aware greedy (S11-S13)
  agentic     LLM scheduler + runtime monitor, cold and warm start (Tables 4/5, Figs. 5/6/9)
  ablation    monitor ablation on S13 (Table 6)
  all         oracle + baselines + agentic (+ ablation with --with-ablation)
  llmcheck    one real scheduling call against the LLM server, before a long run
  merge       combine result folders of separate (e.g. parallel) runs into one report
  export      write one scenario/mapping as a TOOLDAG text trace (optionally convert it)

Long runs: --resume continues an interrupted run in the same --out folder.
README.md covers the setup on an A100 server, how to run, and what each output means.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_CONVERTER_MODULE = "chakra.src.converter.llm_converter_cpu_gpu"

# ===========================================================================
# 1. Tool library -- Table 1 of the paper
# ===========================================================================

# tool: (gpu_ms, cpu_ms, vram_mb, load_ms, workload, workload type)
# Isolated medians of 30 runs on RTX A5500 + EPYC 7713. VRAM is the isolated
# peak, load the one-time model load a gpu_queue launch pays. The workload
# type is all a cold-start prompt gets (Sec. 3.3, Fig. 4).
TABLE1: Dict[str, Tuple[float, float, float, float, str, str]] = {
    "sentiment":      (7,    450,   316,  1022,  "DistilBERT-SST2 sentiment classification", "compute-bound"),
    "bert_base":      (13,   892,   486,  481,   "BERT-base NLI classification", "compute-bound"),
    "distilbert":     (7,    456,   314,  1079,  "DistilBERT-MNLI classification", "compute-bound"),
    "reranker":       (8,    440,   170,  739,   "cross-encoder passage reranking", "compute-bound"),
    "minilm_l6":      (5,    280,   128,  8890,  "MiniLM-L6 sentence embedding", "compute-bound"),
    "llama_3b":       (768,  19465, 6440, 5278,  "Llama-3.2-3B text generation", "compute-bound"),
    "resnet50":       (8,    216,   150,  7699,  "ResNet-50 image classification", "compute-bound"),
    "clip":           (12,   221,   656,  2944,  "CLIP ViT-B/32 image-text embedding", "compute-bound"),
    "whisper_base":   (35,   473,   442,  1792,  "Whisper-Base speech recognition", "compute-bound"),
    "llama_1b":       (447,  8482,  2484, 2519,  "Llama-3.2-1B text generation", "compute-bound"),
    "ocr":            (176,  1386,  248,  3244,  "TrOCR handwriting recognition", "compute-bound"),
    "groupby":        (2,    9,     61,   183,   "scatter-add group-by aggregation", "compute-light"),
    "image_resize":   (4,    17,    340,  4008,  "bilinear image resize + normalization", "compute-light"),
    "audio_features": (0.8,  2,     25,   364,   "STFT + MFCC audio feature extraction", "compute-light"),
    "vector_search":  (18,   43,    3106, 14701, "top-k vector similarity search", "memory-bound"),
    "doc_merger":     (39,   14,    692,  1218,  "embedding reduction (document merge)", "compute-light"),
    "web_search":     (128,  127,   0,    0,     "HTTP search API call", "I/O-bound"),
    "log_analyzer":   (641,  638,   0,    0,     "Apache log parsing", "I/O-bound"),
    "pdf_parser":     (6,    8,     0,    0,     "PDF text extraction", "I/O-bound"),
}

# No GPU implementation ("-" in Table 1); All-GPU leaves them on the CPU.
NEUTRAL_TOOLS = {"web_search", "log_analyzer", "pdf_parser"}
ALIASES = {"whisper": "whisper_base"}  # whisper_x4 is whisper_base repeated 4x

# Amplified tools ("<base>_x<k>", the base computation repeated k times, Sec.
# 3.1) default to k x Table 1 for time and the base VRAM and load -- the VRAM
# rule reproduces Table 3's "Total" column exactly. Where the paper reports a
# measured amplified time, the "paper" profiles use it. The paper is not
# self-consistent here (whisper_x4 CPU is 9305 ms in Fig. 3 but 2156 ms in
# Fig. 7; image_resize_x30 CPU 2201 ms in Fig. 3 but 521 ms in Sec. 3.3;
# ocr_x2 CPU 127654 ms in Fig. 3 vs 2 x 1386); the values kept are the ones
# consistent with k x Table 1.
AMPLIFIED_MEASURED: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "ocr_x2":            (370, None),   # GPU: Table 2 / Fig. 3
    "whisper_x4":        (136, 2156),   # GPU: Table 2 / Fig. 7; CPU: Fig. 7
    "vector_search_x15": (277, 739),    # GPU: Table 2; CPU: Fig. 3
    "image_resize_x30":  (145, 521),    # GPU: Table 2; CPU: Sec. 3.3
    "clip_x12":          (147, 3018),   # Fig. 7
    "bert_base_x12":     (153, 12141),  # Fig. 7
    "groupby_x60":       (90, 520),     # Fig. 7
}


@dataclass(frozen=True)
class ToolProfile:
    name: str
    base: str
    k: int
    gpu_ms: float
    cpu_ms: float
    vram_mb: float
    load_ms: float
    workload: str
    tag: str

    @property
    def neutral(self) -> bool:
        return self.base in NEUTRAL_TOOLS

    @property
    def ratio(self) -> float:
        return self.cpu_ms / self.gpu_ms if self.gpu_ms > 0 else float("inf")

    @property
    def description(self) -> str:
        return self.workload + (f" x{self.k}" if self.k > 1 else "")


PROFILE_FIELDS = ("gpu_ms", "cpu_ms", "vram_mb", "load_ms")

# The S13 tool cards of the paper's Fig. 7 prompt: the values the S13 run of
# Figs. 6, 7 and 9 actually used. Table 1 was re-profiled slightly higher
# afterwards (Fig. 7 caption). It matters: with Table 1, clip_x12 +
# bert_base_x12 = 1142 MB no longer fits S13's 1100 MB, so the storyboard's
# rounds 1-2 (A:GN B:GN) become infeasible -- the oracle mapping is the same
# either way. --profile-overrides fig7-s13 reproduces the storyboard.
FIG7_S13_PROFILES = {
    "clip_x12":      {"gpu_ms": 147, "cpu_ms": 3018,  "vram_mb": 627, "load_ms": 862},
    "bert_base_x12": {"gpu_ms": 153, "cpu_ms": 12141, "vram_mb": 463, "load_ms": 245},
    "whisper_x4":    {"gpu_ms": 136, "cpu_ms": 2156,  "vram_mb": 417, "load_ms": 1212},
    "groupby_x60":   {"gpu_ms": 90,  "cpu_ms": 520,   "vram_mb": 58,  "load_ms": 202},
}


def tool_profile(name: str, mode: str = "paper",
                 overrides: Optional[Dict[str, Dict[str, float]]] = None) -> ToolProfile:
    if mode not in ("paper", "k-scaled"):
        raise ValueError(f"unknown profile mode '{mode}'")
    base, k = name, 1
    m = re.fullmatch(r"(.+)_x(\d+)", name)
    if m:
        base, k = m.group(1), int(m.group(2))
    base = ALIASES.get(base, base)
    if base not in TABLE1:
        raise KeyError(f"unknown tool '{name}'")
    gpu, cpu, vram, load, workload, tag = TABLE1[base]
    gpu, cpu = gpu * k, cpu * k
    if mode == "paper" and name in AMPLIFIED_MEASURED:
        measured_gpu, measured_cpu = AMPLIFIED_MEASURED[name]
        gpu = gpu if measured_gpu is None else measured_gpu
        cpu = cpu if measured_cpu is None else measured_cpu
    values = {"gpu_ms": gpu, "cpu_ms": cpu, "vram_mb": vram, "load_ms": load}
    values.update((overrides or {}).get(name, {}))
    return ToolProfile(name, base, k, float(values["gpu_ms"]), float(values["cpu_ms"]),
                       float(values["vram_mb"]), float(values["load_ms"]), workload, tag)


def load_profile_overrides(spec: Optional[str]) -> Dict[str, Dict[str, float]]:
    """``fig7-s13`` (built in) or a JSON file {tool: {gpu_ms, cpu_ms, vram_mb, load_ms}}.

    A file is how profiles measured on other hardware come in: the paper's
    scheduler ports by re-running the profiler, not by retraining (Sec. 5).
    Any subset of the four fields may be given per tool.
    """
    if not spec:
        return {}
    if spec == "fig7-s13":
        return {tool: dict(v) for tool, v in FIG7_S13_PROFILES.items()}
    try:
        with open(spec, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit(f"--profile-overrides {spec}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"--profile-overrides {spec}: expected an object keyed by tool name")
    for tool, values in data.items():
        try:
            tool_profile(tool)
        except KeyError as e:
            raise SystemExit(f"--profile-overrides {spec}: {e}")
        if not isinstance(values, dict) or not set(values) <= set(PROFILE_FIELDS):
            raise SystemExit(f"--profile-overrides {spec}: '{tool}' may only set "
                             f"{', '.join(PROFILE_FIELDS)}")
        for key, v in values.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not v >= 0:
                raise SystemExit(f"--profile-overrides {spec}: {tool}.{key} must be a number >= 0")
    return data


# ===========================================================================
# 2. Scenarios -- Table 3 of the paper
# ===========================================================================

TASKS = ("A", "B", "C", "D")
GN, GQ, CPU = "gpu_now", "gpu_queue", "cpu"
BINARY = (GN, CPU)
THREE_WAY = (GN, GQ, CPU)
TOPOLOGIES = {
    "diamond": {"A": (), "B": ("A",), "C": ("A",), "D": ("B", "C")},
    "fanout": {"A": (), "B": (), "C": (), "D": ()},
}
TOPOLOGY_TEXT = {"diamond": "Diamond A->(B||C)->D", "fanout": "Fan-out A||B||C||D"}

Mapping = Tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    sid: str
    family: str                       # precedence | contention | memory
    topology: str
    tools: Tuple[str, ...]
    vram_budget_mb: Optional[float]   # None: unconstrained, CPU/GPU only
    paper_oracle: Mapping

    @property
    def tasks(self) -> Tuple[str, ...]:
        return TASKS

    @property
    def three_way(self) -> bool:
        return self.vram_budget_mb is not None

    @property
    def placements(self) -> Tuple[str, ...]:
        return THREE_WAY if self.three_way else BINARY

    def parents(self, task: str) -> Tuple[str, ...]:
        return TOPOLOGIES[self.topology][task]


def _scenario(sid, family, topology, tools, budget, oracle) -> Scenario:
    return Scenario(sid, family, topology, tuple(tools), budget, tuple(oracle))


SCENARIOS: Dict[str, Scenario] = {s.sid: s for s in (
    _scenario("S1", "precedence", "diamond",
              ["whisper_base", "llama_1b", "sentiment", "web_search"], None, [GN, GN, GN, CPU]),
    _scenario("S2", "precedence", "diamond",
              ["minilm_l6", "doc_merger", "groupby", "reranker"], None, [GN, CPU, CPU, GN]),
    _scenario("S3", "precedence", "diamond",
              ["whisper_base", "distilbert", "doc_merger", "web_search"], None, [GN, GN, CPU, GN]),
    _scenario("S4", "precedence", "diamond",
              ["minilm_l6", "vector_search", "doc_merger", "distilbert"], None, [GN, GN, CPU, GN]),
    _scenario("S5", "contention", "fanout",
              ["ocr_x2", "whisper_x4", "vector_search_x15", "image_resize_x30"], None,
              [GN, GN, CPU, GN]),
    _scenario("S6", "contention", "fanout",
              ["ocr_x2", "resnet50_x15", "groupby_x60", "vector_search_x15"], None,
              [GN, GN, GN, CPU]),
    _scenario("S7", "contention", "fanout",
              ["ocr_x2", "clip_x12", "bert_base_x12", "whisper_x4"], None, [GN, GN, GN, GN]),
    _scenario("S8", "contention", "fanout",
              ["ocr_x2", "resnet50_x15", "vector_search_x15", "groupby_x60"], None,
              [GN, GN, CPU, GN]),
    _scenario("S9", "contention", "fanout",
              ["whisper_x4", "clip_x12", "resnet50_x15", "image_resize_x30"], None,
              [GN, GN, GN, GN]),
    _scenario("S10", "contention", "fanout",
              ["ocr_x2", "bert_base_x12", "groupby_x60", "image_resize_x30"], None,
              [GN, GN, GN, CPU]),
    _scenario("S11", "memory", "fanout",
              ["ocr_x2", "clip_x12", "bert_base_x12", "whisper_x4"], 1700, [GN, GN, GQ, GN]),
    _scenario("S12", "memory", "fanout",
              ["ocr_x2", "bert_base_x12", "groupby_x60", "image_resize_x30"], 1700,
              [GN, GN, GN, CPU]),
    _scenario("S13", "memory", "fanout",
              ["clip_x12", "bert_base_x12", "whisper_x4", "groupby_x60"], 1100, [GN, GQ, GN, CPU]),
)}

# Progressive monitor activation (Sec. 3, 5): history only for serial
# precedence, + running averages and symmetric reprobes under GPU contention,
# + all-device reprobes, swap reprobes and exploration hints under VRAM limits.
MONITOR_LEVEL = {"precedence": 0, "contention": 1, "memory": 2}


def parse_scenarios(spec: str) -> List[Scenario]:
    if spec.strip().lower() == "all":
        return list(SCENARIOS.values())
    out: List[Scenario] = []
    for part in spec.split(","):
        part = part.strip().upper()
        if not part:
            continue
        m = re.fullmatch(r"S(\d+)-S?(\d+)", part)
        ids = [f"S{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1)] if m else [part]
        for sid in ids:
            if sid not in SCENARIOS:
                raise SystemExit(f"unknown scenario '{sid}' (S1..S13)")
            if SCENARIOS[sid] not in out:
                out.append(SCENARIOS[sid])
    if not out:
        raise SystemExit("no scenarios selected")
    return out


# ===========================================================================
# 3. Mappings
# ===========================================================================

_SHORT3 = {GN: "GN", GQ: "GQ", CPU: "CPU"}
_SHORT2 = {GN: "gpu", CPU: "cpu"}


def placement_word(scen: Scenario, placement: str) -> str:
    """What prompts and JSON answers use: gpu/cpu, or gpu_now/gpu_queue/cpu."""
    return placement if scen.three_way else _SHORT2[placement]


def fmt_mapping(scen: Scenario, mapping: Sequence[str]) -> str:
    names = _SHORT3 if scen.three_way else _SHORT2
    return " ".join(f"{t}:{names[p]}" for t, p in zip(scen.tasks, mapping))


def mapping_id(scen: Scenario, mapping: Sequence[str]) -> str:
    codes = {GN: "N", GQ: "Q", CPU: "C"} if scen.three_way else {GN: "G", CPU: "C"}
    return "".join(codes[p] for p in mapping)


def parse_placement(text: str, scen: Scenario) -> str:
    v = text.strip().lower().replace("-", "_").replace(" ", "_")
    if v in ("gpu", "gpu_now", "gn", "g", "now", "immediate", "gpu_immediate"):
        return GN
    if v in ("gpu_queue", "gq", "q", "queue", "queued", "gpu_queued"):
        if not scen.three_way:
            raise ValueError(f"gpu_queue is not available in {scen.sid} (CPU/GPU only)")
        return GQ
    if v in ("cpu", "c"):
        return CPU
    raise ValueError(f"unknown placement '{text}'")


def parse_mapping_arg(text: str, scen: Scenario) -> Mapping:
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) != len(scen.tasks):
        raise SystemExit(f"--mapping needs {len(scen.tasks)} placements (A,B,C,D), got '{text}'")
    try:
        return tuple(parse_placement(p, scen) for p in parts)
    except ValueError as e:
        raise SystemExit(str(e))


def enumerate_mappings(scen: Scenario) -> List[Mapping]:
    # All-GPU first, so exact ties resolve toward the GPU-first mapping.
    return list(itertools.product(scen.placements, repeat=len(scen.tasks)))


def per_tool_match(a: Sequence[str], b: Sequence[str]) -> int:
    return sum(x == y for x, y in zip(a, b))


# ===========================================================================
# 4. Converter and ET files
# ===========================================================================

class SimulatorError(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


def load_converter(module_name: str):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # ImportError, or anything raised while importing chakra
        raise SystemExit(
            f"cannot import the converter module '{module_name}': {type(e).__name__}: {e}\n"
            "Put llm_converter_cpu_gpu.py into the simulator's chakra/src/converter/ directory, "
            "make the 'chakra' package importable (pip install -e <chakra dir>, or PYTHONPATH), "
            "or pass --converter-module.")
    missing = [n for n in ("LLMConverter", "ToolTask", "VramBudgetExceeded", "Node",
                           "GlobalMetadata") if not hasattr(mod, n)]
    if missing or not hasattr(mod.LLMConverter, "convert_tool_dag"):
        raise SystemExit(f"{module_name} lacks {', '.join(missing) or 'convert_tool_dag'} -- "
                         "is it the TOOLDAG-enabled llm_converter_cpu_gpu.py?")
    return mod


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result, shift = 0, 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint in ET file")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("malformed varint in ET file")


def read_et(path: str, conv) -> Tuple[Any, List[Any]]:
    """Decode a Chakra ET: a varint-length-prefixed GlobalMetadata, then Nodes."""
    with open(path, "rb") as fh:
        data = fh.read()
    messages, pos = [], 0
    while pos < len(data):
        size, pos = _read_varint(data, pos)
        if pos + size > len(data):
            raise ValueError(f"{path}: truncated message")
        messages.append(data[pos:pos + size])
        pos += size
    if not messages:
        raise ValueError(f"{path}: empty ET file")
    meta = conv.GlobalMetadata()
    meta.ParseFromString(messages[0])
    nodes = []
    for raw in messages[1:]:
        node = conv.Node()
        node.ParseFromString(raw)
        nodes.append(node)
    return meta, nodes


def node_is_cpu(node) -> bool:
    for attr in node.attr:
        if attr.name != "is_cpu_op":
            continue
        for fld in ("bool_val", "int32_val", "int64_val", "uint32_val", "uint64_val"):
            try:
                if attr.HasField(fld):
                    return bool(getattr(attr, fld))
            except ValueError:  # field absent from this schema
                continue
        return False
    return False


# ===========================================================================
# 5. Backends: ASTRA-Sim, and a reference model of its compute semantics
# ===========================================================================

MODEL_SEMANTICS = ("lanes", "gpu_only", "unbounded")


def simulate_compute_graph(nodes: Sequence[Any], semantics: str = "lanes") -> Dict[str, Any]:
    """Discrete-event model of stock ASTRA-Sim on a compute-only ET.

    lanes      per NPU one GPU and one CPU resource, one in-flight op each
               (HardwareResource); ready nodes are issued smallest id first and
               pushed back while their resource is busy, re-tried after every
               completion (Workload::issue_dep_free_nodes).
    gpu_only   the same, but is_cpu_op ignored -- a feeder that never sees the
               CPU flag puts every node on the GPU.
    unbounded  no resource limits -- a simulator that runs all ready compute
               nodes concurrently.

    Used by ``selftest``, by ``--backend model`` for dry runs, and by
    ``crosscheck``, which compares ASTRA-Sim against all three.
    """
    if semantics not in MODEL_SEMANTICS:
        raise ValueError(f"unknown semantics '{semantics}'")
    spec = {}
    for n in nodes:
        lane = "gpu" if semantics == "gpu_only" else ("cpu" if node_is_cpu(n) else "gpu")
        spec[n.id] = (lane, int(n.duration_micros), list(dict.fromkeys(n.data_deps)))
    children = defaultdict(list)
    indegree = {}
    for nid, (_, _, deps) in spec.items():
        for d in deps:
            if d not in spec:
                raise SimulatorError(f"node {nid} depends on node {d}, which is not in the ET")
            children[d].append(nid)
        indegree[nid] = len(deps)
    ready = [nid for nid, k in indegree.items() if k == 0]
    heapify(ready)
    busy = {"gpu": False, "cpu": False}
    events: List[Tuple[int, int, int]] = []
    start: Dict[int, int] = {}
    finish: Dict[int, int] = {}
    state = {"now": 0, "seq": 0}

    def issue() -> None:
        held = []
        while ready:
            nid = heappop(ready)
            lane, duration, _ = spec[nid]
            if semantics != "unbounded":
                if busy[lane]:
                    held.append(nid)
                    continue
                busy[lane] = True
            start[nid] = state["now"]
            heappush(events, (state["now"] + duration, state["seq"], nid))
            state["seq"] += 1
        for nid in held:
            heappush(ready, nid)

    issue()
    while events:
        now, _, nid = heappop(events)
        state["now"] = now
        finish[nid] = now
        if semantics != "unbounded":
            busy[spec[nid][0]] = False
        for c in children[nid]:
            indegree[c] -= 1
            if indegree[c] == 0:
                heappush(ready, c)
        issue()
    if len(finish) != len(spec):
        raise SimulatorError("the ET's dependencies form a cycle; some nodes never ran")
    return {"finish": max(finish.values(), default=0), "node_start": start,
            "node_finish": finish}


class ModelBackend:
    name = "model"

    def __init__(self, conv, semantics: str = "lanes"):
        self.conv = conv
        self.semantics = semantics

    def run(self, prefix: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        _, nodes = read_et(summary["files"][0], self.conv)
        return simulate_compute_graph(nodes, self.semantics)


# ASTRA-Sim 2.x prints "sys[0] finished, <ticks> cycles, exposed communication ..."
# and, with tracing on, "callback,sys->id=0, tick=<t>, node->id=<id>, ..." per node.
DEFAULT_FINISH_RE = r"sys\[(?P<npu>\d+)\][^\n]*?finished[^\d\n]*(?P<ticks>\d+)"
DEFAULT_CALLBACK_RE = (r"callback,\s*sys->id=(?P<npu>\d+),\s*tick=(?P<tick>\d+),"
                       r"\s*node->id=(?P<id>\d+)")


class AstraBackend:
    name = "astra"

    def __init__(self, cmd: Optional[str], finish_re: str, callback_re: str,
                 log_file: Optional[str], timeout: float, cwd: Optional[str],
                 stdin_text: str = "exit\n"):
        if not cmd:
            raise SystemExit("--backend astra needs --astra-cmd (or ASTRA_SIM_CMD), e.g. "
                             "'<astra-sim binary> --workload-configuration={workload} "
                             "--system-configuration=... --network-configuration=... "
                             "--remote-memory-configuration=...'; use --backend model for a dry run")
        if "{workload}" not in cmd:
            raise SystemExit("--astra-cmd must contain {workload}, the ET prefix "
                             "(ASTRA-Sim opens <prefix>.<npu>.et)")
        self.cmd = cmd
        self.finish_re = re.compile(finish_re)
        self.callback_re = re.compile(callback_re)
        if "ticks" not in self.finish_re.groupindex:
            raise SystemExit("--finish-regex needs a named group (?P<ticks>...)")
        if not {"tick", "id"} <= set(self.callback_re.groupindex):
            raise SystemExit("--callback-regex needs named groups (?P<tick>...) and (?P<id>...)")
        self.log_file = log_file
        self.timeout = timeout
        self.cwd = cwd
        # LLMServingSim's AnalyticalAstra is interactive: when an NPU listed in
        # --end-npu-ids finishes it prints "sys[N] iteration I finished, ...",
        # then "Waiting", and reads one command from stdin; "exit" ends the run.
        # A one-shot ASTRA-Sim never reads stdin, so the default is harmless there.
        self.stdin_text = stdin_text

    def _log_size(self) -> int:
        if self.log_file and os.path.exists(self.log_file):
            return os.path.getsize(self.log_file)
        return 0

    def _read_log(self, offset: int) -> str:
        if not self.log_file or not os.path.exists(self.log_file):
            return ""
        with open(self.log_file, "r", errors="replace") as fh:
            if os.path.getsize(self.log_file) >= offset:
                fh.seek(offset)  # only what this run appended
            return "\n" + fh.read()

    def run(self, prefix: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        cmd = self.cmd.format(workload=prefix, et_dir=os.path.dirname(prefix),
                              num_npus=len(summary["files"]))
        args = cmd if os.name == "nt" else shlex.split(cmd)
        offset = self._log_size()
        try:
            proc = subprocess.run(args, input=self.stdin_text, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, universal_newlines=True,
                                  errors="replace", timeout=self.timeout, cwd=self.cwd)
        except FileNotFoundError as e:
            raise SimulatorError(f"cannot launch ASTRA-Sim ({e}): {cmd}") from None
        except subprocess.TimeoutExpired:
            raise SimulatorError(f"ASTRA-Sim timed out after {self.timeout:g}s: {cmd}") from None
        text = proc.stdout + "\n" + proc.stderr + self._read_log(offset)
        if proc.returncode != 0:
            raise SimulatorError(f"ASTRA-Sim exited with {proc.returncode}: {cmd}\n"
                                 f"--- output tail ---\n{text[-3000:]}")
        finishes: Dict[int, int] = {}
        for m in self.finish_re.finditer(text):
            npu = int(m.group("npu")) if "npu" in self.finish_re.groupindex else len(finishes)
            finishes[npu] = int(m.group("ticks"))
        if not finishes:
            raise SimulatorError("no output line matched --finish-regex; adapt it to what this "
                                 f"build prints.\n--- output tail ---\n{text[-3000:]}")
        first_npu = min(finishes)
        node_finish: Dict[int, int] = {}
        for m in self.callback_re.finditer(text):
            npu = int(m.group("npu")) if "npu" in self.callback_re.groupindex else first_npu
            if npu == first_npu:
                node_finish[int(m.group("id"))] = int(m.group("tick"))
        return {"finish": max(finishes.values()), "node_finish": node_finish,
                "npus_reported": len(finishes)}


# ===========================================================================
# 6. Measuring a mapping
# ===========================================================================

@dataclass
class Measurement:
    status: str                       # "ok" or "oom"
    e2e_ms: Optional[float]
    peak_vram_mb: float
    per_tool_ms: Dict[str, float] = field(default_factory=dict)
    detail: str = ""
    samples: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def per_tool_latency(summary: Dict[str, Any], node_finish: Dict[int, int],
                     ticks_per_ms: float) -> Dict[str, float]:
    """Each tool's latency from the moment its inputs are ready to its finish.

    That is what a tool "takes" in a real run, contention included: on a busy
    resource the wait is part of it. Empty if the backend did not report the
    finish of every tool's last node.
    """
    last = summary["last_node"]
    if not node_finish or any(last[t] not in node_finish for t in last):
        return {}
    done = {t: node_finish[last[t]] for t in last}
    out = {}
    for t, parents in summary["dag_parents"].items():
        ready = max((done[p] for p in parents), default=0)
        out[t] = (done[t] - ready) / ticks_per_ms
    return out


class ToolDagSimulator:
    """Turns (scenario, mapping) into ETs, runs a backend, reports a Measurement.

    The simulation is deterministic, so without noise every mapping is run
    once and memoised -- the paper's "median of several executions" of a
    mapping is that single value. ``noise_cv`` > 0 perturbs every tool time by
    an independent lognormal factor per execution and reports the median of
    ``repeats`` executions, re-measured on every call, to emulate run-to-run
    variance.
    """

    def __init__(self, conv, backend, *, profile_mode: str = "paper",
                 profile_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                 num_npus: int = 1, time_units_per_ms: float = 1000.0,
                 sim_ticks_per_ms: float = 1e6, contention: str = "simulator",
                 slice_ms: float = 0.0, cpu_flag: str = "bool", workdir: Optional[str] = None,
                 keep_et: bool = False, repeats: int = 1, noise_cv: float = 0.0, seed: int = 0):
        if num_npus < 1:
            raise SystemExit("--num-npus must be >= 1")
        if noise_cv < 0:
            raise SystemExit("--noise-cv must be >= 0")
        self.conv = conv
        self.backend = backend
        self.profile_mode = profile_mode
        self.profile_overrides = profile_overrides or {}
        self.num_npus = num_npus
        self.time_units_per_ms = time_units_per_ms
        self.sim_ticks_per_ms = sim_ticks_per_ms
        self.contention = contention
        self.slice_ms = slice_ms
        self.cpu_flag = cpu_flag
        self.own_workdir = workdir is None
        self.workdir = workdir or tempfile.mkdtemp(prefix="tooldag_")
        os.makedirs(self.workdir, exist_ok=True)
        self.keep_et = keep_et
        self.noise_cv = noise_cv
        self.repeats = max(1, repeats) if noise_cv > 0 else 1
        self.rng = random.Random(seed)
        self.runs = 0
        self._memo: Dict[Tuple[str, Mapping], Measurement] = {}
        self._profiles: Dict[str, ToolProfile] = {}

    @property
    def ticks_per_ms(self) -> float:
        # The reference model counts in duration_micros units; ASTRA-Sim in ticks.
        return self.time_units_per_ms if self.backend.name == "model" else self.sim_ticks_per_ms

    def profile(self, tool: str) -> ToolProfile:
        if tool not in self._profiles:
            self._profiles[tool] = tool_profile(tool, self.profile_mode, self.profile_overrides)
        return self._profiles[tool]

    def profiles(self, scen: Scenario) -> List[ToolProfile]:
        return [self.profile(t) for t in scen.tools]

    def tool_tasks(self, scen: Scenario, mapping: Sequence[str], factors=None) -> List[Any]:
        tasks = []
        for i, (task, tool, placement) in enumerate(zip(scen.tasks, scen.tools, mapping)):
            p = self.profile(tool)
            fg, fc, fl = factors[i] if factors else (1.0, 1.0, 1.0)
            tasks.append(self.conv.ToolTask.make(task, tool, placement, p.gpu_ms * fg,
                                                 p.cpu_ms * fc, p.load_ms * fl, p.vram_mb,
                                                 scen.parents(task)))
        return tasks

    def plan(self, scen: Scenario, mapping: Sequence[str]):
        converter = self.conv.LLMConverter("plan", os.path.join(self.workdir, "plan"),
                                           self.num_npus)
        return converter.analyze_tool_dag(self.tool_tasks(scen, mapping),
                                          scen.vram_budget_mb or 0.0)

    def feasible(self, scen: Scenario, mapping: Sequence[str]) -> bool:
        return self.plan(scen, mapping).feasible

    def measure(self, scen: Scenario, mapping: Sequence[str]) -> Measurement:
        mapping = tuple(mapping)
        key = (scen.sid, mapping)
        if self.noise_cv == 0 and key in self._memo:
            return self._memo[key]
        plan = self.plan(scen, mapping)
        if not plan.feasible:
            meas = Measurement("oom", None, plan.peak_vram_mb, {},
                               f"{'+'.join(plan.peak_tasks)}={plan.peak_vram_mb:g}MB"
                               f">{plan.vram_budget_mb:g}MB")
        else:
            runs = [self._run_once(scen, mapping,
                                   self._noise_factors() if self.noise_cv > 0 else None)
                    for _ in range(self.repeats)]
            per_tool = {}
            if all(r[1] for r in runs):
                per_tool = {t: statistics.median(r[1][t] for r in runs) for t in scen.tasks}
            meas = Measurement("ok", statistics.median(r[0] for r in runs), plan.peak_vram_mb,
                               per_tool, samples=[r[0] for r in runs])
        if self.noise_cv == 0:
            self._memo[key] = meas
        return meas

    def _noise_factors(self) -> List[Tuple[float, float, float]]:
        sigma = math.sqrt(math.log1p(self.noise_cv ** 2))
        mu = -0.5 * sigma * sigma  # mean-one lognormal
        return [tuple(math.exp(self.rng.gauss(mu, sigma)) for _ in range(3)) for _ in TASKS]

    def _run_once(self, scen: Scenario, mapping: Mapping, factors) -> Tuple[float, Dict[str, float]]:
        self.runs += 1
        run_dir = os.path.join(self.workdir, scen.sid, f"{mapping_id(scen, mapping)}_{self.runs}")
        os.makedirs(run_dir, exist_ok=True)
        prefix = os.path.join(run_dir, "tooldag")
        converter = self.conv.LLMConverter(f"TOOLDAG {scen.sid} {fmt_mapping(scen, mapping)}",
                                           prefix, self.num_npus)
        summary = converter.convert_tool_dag(
            self.tool_tasks(scen, mapping, factors), vram_budget_mb=scen.vram_budget_mb or 0.0,
            time_units_per_ms=self.time_units_per_ms, contention=self.contention,
            slice_ms=self.slice_ms, cpu_flag=self.cpu_flag)
        result = self.backend.run(prefix, summary)  # on failure the ETs stay for inspection
        e2e = result["finish"] / self.ticks_per_ms
        per_tool = per_tool_latency(summary, result.get("node_finish") or {}, self.ticks_per_ms)
        if not self.keep_et:
            shutil.rmtree(run_dir, ignore_errors=True)
        return e2e, per_tool

    def cleanup(self) -> None:
        if self.own_workdir and not self.keep_et:
            shutil.rmtree(self.workdir, ignore_errors=True)


def make_simulator(args, conv, backend=None, **overrides) -> ToolDagSimulator:
    if backend is None:
        backend = (ModelBackend(conv) if args.backend == "model" else
                   AstraBackend(args.astra_cmd, args.finish_regex, args.callback_regex,
                                args.astra_log_file, args.astra_timeout, args.astra_cwd,
                                args.astra_stdin.replace("\\n", "\n")))
    kw = dict(profile_mode=args.profiles,
              profile_overrides=load_profile_overrides(args.profile_overrides),
              num_npus=args.num_npus,
              time_units_per_ms=args.time_units_per_ms, sim_ticks_per_ms=args.sim_ticks_per_ms,
              contention=args.contention, slice_ms=args.slice_ms, cpu_flag=args.cpu_flag,
              workdir=args.workdir, keep_et=args.keep_et, repeats=args.repeats,
              noise_cv=args.noise_cv, seed=args.seed)
    kw.update(overrides)
    return ToolDagSimulator(conv, backend, **kw)


# ===========================================================================
# 7. Oracle and per-round records
# ===========================================================================

@dataclass
class OracleResult:
    scen: Scenario
    rows: List[Tuple[Mapping, Measurement]]
    best: Mapping
    best_e2e: float
    equivalent: List[Mapping]   # within the tie tolerance of the best
    tie_tol: float

    def measurement(self, mapping: Sequence[str]) -> Measurement:
        mapping = tuple(mapping)
        for mp, meas in self.rows:
            if mp == mapping:
                return meas
        raise KeyError(mapping)

    def match_tol(self, mapping: Sequence[str]) -> int:
        return max(per_tool_match(mapping, o) for o in self.equivalent)


def compute_oracle(sim: ToolDagSimulator, scen: Scenario, tie_tol: float) -> OracleResult:
    """The paper's oracle: the feasible mapping with the lowest E2E, by brute force.

    The simulator is deterministic, so near-ties are exact numbers rather than
    noise; mappings within ``tie_tol`` of the best count as oracle-equivalent
    (the paper treats S13's tool D the same way, Sec. 4.5).
    """
    rows = [(mp, sim.measure(scen, mp)) for mp in enumerate_mappings(scen)]
    feasible = [(mp, m) for mp, m in rows if m.ok]
    if not feasible:
        raise SimulatorError(f"{scen.sid}: no feasible mapping")
    best_mp, best_m = min(feasible, key=lambda r: r[1].e2e_ms)
    limit = best_m.e2e_ms * (1.0 + tie_tol) + 1e-9
    equivalent = [mp for mp, m in feasible if m.e2e_ms <= limit]
    return OracleResult(scen, rows, best_mp, best_m.e2e_ms, equivalent, tie_tol)


@dataclass
class RoundRecord:
    scenario: str
    strategy: str
    round: int
    mapping: Mapping
    status: str
    e2e_ms: Optional[float]
    peak_vram_mb: float
    match_strict: int
    match_tol: int
    oracle_equiv: bool
    probes: int = 0
    note: str = ""


def make_record(oracle: OracleResult, strategy: str, rnd: int, mapping: Sequence[str],
                meas: Measurement, probes: int = 0, note: str = "") -> RoundRecord:
    # An OOM round scores 0/4: the mapping cannot run at all (Table 6, Fig. 9).
    mapping = tuple(mapping)
    ok = meas.ok
    return RoundRecord(oracle.scen.sid, strategy, rnd, mapping, meas.status, meas.e2e_ms,
                       meas.peak_vram_mb, per_tool_match(mapping, oracle.best) if ok else 0,
                       oracle.match_tol(mapping) if ok else 0,
                       ok and mapping in oracle.equivalent, probes, note)


# ===========================================================================
# 8. Baselines (Sec. 4.1)
# ===========================================================================

def all_gpu_mapping(sim: ToolDagSimulator, scen: Scenario) -> Mapping:
    """Every GPU-capable tool on the GPU -- the LangGraph/LangChain default."""
    return tuple(CPU if p.neutral else GN for p in sim.profiles(scen))


def heft_cf_mapping(sim: ToolDagSimulator, scen: Scenario, contention_factor: float) -> Mapping:
    """HEFT with a static contention factor (the paper's HEFT-CF).

    Upward ranks over mean device cost, then each task goes to the device
    with the earliest estimated finish. The CPU is one processor, as in HEFT.
    The GPU is modelled the way GPU tools are launched in practice --
    concurrently -- with a single static factor inflating a GPU task's time
    whenever it would overlap another GPU task. That factor is exactly the
    assumption the paper says breaks under real four-way contention.
    """
    prof = dict(zip(scen.tasks, sim.profiles(scen)))
    children = {t: [c for c in scen.tasks if t in scen.parents(c)] for t in scen.tasks}
    rank: Dict[str, float] = {}

    def upward(t: str) -> float:
        if t not in rank:
            p = prof[t]
            rank[t] = (p.gpu_ms + p.cpu_ms) / 2 + max((upward(c) for c in children[t]),
                                                      default=0.0)
        return rank[t]

    order = sorted(scen.tasks, key=lambda t: (-upward(t), scen.tasks.index(t)))
    gpu_busy: List[Tuple[float, float]] = []
    cpu_free = 0.0
    finish: Dict[str, float] = {}
    mapping: Dict[str, str] = {}
    for t in order:  # a parent always outranks its children (positive costs)
        p = prof[t]
        ready = max((finish[q] for q in scen.parents(t)), default=0.0)
        overlaps = any(s < ready + p.gpu_ms and ready < e for s, e in gpu_busy)
        gpu_eft = ready + p.gpu_ms * (contention_factor if overlaps else 1.0)
        cpu_eft = max(cpu_free, ready) + p.cpu_ms
        if gpu_eft <= cpu_eft:
            mapping[t], finish[t] = GN, gpu_eft
            gpu_busy.append((ready, gpu_eft))
        else:
            mapping[t], finish[t] = CPU, cpu_eft
            cpu_free = cpu_eft
    return tuple(mapping[t] for t in scen.tasks)


def run_ucb1(sim: ToolDagSimulator, scen: Scenario, oracle: OracleResult, rounds: int,
             rng: random.Random) -> List[RoundRecord]:
    """UCB1 with every complete (feasible) mapping as an arm (Auer et al.).

    Each arm is pulled once, in a seeded random order, then the upper
    confidence bound decides. Reward = lower_bound / E2E in (0, 1], with the
    lower bound the slowest tool's best-device time.
    """
    arms = [mp for mp in enumerate_mappings(scen) if sim.feasible(scen, mp)]
    rng.shuffle(arms)
    lower = max(min(p.gpu_ms, p.cpu_ms) for p in sim.profiles(scen))
    pulls: Dict[Mapping, int] = defaultdict(int)
    reward_sum: Dict[Mapping, float] = defaultdict(float)
    records = []
    for r in range(rounds):
        if r < len(arms):
            arm = arms[r]
        else:
            arm = max(arms, key=lambda a: reward_sum[a] / pulls[a]
                      + math.sqrt(2.0 * math.log(r) / pulls[a]))
        meas = sim.measure(scen, arm)
        reward = min(1.0, lower / meas.e2e_ms) if meas.ok and meas.e2e_ms > 0 else 0.0
        pulls[arm] += 1
        reward_sum[arm] += reward
        records.append(make_record(oracle, "UCB1", r, arm, meas))
    return records


def run_starpu(sim: ToolDagSimulator, scen: Scenario, oracle: OracleResult,
               rounds: int) -> List[RoundRecord]:
    """StarPU-style greedy: earliest estimated completion, online model updates.

    Every round each tool takes the device with the lower estimated latency;
    with GPU tools launched concurrently both devices see the same ready
    time, so the earliest completion is just the smaller estimate. Estimates
    start from the isolated profiles and become the running mean of the
    latencies measured on that device (StarPU's history-based model).
    Without per-tool times from the backend the model cannot update.
    """
    prof = dict(zip(scen.tasks, sim.profiles(scen)))
    history: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def estimate(t: str, device: str) -> float:
        seen = history[(t, device)]
        if seen:
            return sum(seen) / len(seen)
        return prof[t].gpu_ms if device == GN else prof[t].cpu_ms

    records = []
    for r in range(rounds):
        mapping = tuple(GN if estimate(t, GN) <= estimate(t, CPU) else CPU for t in scen.tasks)
        meas = sim.measure(scen, mapping)
        note = ""
        if meas.ok and meas.per_tool_ms:
            for t, device in zip(scen.tasks, mapping):
                history[(t, device)].append(meas.per_tool_ms[t])
        elif meas.ok:
            note = "no per-tool times from the backend; model not updated"
        records.append(make_record(oracle, "StarPU", r, mapping, meas, note=note))
    return records


def greedy_memory_mapping(sim: ToolDagSimulator, scen: Scenario) -> Mapping:
    """Memory-aware greedy (Sec. 4.4): gpu_now by GPU speedup while VRAM lasts, else CPU."""
    prof = dict(zip(scen.tasks, sim.profiles(scen)))
    order = sorted(scen.tasks, key=lambda t: (-prof[t].ratio, scen.tasks.index(t)))
    mapping = {t: CPU for t in scen.tasks}
    for t in order:
        trial = dict(mapping)
        trial[t] = GN
        if sim.feasible(scen, tuple(trial[x] for x in scen.tasks)):
            mapping = trial
    return tuple(mapping[t] for t in scen.tasks)


# ===========================================================================
# 9. Runtime monitor -- Algorithm 1
# ===========================================================================

@dataclass
class Probe:
    kind: str        # "single" or "swap"
    label: str
    mapping: Mapping
    meas: Measurement


class RuntimeMonitor:
    """The paper's runtime monitor (Algorithm 1).

    It never chooses a mapping. After each round it updates sliding-window
    per-tool latencies and the best mapping seen (lines 1-2); when the
    mapping changed it measures what-if alternatives (lines 3-14); it issues
    an exploration hint when the mapping stagnates (line 15); and it renders
    all of that as prompt fields (line 16).

    level 0  execution history only (serial precedence, S1-S4)
    level 1  + running averages and symmetric single-tool reprobes (S5-S10)
    level 2  + reprobes to every other device, swap reprobes, hints (S11-S13)

    What-if executions feed the running averages too, which is how a device
    the scheduler never picked gets a measured average (Fig. 4).
    ``enabled=False`` is the "LLM only" ablation: no feedback at all.
    """

    def __init__(self, scen: Scenario, sim: ToolDagSimulator, level: int, window: int = 5,
                 hint_after: int = 2, cpu_to_queue: bool = True, swap: bool = True,
                 hints: bool = True, enabled: bool = True):
        self.scen = scen
        self.sim = sim
        self.level = level
        self.hint_after = hint_after
        self.cpu_to_queue = cpu_to_queue
        self.swap = swap
        self.hints = hints
        self.enabled = enabled
        self.windows: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))
        self.history: List[Tuple[int, Mapping, Measurement]] = []
        self.best: Optional[Tuple[float, Mapping, int]] = None
        self.probes: List[Probe] = []
        self.probe_base: Optional[Tuple[int, Mapping, Measurement]] = None
        self.previous: Optional[Mapping] = None
        self.stagnant = 0
        self.hint: Optional[str] = None
        self.probe_count = 0

    # -- per-round update ---------------------------------------------------

    def observe(self, rnd: int, mapping: Sequence[str], meas: Measurement) -> None:
        mapping = tuple(mapping)
        self._absorb(mapping, meas)
        self.history.append((rnd, mapping, meas))
        improved = meas.ok and (self.best is None or meas.e2e_ms < self.best[0])
        if improved:
            self.best = (meas.e2e_ms, mapping, rnd)
        changed = mapping != self.previous
        if self.enabled and self.level >= 1 and changed:
            self._reprobe(rnd, mapping, meas)
        if self.enabled and self.level >= 2 and self.hints:
            self.stagnant = 0 if (changed or improved) else self.stagnant + 1
            self.hint = self._hint_text() if self.stagnant >= self.hint_after else None
        self.previous = mapping

    def _absorb(self, mapping: Mapping, meas: Measurement) -> None:
        if meas.ok and meas.per_tool_ms:
            for t, p in zip(self.scen.tasks, mapping):
                self.windows[(t, p)].append(meas.per_tool_ms[t])

    def _probe(self, mapping: Mapping) -> Measurement:
        meas = self.sim.measure(self.scen, mapping)
        self.probe_count += 1
        self._absorb(mapping, meas)
        return meas

    def _reprobe(self, rnd: int, base: Mapping, base_meas: Measurement) -> None:
        s = self.scen
        self.probes = []
        self.probe_base = (rnd, base, base_meas)
        for i, t in enumerate(s.tasks):
            for alt in s.placements:
                if alt == base[i]:
                    continue
                if not self.cpu_to_queue and base[i] == CPU and alt == GQ:
                    continue
                trial = base[:i] + (alt,) + base[i + 1:]
                meas = self._probe(trial)
                self.probes.append(Probe("single", f"{t}->{placement_word(s, alt)}", trial, meas))
                if meas.status == "oom" and alt == GN and self.level >= 2 and self.swap:
                    for j, u in enumerate(s.tasks):
                        if j == i or base[j] != GN:
                            continue
                        swapped = list(base)
                        swapped[i], swapped[j] = GN, GQ
                        swapped = tuple(swapped)
                        self.probes.append(Probe("swap", f"{t}<->{u}", swapped,
                                                 self._probe(swapped)))

    def unobserved(self) -> List[str]:
        s = self.scen
        return [f"{t}:{placement_word(s, p)}" for t in s.tasks for p in s.placements
                if not self.windows.get((t, p))]

    def _hint_text(self) -> str:
        text = (f"the last {self.stagnant + 1} rounds used the same mapping without improving "
                "the best E2E, so repeating it adds no new evidence. Consider a placement that "
                "has not been measured yet")
        missing = self.unobserved()
        if missing:
            text += " (unobserved: " + ", ".join(missing) + ")"
        return text + "; a temporary regression is acceptable while exploring."

    # -- prompt fields (Algorithm 1, line 16) ---------------------------------

    def _outcome(self, meas: Measurement) -> str:
        if meas.status == "oom":
            return (f"FAIL out of memory ({meas.peak_vram_mb:.0f}MB>"
                    f"{self.scen.vram_budget_mb:.0f}MB)")
        mem = f" ({meas.peak_vram_mb:.0f}MB)" if self.scen.three_way else ""
        return f"E={meas.e2e_ms:.0f} ms{mem} ok"

    def prompt_sections(self) -> List[str]:
        if not self.enabled:
            return []
        s = self.scen
        lines: List[str] = []
        if self.level >= 1:
            lines.append("## Running averages (measured per-tool latency, ms, from inputs ready "
                         "to finish, contention included; sample count in parentheses):")
            parts = []
            for t in s.tasks:
                seen = [f"{placement_word(s, p)}={statistics.fmean(self.windows[(t, p)]):.0f} "
                        f"({len(self.windows[(t, p)])})"
                        for p in s.placements if self.windows.get((t, p))]
                parts.append(f"{t} " + (", ".join(seen) if seen else "(none yet)"))
            lines.append(" | ".join(parts))
            missing = self.unobserved()
            if missing:
                lines.append("Unobserved tool-device pairs: " + ", ".join(missing))
        if self.history:
            head = "## History"
            if self.best:
                head += (f" (best-ever: round {self.best[2]}, E={self.best[0]:.0f} ms, "
                         f"{fmt_mapping(s, self.best[1])})")
            lines.append(head + ":")
            for rnd, mp, meas in self.history:
                star = "*" if self.best and rnd == self.best[2] else " "
                lines.append(f"{star}{rnd:<3} {fmt_mapping(s, mp):<34} {self._outcome(meas)}")
        if self.level >= 1 and self.probes:
            rnd, base, base_meas = self.probe_base
            kinds = ("single-tool + swap reprobes" if any(p.kind == "swap" for p in self.probes)
                     else "single-tool reprobes")
            lines.append(f"## What-If (base = round {rnd}: {fmt_mapping(s, base)}, "
                         f"{self._outcome(base_meas)}; {kinds}):")
            for p in self.probes:
                if p.kind == "single":
                    lines.append(f"{p.label:<14} {self._outcome(p.meas)}")
                else:
                    lines.append(f"SWAP {p.label:<9} {self._outcome(p.meas)}  "
                                 f"full: {fmt_mapping(s, p.mapping)}")
        if self.hint:
            lines.append("## Exploration hint: " + self.hint)
        return lines


# ===========================================================================
# 10. Scheduler prompt (Sec. 3.3, App. B)
# ===========================================================================

RATIO_GUIDELINE = ("Offload guideline (ratio = cpu_time / gpu_time): ratio<1: CPU-preferred. "
                   "1-7: ambiguous, offload only if cpu_time < slowest GPU bottleneck under "
                   "contention. ratio>=7: keep on GPU.")


def _ms(x: float) -> str:
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}".rstrip("0").rstrip(".")


def build_prompt(scen: Scenario, profiles: Sequence[ToolProfile], mode: str,
                 monitor: Optional[RuntimeMonitor], rnd: int, rounds: int) -> str:
    """The four information streams of the paper's prompt.

    (i) topology and execution semantics, (ii) tool cards -- isolated
    profiles in warm start, workload types only in cold start, (iii) history
    and (iv) monitor what-if data. Binary scenarios carry the ratio-based
    offload guideline; three-way scenarios the structural memory / queue /
    timeline rules (prompt versions 4-5 and 6 of App. B). No target mapping
    is ever suggested.
    """
    three = scen.three_way
    L = ["You are the device scheduler of an agentic AI workflow on a server with one GPU and a "
         "multi-core CPU. Choose an execution device for every tool so that the end-to-end "
         "latency (E2E: from workflow launch until the last tool finishes) is as low as possible.",
         f"Scheduling round {rnd} of {rounds}: the mapping you choose is executed and measured, "
         "and the result is reported back to you next round.",
         "",
         f"## Topology: {TOPOLOGY_TEXT[scen.topology]}."
         + (f" VRAM={scen.vram_budget_mb:.0f} MB." if three else "")]
    if scen.topology == "diamond":
        L.append("A runs first. B and C start when A has finished and run in parallel. "
                 "D starts when both B and C have finished.")
    else:
        L.append("All four tools are independent and are launched at the same time.")
    L.append("## Devices:")
    if three:
        L += ["- gpu_now (GN): runs on the GPU immediately. All gpu_now tools are resident on the "
              "GPU together, so their VRAM must fit the budget, otherwise the round FAILS (out of "
              "memory). gpu_now tools share the GPU and slow each other down.",
              "- gpu_queue (GQ): holds no VRAM while waiting. Starts only after every gpu_now tool "
              "has finished, then runs alone on the GPU, queued tools one at a time, after first "
              "paying its model load time.",
              "- cpu: runs on the CPU from launch, in parallel with the GPU, and uses no VRAM. "
              "CPU tools running at the same time share the CPU."]
    else:
        L += ["- gpu: runs on the GPU. GPU tools running at the same time share it and slow each "
              "other down.",
              "- cpu: runs on the CPU and does not touch the GPU. CPU tools running at the same "
              "time share the CPU."]
    if mode == "warm":
        L.append("## Tool profiles (isolated: each tool measured alone; ms, MB):")
        for t, p in zip(scen.tasks, profiles):
            if three:
                L.append(f"{t} {p.name}: mem={p.vram_mb:.0f} gpu={_ms(p.gpu_ms)} "
                         f"cpu={_ms(p.cpu_ms)} load={_ms(p.load_ms)}")
            else:
                ratio = "inf" if math.isinf(p.ratio) else (f"{p.ratio:.1f}" if p.ratio < 10
                                                           else f"{p.ratio:.0f}")
                L.append(f"{t} {p.name}: gpu={_ms(p.gpu_ms)} cpu={_ms(p.cpu_ms)} ratio={ratio}")
    else:
        L.append("## Tools (described by workload type only; no isolated profiles"
                 + (", measurements so far are below):" if monitor.enabled and monitor.history else
                    "; no measurements yet):"))
        for t, p in zip(scen.tasks, profiles):
            L.append(f"{t}: {p.description} ({p.tag})")
    L.append("## Guidelines (soft; measured evidence below overrides them):")
    if three:
        L += ["1. Memory: the VRAM of the gpu_now tools must sum to at most the budget, or the "
              "round fails.",
              "2. Queue trade-off: gpu_queue costs load + gpu time after the gpu_now phase has "
              "ended; cpu costs the cpu time, overlapped with both GPU phases from launch.",
              "3. Timeline: phase 1 = gpu_now tools on the GPU while cpu tools run on the CPU; "
              "phase 2 = gpu_queue tools one by one. E2E = max(end of phase 2, last cpu tool)."]
    else:
        L.append(RATIO_GUIDELINE)
    L.append("Stability: keep the current best mapping unless measured evidence shows an E2E "
             "improvement of more than 5%.")
    sections = monitor.prompt_sections() if monitor else []
    if sections:
        L.append("")
        L += sections
    words = "gpu_now, gpu_queue, cpu" if three else "gpu, cpu"
    example = ", ".join(f'"{t}": "..."' for t in scen.tasks)
    L += ["",
          f"## Output: reason briefly about each tool, then end with ONE line holding only a JSON "
          f"object that maps every tool to one of: {words}. Format: {{{example}}}"]
    return "\n".join(L)


# ===========================================================================
# 11. LLM client (OpenAI-compatible, e.g. vLLM) and answer parsing
# ===========================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def parse_llm_mapping(text: str, scen: Scenario) -> Optional[Mapping]:
    """The mapping in the answer's last JSON object that names every tool.

    Reasoning blocks are dropped first. Keys may be task ids ("A", "A (clip)")
    or tool names; values any spelling parse_placement accepts. An invalid
    final answer returns None rather than falling back to an earlier draft.
    """
    if not text:
        return None
    body = _THINK_RE.sub(" ", text)
    if re.search(r"</think>", body, re.I):  # unbalanced: keep only what follows
        body = re.split(r"</think>", body, flags=re.I)[-1]
    by_tool = {tool.lower(): t for t, tool in zip(scen.tasks, scen.tools)}
    for candidate in reversed(_JSON_OBJECT_RE.findall(body)):
        obj = None
        for attempt in (candidate, candidate.replace("'", '"')):
            try:
                obj = json.loads(attempt)
                break
            except ValueError:
                continue
        if not isinstance(obj, dict):
            continue
        keyed = {}
        for key, value in obj.items():
            k = str(key).strip()
            m = re.match(r"[A-Za-z0-9]+", k)
            task = None
            if m and m.group(0).upper() in scen.tasks:
                task = m.group(0).upper()
            elif k.lower() in by_tool:
                task = by_tool[k.lower()]
            if task is not None:
                keyed[task] = value
        if not all(t in keyed for t in scen.tasks):
            continue  # not the answer object
        try:
            return tuple(parse_placement(str(keyed[t]), scen) for t in scen.tasks)
        except ValueError:
            return None
    return None


class OpenAICompatClient:
    def __init__(self, base_url: str, model: Optional[str], api_key: str, temperature: float,
                 max_tokens: int, timeout: float, http_retries: int = 4, top_p: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.http_retries = max(1, http_retries)

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        last = "no attempt made"
        for attempt in range(self.http_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"
                if e.code < 500 and e.code != 429:
                    break  # a bad request will not get better
            except (urllib.error.URLError, OSError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
            if attempt + 1 < self.http_retries:
                time.sleep(min(60, 10 * 2 ** attempt))  # 10, 20, 40 s: rides out a busy server
        raise LLMError(f"{method} {url} failed: {last}")

    def resolve_model(self) -> str:
        if not self.model:
            data = self._request("GET", "/models")
            ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
            if not ids:
                raise LLMError(f"{self.base_url}/models lists no model; pass --llm-model")
            self.model = ids[0]
        return self.model

    def chat(self, messages: List[dict]) -> Tuple[str, str, dict, str]:
        """One chat completion: (content, reasoning, usage, finish_reason)."""
        payload = {"model": self.resolve_model(), "messages": messages,
                   "temperature": self.temperature, "top_p": self.top_p,
                   "max_tokens": self.max_tokens}
        data = self._request("POST", "/chat/completions", payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected chat response: {str(data)[:300]}") from None
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return content, reasoning, data.get("usage") or {}, choice.get("finish_reason") or ""


class LLMPolicy:
    """The agentic scheduler's decision layer: one LLM call per round."""

    label = "LLM"
    REPAIR = ("IMPORTANT: end your answer with ONE line holding only a JSON object that assigns "
              "every tool ({tasks}) exactly one of: {words}. Keep the reasoning brief.")

    def __init__(self, client: OpenAICompatClient, retries: int):
        self.client = client
        self.retries = max(0, retries)

    def decide(self, prompt: str, scen: Scenario, fallback: Mapping, monitor: RuntimeMonitor,
               profiles: Sequence[ToolProfile], mode: str) -> Tuple[Mapping, dict]:
        """Ask once; on an unparseable answer re-ask, up to ``retries`` times.

        A re-ask is the original prompt plus a reminder, never the previous
        answer: a reasoning model's answer can run to thousands of tokens, so
        appending it would grow every retry toward the server's context limit.
        Each attempt records its wall time, token usage and finish reason
        ("length" means max_tokens cut the answer off).
        """
        words = "gpu_now, gpu_queue, cpu" if scen.three_way else "gpu, cpu"
        repair = self.REPAIR.format(tasks=", ".join(scen.tasks), words=words)
        attempts = []
        for attempt in range(1 + self.retries):
            text = prompt if attempt == 0 else prompt + "\n\n" + repair
            started = time.time()
            content, reasoning, usage, finish = self.client.chat(
                [{"role": "user", "content": text}])
            seconds = time.time() - started
            if finish == "length" and (reasoning or "</think>" not in content.lower()):
                # Cut off by max_tokens before the reasoning ended: any JSON in
                # the text is a draft, not the answer.
                mapping = None
            else:
                mapping = parse_llm_mapping(content, scen)
                if mapping is None and not content.strip():  # all of it went to reasoning
                    mapping = parse_llm_mapping(reasoning, scen)
            attempts.append({"content": content, "reasoning": reasoning, "usage": usage,
                             "finish_reason": finish, "seconds": round(seconds, 3),
                             "parsed": mapping is not None})
            if mapping is not None:
                return mapping, {"attempts": attempts, "fallback": False}
        return tuple(fallback), {"attempts": attempts, "fallback": True}


class DryRunPolicy:
    """NOT the paper's method: a deterministic stand-in that exercises the loop.

    Round 0 follows the ratio rule (warm) or puts everything on the GPU
    (cold); afterwards it adopts the best measured mapping in the monitor's
    evidence when that beats the best-ever by more than 5%. Use it to test
    the pipeline before an LLM endpoint exists -- never report it as LLM
    results.
    """

    label = "DryRun"

    def decide(self, prompt: str, scen: Scenario, fallback: Mapping, monitor: RuntimeMonitor,
               profiles: Sequence[ToolProfile], mode: str) -> Tuple[Mapping, dict]:
        info = {"policy": "dry-run", "fallback": False}
        if not monitor.enabled or not monitor.history:
            if mode == "warm":
                return tuple(CPU if p.ratio < 1 else GN for p in profiles), info
            return tuple(GN for _ in scen.tasks), info
        options = []
        if monitor.best:
            options.append((monitor.best[0], monitor.best[1]))
        options += [(p.meas.e2e_ms, p.mapping) for p in monitor.probes if p.meas.ok]
        if not options:  # nothing feasible measured yet: evict the largest gpu_now tool
            current = list(monitor.history[-1][1])
            resident = [i for i, p in enumerate(current) if p == GN]
            if resident:
                current[max(resident, key=lambda i: profiles[i].vram_mb)] = CPU
            return tuple(current), info
        e2e, mapping = min(options, key=lambda o: o[0])
        if monitor.best and e2e > monitor.best[0] * 0.95:
            mapping = monitor.best[1]
        return tuple(mapping), info


class EmulatedLLMPolicy:
    """Rule-based emulation of the paper's LLM scheduler -- no model is run.

    It decides from the same information the LLM's prompt carries, applying
    the rules that prompt states (Sec. 3.3, App. B); results must be reported
    as "LLM-emulated", never as the paper's LLM.

    Round 0 (or every round when the monitor is disabled):
      cold   workload type only, the priors the paper's LLM used in Fig. 4:
             compute-/memory-bound -> GPU, compute-light / I/O-bound -> CPU.
      warm, CPU/GPU scenarios  the ratio guideline (ratio = cpu/gpu): < 1 CPU,
             >= 7 GPU, 1-7 ambiguous -- offloaded, most CPU-friendly first, only
             if its CPU time hides behind the GPU bottleneck of its DAG stage,
             i.e. the stage's estimated time max(GPU work, CPU work) drops.
      warm, VRAM scenarios  the structural rules: gpu_now by speedup while the
             budget holds; every other tool takes gpu_queue or cpu, whichever
             the two-phase timeline max(phase 1 + queue, CPU work) prices
             lower (S11-S13 are fan-outs, which this estimate assumes).
    Later rounds, from the monitor's measured evidence:
      * with an exploration hint, the best feasible what-if not yet tried;
      * with nothing feasible measured yet, evict the largest gpu_now tool;
      * otherwise the best measured option, adopted only if it beats the
        best-ever E2E by more than the stability threshold (5%), else the
        best-ever mapping is kept.
    """

    label = "LLM-Emu"
    GPU_TAGS = ("compute-bound", "memory-bound")

    def __init__(self, stability: float = 0.05):
        self.stability = stability

    def decide(self, prompt: str, scen: Scenario, fallback: Mapping, monitor: RuntimeMonitor,
               profiles: Sequence[ToolProfile], mode: str) -> Tuple[Mapping, dict]:
        if not monitor.enabled or not monitor.history:
            mapping, why = self._first_guess(scen, profiles, mode, monitor)
        else:
            mapping, why = self._revise(scen, profiles, monitor)
        return tuple(mapping), {"policy": "emulated-llm", "reason": why, "fallback": False}

    # -- round 0 ----------------------------------------------------------------

    def _first_guess(self, scen, profiles, mode, monitor):
        if mode == "cold":
            mapping = tuple(GN if p.tag in self.GPU_TAGS else CPU for p in profiles)
            return mapping, "cold start: workload-type priors (compute/memory-bound -> GPU)"
        if scen.three_way:
            return self._structural(scen, profiles, monitor)
        return self._ratio_rule(scen, profiles)

    def _ratio_rule(self, scen, profiles):
        level: Dict[str, int] = {}
        for t in scen.tasks:  # DAG stage = longest-path depth
            level[t] = 1 + max((level[p] for p in scen.parents(t)), default=-1)
        prof = dict(zip(scen.tasks, profiles))
        mapping = {t: (CPU if prof[t].ratio < 1 else GN) for t in scen.tasks}
        notes = []
        for stage in sorted(set(level.values())):
            members = [t for t in scen.tasks if level[t] == stage]
            gpu = sum(prof[t].gpu_ms for t in members if mapping[t] == GN)
            cpu = sum(prof[t].cpu_ms for t in members if mapping[t] == CPU)
            ambiguous = sorted((t for t in members if 1 <= prof[t].ratio < 7),
                               key=lambda t: prof[t].ratio)
            for t in ambiguous:
                after = max(gpu - prof[t].gpu_ms, cpu + prof[t].cpu_ms)
                if after < max(gpu, cpu):
                    mapping[t] = CPU
                    gpu, cpu = gpu - prof[t].gpu_ms, cpu + prof[t].cpu_ms
                    notes.append(f"{t} offloaded (ratio {prof[t].ratio:.1f} hides behind GPU)")
        why = "warm start: ratio guideline" + ("; " + "; ".join(notes) if notes else "")
        return tuple(mapping[t] for t in scen.tasks), why

    def _structural(self, scen, profiles, monitor):
        prof = dict(zip(scen.tasks, profiles))
        order = sorted(scen.tasks, key=lambda t: (-prof[t].ratio, scen.tasks.index(t)))
        mapping = {t: CPU for t in scen.tasks}
        for t in order:  # memory rule: gpu_now while the budget holds
            trial = dict(mapping)
            trial[t] = GN
            if monitor.sim.feasible(scen, tuple(trial[x] for x in scen.tasks)):
                mapping = trial

        def timeline(m):  # two-phase rule: E2E = max(phase 1 + queue, CPU work)
            phase1 = sum(prof[t].gpu_ms for t in scen.tasks if m[t] == GN)
            queue = sum(prof[t].load_ms + prof[t].gpu_ms for t in scen.tasks if m[t] == GQ)
            return max(phase1 + queue, sum(prof[t].cpu_ms for t in scen.tasks if m[t] == CPU))

        for t in order:  # queue trade-off for every tool left off gpu_now
            if mapping[t] != GN:
                queued = dict(mapping)
                queued[t] = GQ
                if timeline(queued) < timeline(mapping):
                    mapping = queued
        return (tuple(mapping[t] for t in scen.tasks),
                "warm start: memory budget, queue trade-off and two-phase timeline rules")

    # -- later rounds -----------------------------------------------------------

    def _revise(self, scen, profiles, monitor):
        measured = [(p.meas.e2e_ms, p.mapping, p.label) for p in monitor.probes if p.meas.ok]
        if monitor.hint:
            tried = {mp for _, mp, _ in monitor.history}
            fresh = [c for c in measured if c[1] not in tried]
            if fresh:
                e2e, mapping, label = min(fresh, key=lambda c: c[0])
                return mapping, f"exploration hint: trying untried what-if {label} (E={e2e:.0f} ms)"
        if monitor.best is None:
            if measured:
                e2e, mapping, label = min(measured, key=lambda c: c[0])
                return mapping, f"no feasible round yet: best what-if {label} (E={e2e:.0f} ms)"
            current = list(monitor.history[-1][1])
            resident = [i for i, p in enumerate(current) if p == GN]
            if resident:
                i = max(resident, key=lambda i: profiles[i].vram_mb)
                current[i] = GQ if scen.three_way else CPU
            return tuple(current), "out of memory and no feasible what-if: evict largest gpu_now tool"
        best_e2e, best_mapping, best_round = monitor.best
        if measured:
            e2e, mapping, label = min(measured, key=lambda c: c[0])
            if e2e < best_e2e * (1.0 - self.stability):
                return mapping, (f"what-if {label} gives {e2e:.0f} ms, "
                                 f"{100 * (1 - e2e / best_e2e):.0f}% better than best {best_e2e:.0f} ms")
        return best_mapping, (f"keep best-ever (round {best_round}, {best_e2e:.0f} ms): "
                              f"no measured option better by > {self.stability:.0%}")


# ===========================================================================
# 12. The agentic loop
# ===========================================================================

def run_agentic(sim: ToolDagSimulator, scen: Scenario, oracle: OracleResult, policy, mode: str,
                rounds: int, monitor_kwargs: dict, strategy: str,
                log_fh=None) -> List[RoundRecord]:
    """20 rounds of: prompt -> one mapping -> execute -> monitor (Fig. 2)."""
    profiles = sim.profiles(scen)
    monitor = RuntimeMonitor(scen, sim, **monitor_kwargs)
    records = []
    current: Optional[Mapping] = None
    for r in range(rounds):
        prompt = build_prompt(scen, profiles, mode, monitor, r, rounds)
        fallback = current or all_gpu_mapping(sim, scen)
        mapping, info = policy.decide(prompt, scen, fallback, monitor, profiles, mode)
        probes_before = monitor.probe_count
        meas = sim.measure(scen, mapping)
        monitor.observe(r, mapping, meas)  # reprobes run here, before the next prompt
        note = "fallback: no parseable answer, previous mapping kept" if info.get("fallback") else ""
        rec = make_record(oracle, strategy, r, mapping, meas,
                          probes=monitor.probe_count - probes_before, note=note)
        records.append(rec)
        if log_fh is not None:
            log_fh.write(json.dumps({
                "scenario": scen.sid, "strategy": strategy, "round": r, "prompt": prompt,
                "decision": info, "mapping": fmt_mapping(scen, mapping), "status": meas.status,
                "e2e_ms": meas.e2e_ms, "peak_vram_mb": meas.peak_vram_mb,
                "per_tool_ms": meas.per_tool_ms, "oracle_equivalent": rec.oracle_equiv,
                "match": rec.match_tol, "probes": rec.probes}) + "\n")
            log_fh.flush()
        current = tuple(mapping)
    return records


def monitor_settings(args, scen: Scenario, **overrides) -> dict:
    level = MONITOR_LEVEL[scen.family] if args.monitor_level == "auto" else int(args.monitor_level)
    kw = dict(level=level, window=args.window, hint_after=args.hint_after)
    kw.update(overrides)
    return kw


def make_policy(args):
    if args.emulated_llm:
        return EmulatedLLMPolicy()
    if args.dry_run_llm:
        return DryRunPolicy()
    if not args.llm_base_url:
        raise SystemExit("the agentic scheduler needs an OpenAI-compatible endpoint: pass "
                         "--llm-base-url (e.g. http://localhost:8000/v1 for vLLM), or "
                         "--dry-run-llm to test the plumbing without an LLM")
    client = OpenAICompatClient(args.llm_base_url, args.llm_model, args.llm_api_key,
                                args.temperature, args.max_tokens, args.llm_timeout,
                                top_p=args.top_p)
    return LLMPolicy(client, args.llm_retries)


# ===========================================================================
# 13. Experiment session, summaries and reports
# ===========================================================================

STRATEGY_ORDER = ["All-GPU", "HEFT-CF", "UCB1", "StarPU", "Greedy-Mem", "LLM-Cold", "LLM-Warm",
                  "LLM-Emu-Cold", "LLM-Emu-Warm", "DryRun-Cold", "DryRun-Warm"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _fmt(x: Optional[float], digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


# Strategies that produce one mapping, not one per round.
STATIC_STRATEGIES = {"All-GPU", "HEFT-CF", "Greedy-Mem"}


def parse_fmt_mapping(text: str, scen: Scenario) -> Mapping:
    """Inverse of fmt_mapping: 'A:GN B:GQ C:GN D:CPU' -> (gpu_now, gpu_queue, ...)."""
    words = dict(part.split(":", 1) for part in text.split())
    return tuple(parse_placement(words[t], scen) for t in scen.tasks)


def load_results(out_dir: str, rounds: int):
    """Read a results folder back: complete record groups and ablation rows.

    A (scenario, strategy) group counts as complete when it has all its rounds
    (one for the static strategies); a group cut short by an interruption is
    left out so that it is run again in full. Returns (records, done, ablation
    rows, done ablation (scenario, configuration) pairs).
    """
    records: List[RoundRecord] = []
    done = set()
    path = os.path.join(out_dir, "rounds.csv")
    if os.path.exists(path):
        groups: Dict[Tuple[str, str], List[RoundRecord]] = defaultdict(list)
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                scen = SCENARIOS[row["scenario"]]
                groups[(row["scenario"], row["strategy"])].append(RoundRecord(
                    row["scenario"], row["strategy"], int(row["round"]),
                    parse_fmt_mapping(row["mapping"], scen), row["status"],
                    None if row["e2e_ms"] in ("", "-") else float(row["e2e_ms"]),
                    float(row["peak_vram_mb"]), int(row["match_strict"]), int(row["match_tol"]),
                    row["oracle_equivalent"] == "1", int(row["probes"]), row["note"]))
        for key, recs in groups.items():
            need = 1 if key[1] in STATIC_STRATEGIES else rounds
            if len({r.round for r in recs}) >= need:
                records += sorted(recs, key=lambda r: r.round)
                done.add(key)
    ablation, done_ablation = [], set()
    path = os.path.join(out_dir, "ablation.csv")
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ablation.append({
                    "configuration": row["configuration"], "scenario": row["scenario"],
                    "mode": row["mode"], "best_match_tol": int(row["best_match_tol"]),
                    "best_e2e_ms": float(row["best_e2e_ms"]) if row["best_e2e_ms"] else None,
                    "first_oracle_round": (int(row["first_oracle_round"])
                                           if row["first_oracle_round"] else None),
                    "oom_rounds": int(row["oom_rounds"])})
                done_ablation.add((row["scenario"], row["configuration"]))
    return records, done, ablation, done_ablation


def log_entries(path: str, keep) -> List[str]:
    """Lines of an llm_log.jsonl whose (scenario, strategy) satisfies ``keep``."""
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entry = json.loads(line)
                if keep(entry["scenario"], entry["strategy"]):
                    lines.append(line if line.endswith("\n") else line + "\n")
    return lines


class Session:
    def __init__(self, args, conv):
        self.args = args
        self.conv = conv
        self.sim = make_simulator(args, conv)
        self.scenarios = parse_scenarios(args.scenarios)
        self.out = args.out
        os.makedirs(self.out, exist_ok=True)
        self.oracles: Dict[str, OracleResult] = {}
        self.records: List[RoundRecord] = []
        self.ablation: List[dict] = []
        self.done: set = set()            # (scenario, strategy) groups already complete
        self.done_ablation: set = set()   # (scenario, configuration) ablation rows already done
        self.log_path = os.path.join(self.out, "llm_log.jsonl")
        self.started = time.time()
        if getattr(args, "resume", False):
            self.records, self.done, self.ablation, self.done_ablation = load_results(
                self.out, getattr(args, "rounds", 20))
            # Keep only the log of complete groups; a cut-off group is run again.
            kept = log_entries(self.log_path, lambda s, g: (
                (s, g) in self.done or
                (g.startswith("Ablation: ") and (s, g[len("Ablation: "):]) in self.done_ablation)))
            with open(self.log_path, "w", encoding="utf-8") as fh:
                fh.writelines(kept)
            log(f"[resume] {len(self.done)} complete strategy runs and "
                f"{len(self.done_ablation)} ablation rows kept from {os.path.abspath(self.out)}")
        elif hasattr(args, "llm_base_url") and os.path.exists(self.log_path):
            os.remove(self.log_path)  # a fresh run starts a fresh log (it is appended to)

    def oracle(self, scen: Scenario) -> OracleResult:
        if scen.sid not in self.oracles:
            o = compute_oracle(self.sim, scen, self.args.tie_tol)
            self.oracles[scen.sid] = o
            agree = ("same as paper" if o.best == scen.paper_oracle else
                     "paper's oracle within tie tolerance" if scen.paper_oracle in o.equivalent
                     else "differs from paper " + fmt_mapping(scen, scen.paper_oracle))
            log(f"[oracle] {scen.sid:>3}: {fmt_mapping(scen, o.best)}  E2E={o.best_e2e:.1f} ms "
                f"({agree})")
        return self.oracles[scen.sid]

    def add(self, records: List[RoundRecord]) -> None:
        self.records += records
        if records:
            r = min((x for x in records if x.status == "ok"), key=lambda x: x.e2e_ms, default=None)
            best = f"best {fmt_mapping(SCENARIOS[records[0].scenario], r.mapping)} " \
                   f"E2E={r.e2e_ms:.1f} ms" if r else "no feasible round"
            log(f"[{records[0].strategy}] {records[0].scenario:>3}: {best}; "
                f"oracle-equivalent: {'yes' if any(x.oracle_equiv for x in records) else 'no'}")

    # -- writing --------------------------------------------------------------

    def write(self) -> None:
        a = self.args
        with open(os.path.join(self.out, "oracle.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "mapping", "status", "e2e_ms", "peak_vram_mb", "oracle",
                        "oracle_equivalent", "paper_oracle", "per_tool_ms"])
            for sid, o in self.oracles.items():
                for mp, m in sorted(o.rows, key=lambda r: (r[1].e2e_ms is None,
                                                           r[1].e2e_ms or 0.0)):
                    w.writerow([sid, fmt_mapping(o.scen, mp), m.status, _fmt(m.e2e_ms, 3),
                                f"{m.peak_vram_mb:.0f}", int(mp == o.best),
                                int(mp in o.equivalent), int(mp == o.scen.paper_oracle),
                                json.dumps(m.per_tool_ms)])
        if self.records:
            with open(os.path.join(self.out, "rounds.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["scenario", "strategy", "round", "mapping", "status", "e2e_ms",
                            "peak_vram_mb", "match_strict", "match_tol", "oracle_equivalent",
                            "probes", "note"])
                for r in self.records:
                    w.writerow([r.scenario, r.strategy, r.round,
                                fmt_mapping(SCENARIOS[r.scenario], r.mapping), r.status,
                                _fmt(r.e2e_ms, 3), f"{r.peak_vram_mb:.0f}", r.match_strict,
                                r.match_tol, int(r.oracle_equiv), r.probes, r.note])
        summary = summarize(self.records, self.oracles)
        if summary:
            with open(os.path.join(self.out, "summary.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
                w.writeheader()
                w.writerows(summary)
        if self.ablation:
            with open(os.path.join(self.out, "ablation.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.ablation[0].keys()))
                w.writeheader()
                w.writerows(self.ablation)
        with open(os.path.join(self.out, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(render_markdown(self, summary))
        config = {k: v for k, v in vars(a).items() if k not in ("func", "llm_api_key")}
        config.update({"converter_file": getattr(self.conv, "__file__", None),
                       "simulator_runs": self.sim.runs,
                       "wall_seconds": round(time.time() - self.started, 1)})
        with open(os.path.join(self.out, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, default=str)
        log(f"results written to {os.path.abspath(self.out)}")


def summarize(records: List[RoundRecord], oracles: Dict[str, OracleResult]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[RoundRecord]] = defaultdict(list)
    for r in records:
        groups[(r.scenario, r.strategy)].append(r)
    rows = []
    for (sid, strategy), recs in groups.items():
        o = oracles[sid]
        ok = [r for r in recs if r.status == "ok"]
        best = min(ok, key=lambda r: (r.e2e_ms, r.round)) if ok else None
        steady = [r.e2e_ms for r in ok if best is not None and r.mapping == best.mapping]
        first = next((r.round for r in sorted(recs, key=lambda r: r.round) if r.oracle_equiv),
                     None)
        rows.append({
            "scenario": sid, "strategy": strategy, "rounds": len(recs),
            "best_match_strict": max(r.match_strict for r in recs),
            "best_match_tol": max(r.match_tol for r in recs),
            "first_oracle_round": first,
            "best_mapping": fmt_mapping(o.scen, best.mapping) if best else "",
            "best_e2e_ms": round(best.e2e_ms, 3) if best else None,
            "oracle_e2e_ms": round(o.best_e2e, 3),
            "best_over_oracle": round(best.e2e_ms / o.best_e2e, 4) if best else None,
            "steady_mean_ms": round(statistics.fmean(steady), 3) if steady else None,
            "steady_std_ms": round(statistics.stdev(steady), 3) if len(steady) > 1 else
            (0.0 if steady else None),
            "steady_n": len(steady),
            "oom_rounds": sum(r.status == "oom" for r in recs),
            "probes": sum(r.probes for r in recs),
        })
    order = {s: i for i, s in enumerate(STRATEGY_ORDER)}
    rows.sort(key=lambda r: (int(r["scenario"][1:]), order.get(r["strategy"], 99),
                             r["strategy"]))
    return rows


def render_markdown(session: Session, summary: List[dict]) -> str:
    a = session.args
    sim = session.sim
    by = {(r["scenario"], r["strategy"]): r for r in summary}
    present = {r["strategy"] for r in summary}
    strategies = [s for s in STRATEGY_ORDER if s in present] + sorted(present - set(STRATEGY_ORDER))
    L = ["# TOOLDAG experiment results", "",
         f"Backend `{a.backend}`, profiles `{a.profiles}`"
         + (f" + overrides `{a.profile_overrides}`" if a.profile_overrides else "")
         + f", contention `{a.contention}`, "
         f"slice_ms {a.slice_ms:g}, NPUs {a.num_npus}, tie tolerance {a.tie_tol:.1%}, "
         f"repeats {sim.repeats}, noise cv {a.noise_cv:g}, simulator runs {sim.runs}.", ""]
    if session.oracles:
        L += ["## Brute-force oracle vs the paper's Table 3", "",
              "| ID | Tools (A, B, C, D) | Budget MB | Total MB | Simulator oracle | Oracle E2E ms "
              "| Paper oracle | Paper-oracle E2E ms | Agrees |",
              "|---|---|---|---|---|---|---|---|---|"]
        for sid, o in session.oracles.items():
            s = o.scen
            total = sum(sim.profile(t).vram_mb for t in s.tools)
            pm = o.measurement(s.paper_oracle)
            agrees = ("yes" if o.best == s.paper_oracle else
                      "tie" if s.paper_oracle in o.equivalent else "no")
            L.append(f"| {sid} | {', '.join(s.tools)} | "
                     f"{'-' if s.vram_budget_mb is None else f'{s.vram_budget_mb:.0f}'} | "
                     f"{total:.0f} | {fmt_mapping(s, o.best)} | {o.best_e2e:.1f} | "
                     f"{fmt_mapping(s, s.paper_oracle)} | "
                     f"{f'{pm.e2e_ms:.1f}' if pm.ok else 'OOM'} | {agrees} |")
    binary = [sid for sid in session.oracles if not SCENARIOS[sid].three_way]
    if binary and summary:
        head = "| Strategy | " + " | ".join(binary) + " | 4/4 |"
        sep = "|---|" + "---|" * (len(binary) + 1)
        L += ["", "## Oracle-mapping accuracy, CPU/GPU scenarios (Table 4 analogue)", "",
              "Best per-tool match over all rounds; a mapping within the tie tolerance of the "
              "oracle counts as the oracle.", "", head, sep]
        for strategy in strategies:
            cells = [f"{by[(sid, strategy)]['best_match_tol']}/4" if (sid, strategy) in by
                     else "-" for sid in binary]
            done = [c for c in cells if c != "-"]
            if done:
                L.append(f"| {strategy} | " + " | ".join(cells)
                         + f" | {sum(c == '4/4' for c in done)}/{len(done)} |")
        L += ["", "## Best E2E / oracle E2E (Fig. 5 analogue)", "",
              "| Strategy | " + " | ".join(binary) + " |", "|---|" + "---|" * len(binary),
              "| Oracle E2E ms | " + " | ".join(f"{session.oracles[s].best_e2e:.1f}"
                                                for s in binary) + " |"]
        for strategy in strategies:
            cells = []
            for sid in binary:
                r = by.get((sid, strategy))
                cells.append("-" if r is None else
                             ("OOM" if r["best_over_oracle"] is None
                              else f"{r['best_over_oracle']:.2f}"))
            if any(c != "-" for c in cells):
                L.append(f"| {strategy} | " + " | ".join(cells) + " |")
    memory = [sid for sid in session.oracles if SCENARIOS[sid].three_way]
    agentic = [s for s in strategies if s.startswith(("LLM-", "DryRun-"))]
    if memory and summary:
        L += ["", "## Three-way placement under VRAM limits (Table 5 analogue)", "",
              "| Scenario | Budget | Total MB | All-GPU E2E | Greedy E2E | Oracle-map E2E | "
              "Strategy | Best-map E2E | Match | First 4/4 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sid in memory:
            o = session.oracles[sid]
            s = o.scen
            total = sum(sim.profile(t).vram_mb for t in s.tools)

            def e2e_of(name: str) -> str:
                r = by.get((sid, name))
                if r is None:
                    return "-"
                return "OOM" if r["best_e2e_ms"] is None else f"{r['best_e2e_ms']:.1f}"

            for strategy in agentic or ["-"]:
                r = by.get((sid, strategy))
                L.append(f"| {sid} | {s.vram_budget_mb:.0f} | {total:.0f} | {e2e_of('All-GPU')} | "
                         f"{e2e_of('Greedy-Mem')} | {o.best_e2e:.1f} | {strategy} | "
                         f"{'-' if r is None else e2e_of(strategy)} | "
                         f"{'-' if r is None else str(r['best_match_tol']) + '/4'} | "
                         f"{'-' if r is None or r['first_oracle_round'] is None else 'R' + str(r['first_oracle_round'])} |")
    adaptive = [s for s in strategies if s in ("UCB1", "StarPU") or s in agentic]
    contention = [sid for sid in session.oracles if SCENARIOS[sid].family == "contention"]
    if adaptive and contention and summary:
        L += ["", "## Steady-state E2E over rounds using the best-found mapping (Table 7 analogue)",
              "", "| Scenario | Oracle | " + " | ".join(adaptive) + " |",
              "|---|---|" + "---|" * len(adaptive)]
        for sid in contention:
            cells = []
            for strategy in adaptive:
                r = by.get((sid, strategy))
                cells.append("-" if r is None or r["steady_mean_ms"] is None else
                             f"{r['steady_mean_ms']:.1f} +/- {r['steady_std_ms']:.1f} "
                             f"({r['steady_n']})")
            L.append(f"| {sid} | {session.oracles[sid].best_e2e:.1f} | " + " | ".join(cells) + " |")
    if session.ablation:
        L += ["", "## Monitor ablation (Table 6 analogue)", "",
              "| Configuration | Scenario | Mode | Best match | Best E2E ms | First 4/4 |",
              "|---|---|---|---|---|---|"]
        for row in session.ablation:
            best = "OOM" if row["best_e2e_ms"] is None else f"{row['best_e2e_ms']:.1f}"
            first = "-" if row["first_oracle_round"] is None else f"R{row['first_oracle_round']}"
            L.append(f"| {row['configuration']} | {row['scenario']} | {row['mode']} | "
                     f"{row['best_match_tol']}/4 | {best} | {first} |")
    stats = llm_call_stats(session.log_path)
    if stats:
        L += ["", "## Scheduler LLM calls (measured on the LLM server)", "",
              "One decision = one round; it may take several attempts when an answer has no "
              "valid JSON mapping. Not part of any E2E above (the paper keeps the scheduler LLM "
              "on its own GPU, Sec. 4.1).", "",
              "| Strategy | Decisions | Attempts | Mean s / decision | Median s | Max s | "
              "Mean completion tokens | Cut off by max_tokens | Fallbacks |",
              "|---|---|---|---|---|---|---|---|---|"]
        for strategy, s in stats.items():
            L.append(f"| {strategy} | {s['decisions']} | {s['attempts']} | {s['mean_s']:.1f} | "
                     f"{s['median_s']:.1f} | {s['max_s']:.1f} | {s['mean_tokens']:.0f} | "
                     f"{s['length_cut']} | {s['fallbacks']} |")
    L += ["", "Per-round data: rounds.csv; every mapping: oracle.csv; LLM prompts and answers: "
          "llm_log.jsonl.", ""]
    return "\n".join(L)


def llm_call_stats(path: str) -> Dict[str, dict]:
    """Per strategy: time, tokens and failures of the real LLM calls in llm_log.jsonl.

    Only real-LLM decisions carry timed attempts; emulated and dry-run
    decisions are skipped.
    """
    per: Dict[str, List[dict]] = defaultdict(list)
    for line in log_entries(path, lambda s, g: True):
        entry = json.loads(line)
        attempts = entry.get("decision", {}).get("attempts") or []
        if attempts and all("seconds" in x for x in attempts):
            per[entry["strategy"]].append(entry)
    out = {}
    for strategy, entries in per.items():
        seconds = [sum(x["seconds"] for x in e["decision"]["attempts"]) for e in entries]
        attempts = [x for e in entries for x in e["decision"]["attempts"]]
        tokens = [x["usage"].get("completion_tokens") for x in attempts
                  if isinstance(x.get("usage"), dict) and x["usage"].get("completion_tokens")]
        out[strategy] = {
            "decisions": len(entries), "attempts": len(attempts),
            "mean_s": statistics.fmean(seconds), "median_s": statistics.median(seconds),
            "max_s": max(seconds),
            "mean_tokens": statistics.fmean(tokens) if tokens else 0.0,
            "length_cut": sum(x.get("finish_reason") == "length" for x in attempts),
            "fallbacks": sum(bool(e["decision"].get("fallback")) for e in entries)}
    return out


# ===========================================================================
# 14. Subcommands
# ===========================================================================

def _in_session(args, conv, body) -> int:
    """Run ``body(session)``; results are written even if it fails part-way.

    A long run that loses its LLM endpoint or simulator at scenario 12 (or
    is interrupted) keeps everything measured so far; the error still
    propagates, so the exit code reports the failure.
    """
    session = Session(args, conv)
    try:
        body(session)
    finally:
        try:
            session.write()
        except Exception as e:  # never mask the original error
            log(f"could not write results: {type(e).__name__}: {e}")
        session.sim.cleanup()
    return 0


def cmd_oracle(args, conv) -> int:
    return _in_session(args, conv, lambda s: [s.oracle(scen) for scen in s.scenarios])


def _run_baselines(session: Session) -> None:
    a, sim = session.args, session.sim

    def todo(sid: str, name: str) -> bool:
        if (sid, name) in session.done:
            log(f"[resume] {name} {sid}: already complete, skipped")
            return False
        return True

    for scen in session.scenarios:
        o = session.oracle(scen)
        if todo(scen.sid, "All-GPU"):
            mp = all_gpu_mapping(sim, scen)
            session.add([make_record(o, "All-GPU", 0, mp, sim.measure(scen, mp))])
        if scen.three_way:
            # HEFT-CF, UCB1 and StarPU have no three-way form (Sec. 4.1).
            if todo(scen.sid, "Greedy-Mem"):
                mp = greedy_memory_mapping(sim, scen)
                session.add([make_record(o, "Greedy-Mem", 0, mp, sim.measure(scen, mp))])
            continue
        if todo(scen.sid, "HEFT-CF"):
            mp = heft_cf_mapping(sim, scen, a.heft_cf)
            session.add([make_record(o, "HEFT-CF", 0, mp, sim.measure(scen, mp))])
        if todo(scen.sid, "UCB1"):
            session.add(run_ucb1(sim, scen, o, a.rounds,
                                 random.Random(a.seed * 1000 + int(scen.sid[1:]))))
        if todo(scen.sid, "StarPU"):
            session.add(run_starpu(sim, scen, o, a.rounds))


def _run_agentic(session: Session) -> None:
    a, sim = session.args, session.sim
    policy = make_policy(a)
    modes = [m.strip().lower() for m in a.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("cold", "warm"):
            raise SystemExit(f"unknown mode '{m}' (cold, warm)")
    with open(session.log_path, "a", encoding="utf-8") as fh:
        for scen in session.scenarios:
            o = session.oracle(scen)
            for mode in modes:
                name = f"{policy.label}-{mode.title()}"
                if (scen.sid, name) in session.done:
                    log(f"[resume] {name} {scen.sid}: already complete, skipped")
                    continue
                session.add(run_agentic(sim, scen, o, policy, mode, a.rounds,
                                        monitor_settings(a, scen), name, fh))


ABLATIONS = [
    ("Full system", {}),
    ("w/o cpu->queue reprobe", {"cpu_to_queue": False}),
    ("w/o swap reprobe", {"swap": False}),
    ("w/o exploration hints", {"hints": False}),
    ("LLM only (no monitor)", {"enabled": False}),
]


def _run_ablation(session: Session) -> None:
    a, sim = session.args, session.sim
    scen = SCENARIOS[a.ablation_scenario.upper()]
    o = session.oracle(scen)
    policy = make_policy(a)
    with open(session.log_path, "a", encoding="utf-8") as fh:
        for label, overrides in ABLATIONS:
            name = f"Ablation: {label}"
            if (scen.sid, label) in session.done_ablation:
                log(f"[resume] ablation '{label}' {scen.sid}: already complete, skipped")
                continue
            recs = run_agentic(sim, scen, o, policy, a.ablation_mode, a.rounds,
                               monitor_settings(a, scen, **overrides), name, fh)
            ok = [r for r in recs if r.status == "ok"]
            best = min(ok, key=lambda r: r.e2e_ms) if ok else None
            first = next((r.round for r in recs if r.oracle_equiv), None)
            session.ablation.append({
                "configuration": label, "scenario": scen.sid, "mode": a.ablation_mode,
                "best_match_tol": max(r.match_tol for r in recs),
                "best_e2e_ms": round(best.e2e_ms, 3) if best else None,
                "first_oracle_round": first, "oom_rounds": sum(r.status == "oom" for r in recs)})
            log(f"[ablation] {label}: best match {session.ablation[-1]['best_match_tol']}/4, "
                f"best E2E {'OOM' if best is None else f'{best.e2e_ms:.1f} ms'}")


def cmd_baselines(args, conv) -> int:
    return _in_session(args, conv, _run_baselines)


def cmd_agentic(args, conv) -> int:
    return _in_session(args, conv, _run_agentic)


def cmd_ablation(args, conv) -> int:
    args.scenarios = args.ablation_scenario
    return _in_session(args, conv, _run_ablation)


def cmd_all(args, conv) -> int:
    def body(session: Session) -> None:
        _run_baselines(session)
        if args.llm_base_url or args.dry_run_llm or args.emulated_llm:
            _run_agentic(session)
            if args.with_ablation:
                _run_ablation(session)
        else:
            log("[all] no --llm-base-url / --dry-run-llm: skipping the agentic scheduler")
    return _in_session(args, conv, body)


def cmd_merge(args, conv) -> int:
    """Combine the result folders of separate runs (e.g. parallel ones) into one report.

    Only complete strategy runs are taken; the same run in two folders is an
    error. Oracles are recomputed (deterministic), logs are concatenated.
    """
    dirs = [d.strip() for d in args.from_dirs.split(",") if d.strip()]
    if os.path.abspath(args.out) in {os.path.abspath(d) for d in dirs}:
        raise SystemExit("--out must be a new folder, not one of the --from folders")
    records, done, ablation, done_ablation, logs = [], set(), [], set(), []
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"--from: no such folder '{d}'")
        recs, dn, abl, dab = load_results(d, args.rounds)
        clash = sorted(dn & done) + sorted(dab & done_ablation)
        if clash:
            raise SystemExit(f"{d} repeats runs already taken from another folder: {clash[:3]}")
        records += recs
        done |= dn
        ablation += abl
        done_ablation |= dab
        logs += log_entries(os.path.join(d, "llm_log.jsonl"), lambda s, g, dn=dn, dab=dab: (
            (s, g) in dn or (g.startswith("Ablation: ") and (s, g[len("Ablation: "):]) in dab)))
        log(f"[merge] {d}: {len(dn)} strategy runs, {len(dab)} ablation rows")
    if not records and not ablation:
        raise SystemExit("nothing complete to merge")
    sids = sorted({r.scenario for r in records} | {row["scenario"] for row in ablation},
                  key=lambda s: int(s[1:]))
    args.scenarios = ",".join(sids)
    session = Session(args, conv)
    try:
        session.records, session.done = records, done
        session.ablation, session.done_ablation = ablation, done_ablation
        with open(session.log_path, "w", encoding="utf-8") as fh:
            fh.writelines(logs)
        for sid in sids:
            session.oracle(SCENARIOS[sid])
        session.write()
    finally:
        session.sim.cleanup()
    return 0


def cmd_llmcheck(args, conv) -> int:
    """One real scheduling call before a long run: endpoint, model, time, parseable answer."""
    if not args.llm_base_url:
        raise SystemExit("llmcheck needs --llm-base-url (e.g. http://127.0.0.1:8000/v1)")
    scen = SCENARIOS[args.check_scenario.upper()]
    sim = make_simulator(args, conv, backend=ModelBackend(conv))
    try:
        client = OpenAICompatClient(args.llm_base_url, args.llm_model, args.llm_api_key,
                                    args.temperature, args.max_tokens, args.llm_timeout,
                                    top_p=args.top_p)
        policy = LLMPolicy(client, args.llm_retries)
        print(f"endpoint: {args.llm_base_url}   model: {client.resolve_model()}")
        print(f"sampling: temperature={args.temperature} top_p={args.top_p} "
              f"max_tokens={args.max_tokens}")
        profiles = sim.profiles(scen)
        monitor = RuntimeMonitor(scen, sim, **monitor_settings(args, scen))
        prompt = build_prompt(scen, profiles, args.check_mode, monitor, 0, args.rounds)
        print(f"asking for the round-0 mapping of {scen.sid} ({args.check_mode} start) ...",
              flush=True)
        mapping, info = policy.decide(prompt, scen, all_gpu_mapping(sim, scen), monitor,
                                      profiles, args.check_mode)
        for i, att in enumerate(info["attempts"], 1):
            print(f"attempt {i}: {att['seconds']:.1f} s, finish_reason="
                  f"{att['finish_reason'] or '-'}, usage={att['usage']}, parsed={att['parsed']}")
        last = info["attempts"][-1]
        print("answer, last 600 characters:\n" + (last["content"] or last["reasoning"]).strip()[-600:])
        if info["fallback"]:
            print("FAIL: no valid JSON mapping in any attempt. If finish_reason is 'length', "
                  "raise --max-tokens (and the server's --max-model-len).")
            return 1
        meas = sim.measure(scen, mapping)
        result = (f"E2E {meas.e2e_ms:.1f} ms" if meas.ok else "out of memory")
        print(f"PASS: the answer holds a valid mapping, {fmt_mapping(scen, mapping)} "
              f"(round 0 alone: {result} on the reference model; the oracle is "
              f"{fmt_mapping(scen, compute_oracle(sim, scen, args.tie_tol).best)}, which the "
              f"scheduler is meant to reach over the rounds)")
        return 0
    finally:
        sim.cleanup()


def tooldag_text(sim: ToolDagSimulator, scen: Scenario, mapping: Mapping) -> str:
    tasks = sim.tool_tasks(scen, mapping)
    header = (f"TOOLDAG vram_budget_mb: {scen.vram_budget_mb or 0:g} "
              f"time_units_per_ms: {sim.time_units_per_ms:g} contention: {sim.contention} "
              f"slice_ms: {sim.slice_ms:g} cpu_flag: {sim.cpu_flag}")
    rows = [" ".join(t.to_cols()) for t in tasks]
    return "\n".join([header, str(len(tasks)),
                      "task tool placement gpu_ms cpu_ms load_ms vram_mb deps"] + rows) + "\n"


def cmd_export(args, conv) -> int:
    scen = SCENARIOS[args.scenario.upper()]
    mapping = parse_mapping_arg(args.mapping, scen)
    sim = make_simulator(args, conv, backend=ModelBackend(conv))
    try:
        text = tooldag_text(sim, scen, mapping)
        if args.trace:
            with open(args.trace, "w", encoding="utf-8") as fh:
                fh.write(text)
            log(f"wrote {args.trace}")
        else:
            sys.stdout.write(text)
        if args.convert:
            if not args.trace:
                raise SystemExit("--convert needs --trace (the converter reads a file)")
            converter = conv.LLMConverter(args.trace, args.convert, args.num_npus)
            try:
                summary = converter.convert()
            except conv.VramBudgetExceeded as e:
                raise SystemExit(f"not converted: {e}")
            log(f"converted -> {', '.join(summary['files'])} "
                f"(peak VRAM {summary['peak_vram_mb']:g} MB)")
    finally:
        sim.cleanup()
    return 0


# -- crosscheck -------------------------------------------------------------

def cmd_crosscheck(args, conv) -> int:
    """Compare ASTRA-Sim with the reference model on every feasible mapping.

    If they agree, the simulator honours is_cpu_op, serializes each NPU's
    CPU and GPU work and uses the configured time units. Otherwise the
    pattern of disagreement points at the cause.
    """
    if args.backend != "astra":
        raise SystemExit("crosscheck compares ASTRA-Sim against the model: use --backend astra")
    args.noise_cv = 0.0
    astra = make_simulator(args, conv)
    models = {sem: make_simulator(args, conv, backend=ModelBackend(conv, sem))
              for sem in MODEL_SEMANTICS}
    rows = []
    per_tool_checked = per_tool_bad = 0
    try:
        for scen in parse_scenarios(args.scenarios):
            for mp in enumerate_mappings(scen):
                if not astra.feasible(scen, mp):
                    continue
                ma = astra.measure(scen, mp)
                mm = {sem: models[sem].measure(scen, mp) for sem in MODEL_SEMANTICS}
                rows.append((scen, mp, ma.e2e_ms, {s: m.e2e_ms for s, m in mm.items()}))
                if ma.per_tool_ms:
                    for t in scen.tasks:
                        per_tool_checked += 1
                        ref = mm["lanes"].per_tool_ms[t]
                        if abs(ma.per_tool_ms[t] - ref) > max(1e-3, 1e-3 * ref):
                            per_tool_bad += 1
            log(f"[crosscheck] {scen.sid} done")
    finally:
        astra.cleanup()
        for m in models.values():
            m.cleanup()
    if not rows:
        raise SystemExit("no feasible mapping was simulated")

    def worst(sem: str) -> float:
        return max(abs(a / v[sem] - 1.0) for _, _, a, v in rows)

    def ratio_spread(sem: str) -> Tuple[float, float]:
        ratios = [a / v[sem] for _, _, a, v in rows]
        mean = statistics.fmean(ratios)
        return mean, (max(ratios) - min(ratios)) / mean

    tol = args.crosscheck_tol
    print(f"mappings compared: {len(rows)}")
    for sem in MODEL_SEMANTICS:
        print(f"  ASTRA-Sim vs model[{sem:<9}]: worst relative difference {worst(sem):.2%}")
    if per_tool_checked:
        print(f"  per-tool latencies: {per_tool_checked - per_tool_bad}/{per_tool_checked} agree")
    else:
        print("  per-tool latencies: none parsed (enable ASTRA-Sim's per-node trace log and set "
              "--callback-regex; the monitor and StarPU need them)")
    if worst("lanes") <= tol:
        print("VERDICT: PASS -- ASTRA-Sim reproduces the reference semantics (CPU and GPU "
              "resources per NPU, one op in flight each; time units consistent).")
        return 0
    for sem, cause in (("lanes", None),
                       ("gpu_only", "is_cpu_op is not honoured: every node runs on the GPU "
                                    "resource. Check how et_feeder_node.cpp reads is_cpu_op; try "
                                    "--cpu-flag int32."),
                       ("unbounded", "compute nodes run concurrently without per-NPU resource "
                                     "limits. Re-run with --contention serialize.")):
        mean, spread = ratio_spread(sem)
        if spread <= tol:
            unit = ("time units: ASTRA-Sim reports {:.6g}x the model. Multiply --sim-ticks-per-ms "
                    "by {:.6g} (e.g. 1e6 -> 1e3 if its tick is a microsecond, or adjust "
                    "--time-units-per-ms if it reads duration_micros as raw ticks).").format(
                        mean, mean)
            print("VERDICT: MISMATCH -- " + (cause + " Also " + unit if cause and abs(mean - 1)
                                              > tol else cause or unit))
            return 1
        if cause and worst(sem) <= tol:
            print("VERDICT: MISMATCH -- " + cause)
            return 1
    print("VERDICT: MISMATCH -- no reference semantics explains ASTRA-Sim's timings. Keep the ETs "
          "(--keep-et), inspect them and the simulator log; worst cases:")
    rows.sort(key=lambda r: -abs(r[2] / r[3]["lanes"] - 1.0))
    for scen, mp, a, v in rows[:10]:
        print(f"  {scen.sid:>3} {fmt_mapping(scen, mp):<34} ASTRA-Sim {a:.3f} ms, "
              f"model {v['lanes']:.3f} ms")
    return 1


# -- selftest ---------------------------------------------------------------

# Simulator oracle and its E2E (ms) under stock ASTRA-Sim semantics with the
# "paper" profiles, worked out independently of this code.
SELFTEST_EXPECTED = {
    "S1": ((GN, GN, CPU, CPU), 612), "S2": ((GN, CPU, GN, GN), 27),
    "S3": ((GN, GN, CPU, CPU), 176), "S4": ((GN, GN, CPU, GN), 30),
    "S5": ((GN, GN, CPU, GN), 739), "S6": ((GN, GN, GN, CPU), 739),
    "S7": ((GN, GN, GN, GN), 806), "S8": ((GN, GN, CPU, GN), 739),
    "S9": ((GN, GN, GN, CPU), 521), "S10": ((GN, GN, GN, CPU), 613),
    "S11": ((GN, GN, GQ, GN), 1287), "S12": ((GN, GN, GN, CPU), 613),
    "S13": ((GN, GQ, GN, CPU), 917),
}


def cmd_selftest(args, conv) -> int:
    failures: List[str] = []

    def check(cond: bool, what: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + what)
        if not cond:
            failures.append(what)

    workdir = tempfile.mkdtemp(prefix="tooldag_selftest_")
    base = dict(profile_mode="paper", num_npus=2, time_units_per_ms=1000.0,
                contention="simulator", slice_ms=0.0, cpu_flag="bool", workdir=workdir)
    sim = ToolDagSimulator(conv, ModelBackend(conv), **base)
    try:
        s13 = SCENARIOS["S13"]
        # Memory: S13 round 0 of the paper (Fig. 6) must be infeasible.
        r0 = sim.measure(s13, (GN, GN, GN, CPU))
        check(r0.status == "oom" and abs(r0.peak_vram_mb - 1584) < 1e-6,
              f"S13 A,B,C on gpu_now is OOM at 1584 MB > 1100 MB (got {r0.status}, "
              f"{r0.peak_vram_mb:g} MB)")
        plan = sim.plan(s13, (GN, GQ, GN, CPU))
        check(plan.feasible and abs(plan.peak_vram_mb - 1098) < 1e-6,
              f"S13 oracle mapping fits: A+C = 1098 MB (got {plan.peak_vram_mb:g})")
        check(sorted(plan.aug_parents["B"]) == ["A", "C"],
              f"queued B waits for the gpu_now tools A and C (got {plan.aug_parents['B']})")
        try:
            sim.conv.LLMConverter("x", os.path.join(workdir, "oom"), 1).convert_tool_dag(
                sim.tool_tasks(s13, (GN, GN, GN, CPU)), vram_budget_mb=1100)
            check(False, "convert_tool_dag raises VramBudgetExceeded for an OOM mapping")
        except conv.VramBudgetExceeded as e:
            check(e.tasks == ["A", "B", "C"] and not os.path.exists(
                os.path.join(workdir, "oom.0.et")),
                  "convert_tool_dag raises VramBudgetExceeded before writing any file")

        # ET structure of the S13 oracle mapping.
        prefix = os.path.join(workdir, "s13")
        summary = conv.LLMConverter("s13", prefix, 2).convert_tool_dag(
            sim.tool_tasks(s13, (GN, GQ, GN, CPU)), vram_budget_mb=1100)
        check(len(summary["files"]) == 2 and all(os.path.exists(f) for f in summary["files"]),
              "one ET per NPU is written")
        _, nodes = read_et(summary["files"][0], conv)
        _, nodes1 = read_et(summary["files"][1], conv)
        check([n.SerializeToString() for n in nodes] == [n.SerializeToString() for n in nodes1],
              "every NPU file carries the same graph")
        got = [(n.name, int(n.duration_micros), node_is_cpu(n), sorted(n.data_deps)) for n in nodes]
        want = [("COMP_NODE_TOOL_A_clip_x12_GPU", 147000, False, []),
                ("COMP_NODE_TOOL_C_whisper_x4_GPU", 136000, False, []),
                ("COMP_NODE_TOOL_D_groupby_x60_CPU", 520000, True, []),
                ("COMP_NODE_TOOL_B_bert_base_x12_LOAD", 481000, False, [0, 1]),
                ("COMP_NODE_TOOL_B_bert_base_x12_GPU", 153000, False, [3])]
        check(got == want, f"S13 oracle ET nodes, durations, is_cpu_op and deps (got {got})")
        check(all(n.type == conv.COMP_NODE for n in nodes), "all TOOLDAG nodes are COMP_NODEs")

        # Text paths: convert() and convert_rows() give the same nodes.
        text = tooldag_text(sim, s13, (GN, GQ, GN, CPU))
        trace = os.path.join(workdir, "s13.txt")
        with open(trace, "w", encoding="utf-8") as fh:
            fh.write(text)
        conv.LLMConverter(trace, os.path.join(workdir, "s13txt"), 1).convert()
        _, nodes_txt = read_et(os.path.join(workdir, "s13txt.0.et"), conv)
        check([n.SerializeToString() for n in nodes_txt] == [n.SerializeToString() for n in nodes],
              "convert() on the TOOLDAG text trace gives the same nodes")
        lines = text.splitlines()
        conv.LLMConverter("rows", os.path.join(workdir, "s13rows"), 1).convert_rows(
            lines[0], [ln.split() for ln in lines[3:]])
        _, nodes_rows = read_et(os.path.join(workdir, "s13rows.0.et"), conv)
        check([n.SerializeToString() for n in nodes_rows] == [n.SerializeToString() for n in nodes],
              "convert_rows() gives the same nodes")

        # Malformed traces are rejected.
        def rejects(fn, what: str) -> None:
            try:
                fn()
                check(False, what)
            except ValueError:
                check(True, what)

        c = conv.LLMConverter("bad", os.path.join(workdir, "bad"), 1)
        rejects(lambda: c.tool_dag_options(c.parse_header_fields(["vram_budget:", "1100"])),
                "unknown TOOLDAG header key is rejected")
        rejects(lambda: c.parse_header_fields(["vram_budget_mb:"]),
                "dangling TOOLDAG header token is rejected")
        rejects(lambda: conv.ToolTask("A clip_x12 gpu_later 1 2 3 4 -"),
                "unknown placement is rejected")
        rejects(lambda: c.analyze_tool_dag([conv.ToolTask("A t gpu 1 1 0 0 B"),
                                            conv.ToolTask("B t cpu 1 1 0 0 A")]),
                "dependency cycle is rejected")
        rejects(lambda: c.analyze_tool_dag([conv.ToolTask("A t gpu 1 1 0 0 Z")]),
                "unknown dependency is rejected")

        # Timings under stock semantics.
        s5 = SCENARIOS["S5"]
        m = sim.measure(s5, (GN, GN, GN, GN))
        check(abs(m.e2e_ms - 928) < 1e-6 and m.per_tool_ms == {"A": 370, "B": 506, "C": 783,
                                                               "D": 928},
              f"S5 all-GPU serializes on the GPU: E2E 928, per tool 370/506/783/928 "
              f"(got {m.e2e_ms}, {m.per_tool_ms})")
        r1 = sim.measure(s13, (GN, GN, CPU, CPU))
        check(r1.status == "oom" and abs(r1.peak_vram_mb - 1142) < 1e-6,
              f"with Table 1 VRAM the paper's S13 round-1 mapping A:GN B:GN is OOM "
              f"(656+486 = 1142 MB; got {r1.status}, {r1.peak_vram_mb:g} MB)")
        for mp, want_ms in (((GN, CPU, GN, CPU), 12661), ((GQ, GN, GN, CPU), 3380)):
            got_ms = sim.measure(s13, mp).e2e_ms
            check(got_ms is not None and abs(got_ms - want_ms) < 1e-6,
                  f"S13 {fmt_mapping(s13, mp)} E2E {want_ms} ms (got {got_ms})")
        fig7 = ToolDagSimulator(conv, ModelBackend(conv),
                                profile_overrides=load_profile_overrides("fig7-s13"), **base)
        for mp, want_ms, want_mb in (((GN, GN, CPU, CPU), 2676, 1090),
                                     ((GN, GN, GQ, CPU), 1648, 1090),
                                     ((GN, GQ, GN, CPU), 681, 1044)):
            mm = fig7.measure(s13, mp)
            check(mm.ok and abs(mm.e2e_ms - want_ms) < 1e-6 and abs(mm.peak_vram_mb - want_mb) < 1e-6,
                  f"fig7-s13: S13 {fmt_mapping(s13, mp)} E2E {want_ms} ms at {want_mb} MB "
                  f"(got {mm.status}, {mm.e2e_ms}, {mm.peak_vram_mb:g} MB)")
        check(compute_oracle(fig7, s13, 0.0).best == (GN, GQ, GN, CPU),
              "fig7-s13: the S13 oracle mapping is unchanged")
        override_file = os.path.join(workdir, "overrides.json")
        with open(override_file, "w", encoding="utf-8") as fh:
            json.dump({"ocr_x2": {"cpu_ms": 1000}}, fh)
        ov = tool_profile("ocr_x2", "paper", load_profile_overrides(override_file))
        check(ov.cpu_ms == 1000 and ov.gpu_ms == 370 and ov.vram_mb == 248,
              "a profile-override file replaces only the fields it gives")
        for sid, (want_mp, want_ms) in SELFTEST_EXPECTED.items():
            o = compute_oracle(sim, SCENARIOS[sid], 0.0)
            check(o.best == want_mp and abs(o.best_e2e - want_ms) < 1e-6,
                  f"{sid} oracle {fmt_mapping(SCENARIOS[sid], want_mp)} = {want_ms} ms "
                  f"(got {fmt_mapping(SCENARIOS[sid], o.best)} = {o.best_e2e:g})")

        # Options that must not change E2E on the paper's topologies.
        variants = {
            "contention=serialize": ToolDagSimulator(conv, ModelBackend(conv),
                                                     **{**base, "contention": "serialize"}),
            "contention=serialize on an unbounded simulator": ToolDagSimulator(
                conv, ModelBackend(conv, "unbounded"), **{**base, "contention": "serialize"}),
            "slice_ms=5 (processor sharing)": ToolDagSimulator(conv, ModelBackend(conv),
                                                               **{**base, "slice_ms": 5.0}),
            "cpu_flag=int32": ToolDagSimulator(conv, ModelBackend(conv),
                                               **{**base, "cpu_flag": "int32"}),
        }
        for label, other in variants.items():
            diffs = []
            for scen in SCENARIOS.values():
                for mp in enumerate_mappings(scen):
                    a_, b_ = sim.measure(scen, mp), other.measure(scen, mp)
                    if a_.status != b_.status or (a_.ok and abs(a_.e2e_ms - b_.e2e_ms) > 1e-6):
                        diffs.append(f"{scen.sid} {fmt_mapping(scen, mp)}")
            check(not diffs, f"{label}: same E2E for all {sum(len(enumerate_mappings(s)) for s in SCENARIOS.values())} "
                             f"mappings" + (f" (differs: {diffs[:3]})" if diffs else ""))
        sliced = variants["slice_ms=5 (processor sharing)"].measure(s5, (GN, GN, GN, GN))
        ideal = {"A": 928, "B": 544, "C": 835, "D": 571}  # processor sharing of 370/136/277/145
        check(all(abs(sliced.per_tool_ms[t] - ideal[t]) <= 5.0 for t in ideal),
              f"slicing makes S5's GPU tools share the GPU like processor sharing, within one "
              f"slice of {ideal} (got {sliced.per_tool_ms})")
        unbounded = ToolDagSimulator(conv, ModelBackend(conv, "unbounded"), **base)
        check(abs(unbounded.measure(s5, (GN, GN, GN, GN)).e2e_ms - 370) < 1e-6,
              "without resource limits S5 all-GPU would take only the longest tool (370 ms) -- "
              "what --contention serialize guards against")

        # Monitor (Algorithm 1) on the paper's S13 storyboard (Fig. 6/7 values).
        mon = RuntimeMonitor(s13, fig7, level=2)
        mon.observe(0, (GN, GN, CPU, CPU), fig7.measure(s13, (GN, GN, CPU, CPU)))
        labels = [p.label for p in mon.probes]
        check("C->gpu_queue" in labels and "C->gpu_now" in labels and "A->cpu" in labels,
              "monitor reprobes every other device of every tool")
        mon2 = RuntimeMonitor(s13, fig7, level=2)
        mon2.observe(0, (GN, GN, GQ, CPU), fig7.measure(s13, (GN, GN, GQ, CPU)))
        swaps = {p.label: p for p in mon2.probes if p.kind == "swap"}
        swap = swaps.get("C<->B")
        check(swap is not None and swap.mapping == (GN, GQ, GN, CPU) and swap.meas.ok
              and abs(swap.meas.e2e_ms - 681) < 1e-6 and abs(swap.meas.peak_vram_mb - 1044) < 1e-6,
              "C->gpu_now OOM triggers the C<->B swap reprobe, which finds the oracle at "
              "1044 MB (Fig. 7)")
        check(fig7.measure(s13, (GN, GN, GN, CPU)).status == "oom" and "C<->A" in swaps,
              "every gpu_now tool is tried as the swap partner")
        prompt = build_prompt(s13, fig7.profiles(s13), "warm", mon2, 3, 20)
        check("SWAP C<->B" in prompt and "full: A:GN B:GQ C:GN D:CPU" in prompt
              and "A clip_x12: mem=627 gpu=147 cpu=3018 load=862" in prompt,
              "prompt carries Fig. 7's tool cards, history and swap what-ifs")
        check(parse_llm_mapping('<think>{"A":"cpu","B":"cpu","C":"cpu","D":"cpu"}</think>\n'
                                'So: {"A": "gpu_now", "B": "gpu_queue", "C": "GN", "D": "cpu"}',
                                s13) == (GN, GQ, GN, CPU),
              "LLM answer parser takes the final JSON after the reasoning")
        check(parse_llm_mapping('{"A": "gpu", "B": "gpu_queue", "C": "cpu", "D": "cpu"}',
                                SCENARIOS["S5"]) is None,
              "gpu_queue is refused in a CPU/GPU-only scenario")

        class _Replies:  # canned server answers: (content, reasoning, usage, finish_reason)
            def __init__(self, replies):
                self.replies, self.prompts = list(replies), []

            def chat(self, messages):
                self.prompts.append(messages)
                return self.replies.pop(0)

        draft = 'Maybe {"A": "cpu", "B": "cpu", "C": "cpu", "D": "cpu"} ... still thinking'
        final = 'ok</think>\n{"A": "gpu_now", "B": "gpu_queue", "C": "gpu_now", "D": "cpu"}'
        stub = _Replies([(draft, "", {}, "length"), (final, "", {}, "stop")])
        got, info = LLMPolicy(stub, 2).decide(prompt, s13, (GN,) * 4, mon2, fig7.profiles(s13),
                                              "warm")
        check(got == (GN, GQ, GN, CPU) and len(info["attempts"]) == 2 and not info["fallback"]
              and len(stub.prompts[1]) == 1 and "IMPORTANT: end your answer" in
              stub.prompts[1][0]["content"],
              "an answer cut off by max_tokens is not parsed; the re-ask is one fresh prompt")
        stub = _Replies([("", draft, {}, "length")] * 3)
        got, info = LLMPolicy(stub, 2).decide(prompt, s13, (GN, GQ, CPU, CPU), mon2,
                                              fig7.profiles(s13), "warm")
        check(got == (GN, GQ, CPU, CPU) and info["fallback"] and len(info["attempts"]) == 3,
              "after the retries the round keeps the fallback mapping")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print()
    if failures:
        print(f"selftest: {len(failures)} check(s) FAILED")
        return 1
    print("selftest: all checks passed")
    return 0


# ===========================================================================
# 15. Command line
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    sim = argparse.ArgumentParser(add_help=False)
    g = sim.add_argument_group("converter and simulator")
    g.add_argument("--converter-module",
                   default=os.environ.get("TOOLDAG_CONVERTER_MODULE", DEFAULT_CONVERTER_MODULE),
                   help="import path of the TOOLDAG-enabled converter (default: %(default)s)")
    g.add_argument("--backend", choices=("astra", "model"), default="astra",
                   help="astra = run ASTRA-Sim (the experiment); model = reference model "
                        "(dry run, no simulator)")
    g.add_argument("--astra-cmd", default=os.environ.get("ASTRA_SIM_CMD"),
                   help="ASTRA-Sim command line with {workload} for the ET prefix; may also use "
                        "{et_dir} and {num_npus} (env ASTRA_SIM_CMD)")
    g.add_argument("--astra-cwd", default=None, help="working directory for ASTRA-Sim")
    g.add_argument("--astra-timeout", type=float, default=600.0, help="seconds per run")
    g.add_argument("--astra-log-file", default=None,
                   help="log file ASTRA-Sim writes, parsed in addition to stdout/stderr "
                        "(LLMServingSim: <astra-cwd>/log/log.log holds the per-node lines)")
    g.add_argument("--astra-stdin", default="exit\\n",
                   help="text fed to ASTRA-Sim's stdin (LLMServingSim's interactive binary "
                        "waits for 'exit' after the workload finishes)")
    g.add_argument("--finish-regex", default=DEFAULT_FINISH_RE,
                   help="regex for the per-sys finish line; groups (?P<ticks>) [and (?P<npu>)]")
    g.add_argument("--callback-regex", default=DEFAULT_CALLBACK_RE,
                   help="regex for per-node completion lines; groups (?P<tick>) (?P<id>) "
                        "[and (?P<npu>)]")
    g.add_argument("--num-npus", type=int, default=1,
                   help="ET replicas to write; must equal the NPU count of the network config")
    g.add_argument("--time-units-per-ms", type=float, default=1e6,
                   help="duration_micros units per ms written into the ET. LLMServingSim's "
                        "ASTRA-Sim takes the field as nanoseconds (Workload::issue_replay), "
                        "hence 1e6; stock ASTRA-Sim 2.x reads microseconds (1000)")
    g.add_argument("--sim-ticks-per-ms", type=float, default=1e6,
                   help="ASTRA-Sim ticks per ms in its output (1e6: nanosecond ticks)")
    g.add_argument("--contention", choices=("simulator", "serialize"), default="simulator",
                   help="serialize = explicit per-resource ordering edges in the ET")
    g.add_argument("--slice-ms", type=float, default=0.0,
                   help="> 0 splits tool work into slices shared round-robin (processor sharing)")
    g.add_argument("--cpu-flag", choices=("bool", "int32"), default="bool",
                   help="encoding of the is_cpu_op attribute")
    g.add_argument("--profiles", choices=("paper", "k-scaled"), default="paper",
                   help="amplified-tool times: paper-measured where reported, or k x Table 1")
    g.add_argument("--profile-overrides", default=None, metavar="fig7-s13|FILE.json",
                   help="override tool cards: 'fig7-s13' = the S13 values the paper's Fig. 6/7 "
                        "run used, or a JSON file {tool: {gpu_ms, cpu_ms, vram_mb, load_ms}}")
    g.add_argument("--repeats", type=int, default=1,
                   help="executions per measurement (only matters with --noise-cv)")
    g.add_argument("--noise-cv", type=float, default=0.0,
                   help="coefficient of variation of per-execution lognormal noise (0 = exact)")
    g.add_argument("--tie-tol", type=float, default=0.01,
                   help="mappings within this fraction of the oracle E2E count as the oracle")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workdir", default=None, help="where ETs are written (default: a temp dir)")
    g.add_argument("--keep-et", action="store_true", help="keep every generated ET")
    g.add_argument("--scenarios", default="S1-S13", help="e.g. S1-S13, S5,S13, all")
    g.add_argument("--out", default="tooldag_results", help="results directory")

    strat = argparse.ArgumentParser(add_help=False)
    h = strat.add_argument_group("strategies")
    h.add_argument("--rounds", type=int, default=20, help="scheduling rounds per run")
    h.add_argument("--heft-cf", type=float, default=1.5,
                   help="HEFT-CF static GPU contention factor")
    h.add_argument("--resume", action="store_true",
                   help="continue an interrupted run in the same --out folder: complete strategy "
                        "runs are kept, the rest is run again")

    agent = argparse.ArgumentParser(add_help=False)
    k = agent.add_argument_group("agentic scheduler")
    k.add_argument("--modes", default="cold,warm", help="cold, warm or cold,warm")
    k.add_argument("--monitor-level", default="auto", choices=("auto", "0", "1", "2"),
                   help="auto = 0 for S1-S4, 1 for S5-S10, 2 for S11-S13")
    k.add_argument("--window", type=int, default=5, help="running-average window")
    k.add_argument("--hint-after", type=int, default=2,
                   help="stagnant repeats before an exploration hint")
    k.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"),
                   help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 (vLLM); "
                        "env OPENAI_BASE_URL")
    k.add_argument("--llm-model", default=os.environ.get("LLM_MODEL"),
                   help="served model name (default: the first one /models lists)")
    k.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    k.add_argument("--temperature", type=float, default=0.6,
                   help="0.6 with top-p 0.95 is DeepSeek's recommendation for the R1 distills")
    k.add_argument("--top-p", type=float, default=0.95)
    k.add_argument("--max-tokens", type=int, default=8192,
                   help="answer budget incl. reasoning; prompt (<2k tokens) + this must fit the "
                        "server's --max-model-len")
    k.add_argument("--llm-timeout", type=float, default=900.0, help="seconds per LLM call")
    k.add_argument("--llm-retries", type=int, default=2,
                   help="re-asks when the answer holds no valid mapping")
    k.add_argument("--emulated-llm", action="store_true",
                   help="rule-based emulation of the paper's LLM scheduler (no model is run); "
                        "report as LLM-emulated")
    k.add_argument("--dry-run-llm", action="store_true",
                   help="replace the LLM by a deterministic stand-in (plumbing test only)")

    abl = argparse.ArgumentParser(add_help=False)
    m = abl.add_argument_group("ablation")
    m.add_argument("--ablation-scenario", default="S13")
    m.add_argument("--ablation-mode", default="warm", choices=("cold", "warm"))

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest", parents=[sim], help="converter + model checks (no simulator)")
    p = sub.add_parser("crosscheck", parents=[sim], help="ASTRA-Sim vs reference model")
    p.add_argument("--crosscheck-tol", type=float, default=0.002)
    sub.add_parser("oracle", parents=[sim], help="brute-force oracle for every scenario")
    sub.add_parser("baselines", parents=[sim, strat], help="All-GPU, HEFT-CF, UCB1, StarPU, greedy")
    sub.add_parser("agentic", parents=[sim, strat, agent], help="LLM scheduler + runtime monitor")
    sub.add_parser("ablation", parents=[sim, strat, agent, abl], help="Table 6 monitor ablation")
    p = sub.add_parser("all", parents=[sim, strat, agent, abl], help="everything")
    p.add_argument("--with-ablation", action="store_true")
    p = sub.add_parser("llmcheck", parents=[sim, strat, agent],
                       help="one real LLM scheduling call, to verify the LLM server")
    p.add_argument("--check-scenario", default="S13")
    p.add_argument("--check-mode", default="warm", choices=("cold", "warm"))
    p = sub.add_parser("merge", parents=[sim, strat], help="combine result folders into one report")
    p.add_argument("--from", dest="from_dirs", required=True,
                   help="comma-separated result folders, e.g. res_a,res_b,res_c")
    p = sub.add_parser("export", parents=[sim], help="write a TOOLDAG text trace")
    p.add_argument("--scenario", required=True)
    p.add_argument("--mapping", required=True, help="e.g. GN,GQ,GN,CPU or gpu,gpu,cpu,gpu")
    p.add_argument("--trace", default=None, help="output file (default: stdout)")
    p.add_argument("--convert", default=None, metavar="PREFIX",
                   help="also run the converter's convert() on the trace, writing PREFIX.<npu>.et")
    return parser


COMMANDS = {"selftest": cmd_selftest, "crosscheck": cmd_crosscheck, "oracle": cmd_oracle,
            "baselines": cmd_baselines, "agentic": cmd_agentic, "ablation": cmd_ablation,
            "all": cmd_all, "export": cmd_export, "merge": cmd_merge, "llmcheck": cmd_llmcheck}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    conv = load_converter(args.converter_module)
    try:
        return COMMANDS[args.command](args, conv)
    except (SimulatorError, LLMError) as e:
        log(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
