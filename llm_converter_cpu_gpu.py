import heapq
import logging
import math
import re
from io import TextIOWrapper
from typing import Any, Dict, List
from enum import Enum

from ...schema.protobuf.et_def_pb2 import *
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ..third_party.utils.protolib import encodeMessage as encode_message


# Memory type is deprecated for latest version of chakra & astra-sim
# It only uses tensor_size for the remote memory issue in Workload.cc
# Added memory types using enum and add tensor_loc in node attribute in et_feeder_node.cpp
class MemoryType(Enum):
    INVALID_MEMORY = 0
    LOCAL_MEMORY = 1
    REMOTE_MEMORY = 2
    CXL_MEMORY = 3
    STORAGE_MEMORY = 4


class Layer:
    def __init__(self, line: str = None, cols: List[str] = None):
        try:
            # ``cols`` lets a caller that already holds the fields skip the
            # format-and-resplit round trip; ``line`` is the text path.
            col = cols if cols is not None else line.strip().split()
            if col[0] == 'EXPERT': # If Expert Flag
                self.name = col[0]
                self.expert_num = col[1]
                self.comm_type, self.involved_dim = self._parse_comm_type(str(col[2]) if len(col) > 2 else "NONE")
                self.comm_size = int(col[3]) if len(col) > 3 else 0
                self.is_expert = True
                self.is_pim = False
                self.comm_node = None
                self.comp_node = None
            elif col[0] == 'PIM': # If PIM Flag
                self.name = col[0]
                self.pim_num = col[1]
                self.is_expert = False
                self.is_pim = True
                self.comm_node = None
                self.comp_node = None
                self.comm_type = "NONE"
                self.involved_dim = None
            else:
                self.is_expert = False
                self.is_pim = False
                self.name = col[0]

                # compuation
                self.comp_time = int(col[1])
                self.comp_node = None

                # memory
                self.input_memory_loc = str(col[2])
                self.input_memory_size = int(col[3])
                self.input_memory_node = None
                self.weight_memory_loc = str(col[4])
                self.weight_memory_size = int(col[5])
                self.weight_memory_node = None
                self.output_memory_loc = str(col[6])
                self.output_memory_size = int(col[7])
                self.output_memory_node = None

                # communication (supports ALLREDUCE:1,0 format for involved_dim)
                self.comm_type, self.involved_dim = self._parse_comm_type(str(col[8]))
                self.comm_size = int(col[9])
                self.comm_node = None

                self.misc = str(col[10])
        except:
            raise ValueError(f"Cannot parse the following layer -- \"{line}\"")

    @staticmethod
    def _parse_comm_type(s: str):
        """Parse comm_type string, extracting optional involved_dim.

        'ALLREDUCE:1,0' -> ('ALLREDUCE', [True, False])
        'ALLTOALL'      -> ('ALLTOALL', None)
        'NONE'          -> ('NONE', None)
        """
        if ':' in s:
            comm_type, dim_str = s.split(':', 1)
            involved_dim = [v == '1' for v in dim_str.split(',')]
            return comm_type, involved_dim
        return s, None


# ---------------------------------------------------------------------------
# TOOLDAG: tool-level CPU / GPU placement for agentic workflows
# ---------------------------------------------------------------------------
#
# Models the setting of Lu & Reda, "Agentic CPU-GPU Scheduling for
# Heterogeneous AI Workloads" (arXiv:2607.22242). A workflow is a small DAG of
# tool calls (Whisper, BERT, vector search, ...) and every tool is placed on
# one of three execution modes:
#
#   cpu        runs on the host CPU and holds no VRAM;
#   gpu_now    runs on the GPU as soon as its inputs are ready and keeps its
#              model resident -- it counts toward the VRAM budget -- while
#              it runs;
#   gpu_queue  waits for the immediate (gpu_now) phase to finish, then runs
#              alone on the GPU, one queued tool at a time, after paying its
#              model load time; it holds no VRAM while it waits.
#
# A tool becomes one COMP_NODE (two for gpu_queue: load, then compute) whose
# is_cpu_op attribute sends it to the NPU's CPU or GPU resource. Contention
# between co-placed tools is then whatever ASTRA-Sim's per-NPU resources do
# with concurrent compute nodes -- the converter only describes the work.
# ASTRA-Sim does not track VRAM, so a mapping whose co-resident tools exceed
# the budget is rejected here, before anything is written, the way the real
# system fails with OOM.

TOOL_PLACEMENTS = ("cpu", "gpu_now", "gpu_queue")

_PLACEMENT_ALIASES = {
    "cpu": "cpu",
    "gpu": "gpu_now", "gpu_now": "gpu_now", "gn": "gpu_now",
    "gpu_queue": "gpu_queue", "gq": "gpu_queue", "queue": "gpu_queue",
}

# Task ids stay alphanumeric so node names (TOOL_<task>_<tool>_<phase>) split
# unambiguously; a tool name only has to be one whitespace-free token.
_TASK_ID_RE = re.compile(r"[A-Za-z0-9]+")
_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]+")

# Header keys of a TOOLDAG trace and their types. Anything else is rejected:
# a mistyped vram_budget_mb would otherwise silently drop the memory limit.
TOOL_DAG_HEADER_KEYS = {
    "vram_budget_mb": float,
    "time_units_per_ms": float,
    "contention": str,
    "slice_ms": float,
    "cpu_flag": str,
}


class ToolTask:
    """One tool invocation of a TOOLDAG trace.

    Text row, whitespace separated:

        task  tool  placement  gpu_ms  cpu_ms  load_ms  vram_mb  deps

    ``deps`` lists the ids of the tasks whose output this one consumes,
    comma separated, or ``-`` for a source. Both device times are carried so
    the trace keeps the whole tool card; only the placed device is simulated.
    """

    NUM_COLS = 8

    def __init__(self, line: str = None, cols: List[Any] = None):
        col = cols if cols is not None else (line or "").strip().split()
        try:
            if len(col) != self.NUM_COLS:
                raise ValueError(f"expected {self.NUM_COLS} columns, got {len(col)}")
            self.task = str(col[0])
            self.tool = str(col[1])
            self.placement = self.normalize_placement(str(col[2]))
            self.gpu_ms = float(col[3])
            self.cpu_ms = float(col[4])
            self.load_ms = float(col[5])
            self.vram_mb = float(col[6])
            deps = str(col[7])
            self.deps = [] if deps == "-" else [d for d in deps.split(",") if d]
            self._validate()
        except ValueError as e:
            shown = line.strip() if line is not None else " ".join(map(str, col))
            raise ValueError(f"Cannot parse the following tool task -- \"{shown}\" ({e})") from None

    @classmethod
    def make(cls, task: str, tool: str, placement: str, gpu_ms: float, cpu_ms: float,
             load_ms: float = 0.0, vram_mb: float = 0.0, deps=()) -> "ToolTask":
        return cls(cols=[task, tool, placement, gpu_ms, cpu_ms, load_ms, vram_mb,
                         ",".join(deps) if deps else "-"])

    @staticmethod
    def normalize_placement(placement: str) -> str:
        try:
            return _PLACEMENT_ALIASES[placement.strip().lower()]
        except KeyError:
            raise ValueError(f"unknown placement '{placement}' "
                             f"(expected one of {', '.join(TOOL_PLACEMENTS)})") from None

    def _validate(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task):
            raise ValueError(f"task id '{self.task}' must be alphanumeric")
        if not _TOOL_NAME_RE.fullmatch(self.tool):
            raise ValueError(f"tool name '{self.tool}' must be one token of [A-Za-z0-9_.-]")
        for field in ("gpu_ms", "cpu_ms", "load_ms", "vram_mb"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite number >= 0, got {value}")
        for dep in self.deps:
            if not _TASK_ID_RE.fullmatch(dep):
                raise ValueError(f"dependency '{dep}' is not a task id")
            if dep == self.task:
                raise ValueError("a task cannot depend on itself")

    def to_cols(self) -> List[str]:
        """The row as text columns; ``ToolTask(cols=t.to_cols())`` rebuilds ``t``."""
        def num(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else repr(v)
        return [self.task, self.tool, self.placement, num(self.gpu_ms), num(self.cpu_ms),
                num(self.load_ms), num(self.vram_mb), ",".join(self.deps) or "-"]


class VramBudgetExceeded(ValueError):
    """The tools that can be resident on the GPU together exceed the budget.

    Raised before any file is written: the real system fails with OOM, so
    there is nothing to simulate. ``tasks`` names the co-resident set that
    reaches ``peak_mb``.
    """

    def __init__(self, peak_mb: float, budget_mb: float, tasks: List[str]):
        self.peak_mb = peak_mb
        self.budget_mb = budget_mb
        self.tasks = list(tasks)
        super().__init__(f"VRAM budget exceeded: {'+'.join(self.tasks)} = {peak_mb:g} MB "
                         f"> {budget_mb:g} MB")


class ToolDagPlan:
    """Precedence and memory analysis of one TOOLDAG mapping; writes nothing.

    ``aug_parents`` is the DAG plus the queue semantics: every queued task
    waits for the immediate-phase gpu_now tasks and for the queued task
    before it. ``order`` is a topological order of that graph (ties broken by
    row order) and ``level`` the longest-path depth of each task in it.
    """

    def __init__(self, by_id: Dict[str, ToolTask], dag_parents: Dict[str, List[str]],
                 aug_parents: Dict[str, List[str]], order: List[str], level: Dict[str, int],
                 peak_vram_mb: float, peak_tasks: List[str], vram_budget_mb: float):
        self.by_id = by_id
        self.dag_parents = dag_parents
        self.aug_parents = aug_parents
        self.order = order
        self.level = level
        self.peak_vram_mb = peak_vram_mb
        self.peak_tasks = peak_tasks
        self.vram_budget_mb = vram_budget_mb

    @property
    def feasible(self) -> bool:
        return self.vram_budget_mb <= 0 or self.peak_vram_mb <= self.vram_budget_mb + 1e-9


class LLMConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_npus: int,
        npu_offset: int = 0,
        local_offloading: bool = False,
    ):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.npu_offset = npu_offset
        self.local_offloading = local_offloading
        self.next_node_id = 0

        # For send & recv nodes
        self.next_comm_tag = 0
        self.comm_tag_dict = dict()

        # Set by convert_rows so get_layers can return them instead of
        # parsing a file. See convert_rows.
        self._layers = None

    def get_global_metadata(self):
        # ``input_file`` carries the trace's *path*, not its contents.
        #
        # It used to embed the whole trace text. Nothing consumes it:
        # ETFeeder::readGlobalMetadata() reads the message into a local
        # shared_ptr and drops it on the floor, and no other reader touches
        # the attribute. So every byte of it was written by the converter,
        # re-parsed by the feeder's protobuf reader, and discarded.
        #
        # It was not a small overhead. On the swe-bench MoE DP+EP example a
        # trace is ~82 KB of a 117 KB .et -- 70% of the file -- and the
        # simulator generates 8,810 of them for one session, so it was
        # ~720 MB of encode/write/read/parse per run, plus a second full
        # read of the trace file here purely to obtain the text.
        #
        # The path keeps the provenance that made this attribute useful for
        # debugging; the trace itself is written to disk by --save-trace-text.
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=self.input_filename),
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata
    
    def get_layers(self, f: TextIOWrapper) -> List[Layer]:
        if self._layers is not None:
            return self._layers
        layers: List[Layer] = []
        for line in f:
            layers.append(Layer(line))
        return layers

    def convert_rows(self, header_line: str, rows: List[List[str]]) -> Any:
        """Convert from pre-parsed field lists, with no text round trip.

        The simulator already holds every field. Formatting them into padded
        columns, writing the file, reading it back and splitting each line
        again is pure overhead now that the converter runs in the same
        process -- 0.65 ms of formatting and writing per batch plus 0.9 ms of
        reading and re-parsing, against 8,810 batches on the swe-bench MoE
        DP+EP example.

        Nothing downstream changes: convert_common, convert_prefill and
        convert_event each touch the file handle exactly once, to call
        get_layers, so seeding the layers is the whole of it. Header parsing
        mirrors convert() on a string instead of a readline, and num_layers
        is len(rows) -- which is precisely what convert() reads off line two,
        since that is what the writer puts there.

        A TOOLDAG header routes the rows to convert_tool_dag instead -- they
        are tool tasks, not layers -- and its summary is returned.
        """
        first_line = header_line.strip().split()
        execution_type = first_line[0]
        if execution_type == "TOOLDAG":
            return self.convert_tool_dag(
                [ToolTask(cols=cols) for cols in rows],
                **self.tool_dag_options(self.parse_header_fields(first_line[1:])))

        self._layers = [Layer(cols=cols) for cols in rows]

        header = {}
        fields = first_line[1:]
        for i in range(0, len(fields) - 1, 2):
            if fields[i].endswith(":"):
                header[fields[i][:-1]] = fields[i + 1]

        num_npu_group = int(header.get("model_parallel_NPU_group", 0))
        boundary_str = header.get("pp_stage_boundaries", "")
        stage_boundaries = (
            [int(b) for b in boundary_str.split(",")] if boundary_str else []
        )
        num_layers = len(rows)

        if execution_type in ("COLOCATED", "DECODE"):
            if num_npu_group <= 0:
                raise ValueError(f"model_parallel_NPU_group <= 0")
            self.convert_common(None, num_layers, num_npu_group, stage_boundaries)
        elif execution_type == "PREFILL":
            if num_npu_group <= 0:
                raise ValueError(f"model_parallel_NPU_group <= 0")
            self.convert_prefill(None, num_layers, num_npu_group, stage_boundaries)
        elif execution_type == "EVENT":
            self.convert_event(None, num_layers)
        else:
            raise ValueError(f"Unsupported execution type, {execution_type}")

    def get_next_node_id(self) -> int:
        ret = self.next_node_id
        self.next_node_id += 1
        return ret
    
    def get_next_comm_tag(self) -> int:
        ret = self.next_comm_tag
        self.next_comm_tag += 1
        return ret

    def get_node(self, name: str, node_type: NodeType) -> Any:
        node = Node()
        node.id = self.get_next_node_id()
        node.name = name
        node.type = node_type
        return node

    def get_comp_node(self, layer_name: str, comp_time: int, is_cpu: bool = False) -> Any:
        node = self.get_node("COMP_NODE_" + layer_name, COMP_NODE)
        node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=is_cpu))
        return node

    def get_cpu_comp_node(self, layer_name: str, comp_time: int) -> Any:
        node = self.get_node("CPU_COMP_NODE_" + layer_name, COMP_NODE)
        node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=True))
        return node
    
    def get_comm_type(self, comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return 0
    
    def get_comm_coll_node(self, layer_name: str, comm_type: str, comm_size: int,
                           involved_dim: list = None) -> Any:
        node = self.get_node(f"COMM_COLL_NODE_{layer_name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        if involved_dim is not None:
            node.attr.append(ChakraAttr(name="involved_dim",
                                        bool_list=BoolList(values=involved_dim)))
        return node

    def get_comm_node(self, is_send: bool, layer_name: str, comm_type: str, comm_size: int,
                           comm_src: int, comm_dst: int, id: int = 0) -> Any:
        if is_send:
            node = self.get_node(
                    f"COMM_SEND_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_SEND_NODE)
        else:
            node = self.get_node(
                    f"COMM_RECV_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_RECV_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=comm_src))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=comm_dst))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        comm_key = f"{comm_src}_{comm_dst}_{id}"
        if comm_key in self.comm_tag_dict:
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.comm_tag_dict[comm_key]))
        else:
            new_tag = self.get_next_comm_tag()
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=new_tag))
            self.comm_tag_dict[comm_key] = new_tag

        # check if SEND/RECV pair have same tags
        # print(f"name: {node.name}, src: {node.comm_src}, dst: {node.comm_dst}, size: {node.comm_size}, key: {comm_key}, tag: {node.comm_tag}")
        return node
    
    def get_mem_type(self, mem_type: str) -> int:
        mem_type = mem_type.split(':')[0]  # Exclude the device number if present
        if mem_type == "LOCAL":
            return MemoryType.LOCAL_MEMORY.value
        elif mem_type == "REMOTE":
            return MemoryType.REMOTE_MEMORY.value
        elif mem_type == "CXL":
            return MemoryType.CXL_MEMORY.value
        elif mem_type == "STORAGE":
            return MemoryType.STORAGE_MEMORY.value
        return MemoryType.INVALID_MEMORY.value
    
    def get_mem_device(self, mem_type: str) -> int:
        """
        Extract the device index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 1
        - "REMOTE:1.3"      -> returns 1  (device = 1, channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            dev_part = parts[1].split(".", 1)[0]
            if dev_part.isdigit():
                return int(dev_part)
        return 0  # Default device number


    def get_mem_channel(self, mem_type: str) -> int:
        """
        Extract the channel index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 0  (no channel specified)
        - "REMOTE:1.3"      -> returns 3  (channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            sub = parts[1].split(".", 1)
            if len(sub) == 2:
                chan_part = sub[1]
                if chan_part.isdigit():
                    return int(chan_part)
        return 0  # Default channel number


    def get_memory_load_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_LOAD_NODE_" + layer_name + "_" + tensor_type, MEM_LOAD_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        return node

    def get_memory_store_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_STORE_NODE_" + layer_name + "_" + tensor_type, MEM_STORE_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        return node

    def get_pim_compute_node(self, layer_name: str, tensor_type: str, comp_time: int, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("PIM_COMP_NODE_" + layer_name + "_" + tensor_type, PIM_COMP_NODE)
        node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_channel", uint32_val=self.get_mem_channel(mem_type)))
        return node
        
    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    def get_stage_edges(self, num_layers: int, num_npu_group: int,
                        stage_boundaries: List[int]) -> List[Any]:
        """Resolve the [start, end) trace-line range owned by each pipeline stage.

        ``stage_boundaries`` comes from the trace header and holds the line
        index at which every stage after the first begins. The frontend puts
        those cuts on transformer-block boundaries, which is what makes the
        tensor crossing a stage boundary the hidden state: the sending
        layer's output_size then equals the receiving layer's input_size, and
        the SEND/RECV pair matches in ASTRA-sim (its callback tracker keys on
        chunk size, so a mismatch deadlocks the run instead of erroring).
        Splitting the line count evenly instead lands cuts inside a block --
        e.g. between qkv_proj and rotary_emb, whose declared sizes differ by
        the V projection -- so do not reintroduce that.
        """
        if num_npu_group == 1:
            return [(0, num_layers)]
        if len(stage_boundaries) != num_npu_group - 1:
            raise ValueError(
                f"trace declares model_parallel_NPU_group: {num_npu_group} but "
                f"carries {len(stage_boundaries)} pp_stage_boundaries "
                f"(expected {num_npu_group - 1}); regenerate the trace")
        edges = [0] + list(stage_boundaries) + [num_layers]
        for i in range(len(edges) - 1):
            if not 0 <= edges[i] < edges[i + 1] <= num_layers:
                raise ValueError(
                    f"pp_stage_boundaries {stage_boundaries} are not a strictly "
                    f"increasing split of {num_layers} layers")
        return [(edges[i], edges[i + 1]) for i in range(num_npu_group)]

    def convert_common(self, f: TextIOWrapper, num_layers: int, num_npu_group: int,
                       stage_boundaries: List[int] = None):
        layers: list[Layer] = self.get_layers(f)

        # vllm: check eviction or load
        evict = None
        load = None
        ev_ld_cnt = 0
        for i in range(2):
            if 'kv_load' in layers[i].name:
                load = self.get_memory_load_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size, # already per npu kv_cache size
                        )
            elif 'kv_evict' in layers[i].name:
                evict = self.get_memory_store_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size,
                        )
            else:
                continue
            ev_ld_cnt += 1
            
        layers = layers[ev_ld_cnt:]
        num_layers -= ev_ld_cnt

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        stage_edges = self.get_stage_edges(num_layers, num_npu_group,
                                           stage_boundaries or [])

        for npu_group in range(num_npu_group):
            for npu_offset in range(npus_per_group):
                # Re-read the authoritative edges per rank: the walk below may
                # rebind layer_end if it ever overruns.
                layer_start, layer_end = stage_edges[npu_group]
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                output_filename = "%s.%d.et" % (self.output_filename, npu_id)
                first_comp_node = True
                with open(output_filename, "wb") as g:
                    global_metadata = self.get_global_metadata()
                    encode_message(g, global_metadata)
                    if evict != None:
                        encode_message(g, evict)
                    if load != None:
                        encode_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        encode_message(g, input_load_node)                  
                    else:
                        if layers[layer_start].is_expert or layers[layer_start].is_pim:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)

                    expert_start = False
                    pim_start = False
                    attn_remain = False # to handle remaining prefill attention after pim
                    pim_parent_nodes = []
                    pim_comp_nodes = []
                    past_pim_comp_nodes = []
                    last_batch_type = "BATCH_1"
                    layer_num = layer_start
                    while expert_start or pim_start or attn_remain or layer_num < layer_end:
                        if not layers[layer_num].is_expert and not layers[layer_num].is_pim: 
                            if (self.local_offloading or layers[layer_num].weight_memory_loc != "LOCAL") and layers[layer_num].weight_memory_size > 0:
                                # Load weight (for weight offloading)
                                weight_load_node = self.get_memory_load_node(
                                    layers[layer_num].name,
                                    "WEIGHT",
                                    layers[layer_num].weight_memory_loc,
                                    layers[layer_num].weight_memory_size,
                                )
                                layers[layer_num].weight_memory_node = weight_load_node
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node) # dependent to previous comp_node due to gate function
                                encode_message(g, weight_load_node)
                            # Compute
                            if layers[layer_num].comp_time != 0 and not pim_start: # pim computation is handled pim_comp_node
                                gpu_time = max(1, (layers[layer_num].comp_time * 3) // 4)
                                cpu_time = max(1, layers[layer_num].comp_time - gpu_time)

                                gpu_comp_node = self.get_comp_node(
                                    layers[layer_num].name + "_GPU",
                                    layers[layer_num].comp_time)

                                encode_message(g, gpu_comp_node)

                                comp_node = gpu_comp_node
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                # End of expert, add ALLTOALL communication after expert computation
                                if layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    encode_message(g, comm_coll_node)
                                layer_num += 1
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1
                        # pim layer starts
                        elif layers[layer_num].is_pim:
                            pim_start = True
                            # check attention end
                            if layers[layer_num].pim_num == 'END':
                                pim_start = False
                                if "attn" in layers[layer_num + 1].name:
                                    attn_remain = True # prefill attn remains
                                layer_num += 1
                                continue
                            # add pim parent nodes for dependency
                            elif int(layers[layer_num].pim_num) == 0:
                                if first_comp_node:
                                    if npu_group == 0:
                                        pim_parent_nodes.append(input_load_node)
                                    else:
                                        pim_parent_nodes.append(receive_input_node)
                                    if evict != None:
                                        pim_parent_nodes.append(evict)
                                    if load != None:
                                        pim_parent_nodes.append(load)
                                    # discarded weight parent because attention has no weight
                                    first_comp_node = False
                                else:
                                    # discarded weight parent because attention has no weight
                                    if layers[layer_num - 1].comm_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                                    else:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                            # round robin assignment
                            pim_id = int(layers[layer_num].pim_num) % npus_per_group
                            if npu_offset != pim_id:
                                # go to next attention
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_pim:
                                        break
                            else:
                                layer_num += 1

                    # The frontend cuts stages on transformer-block boundaries,
                    # so the walk above lands exactly on the stage edge. Expert
                    # and PIM blocks live inside a block and can no longer be
                    # straddled; warn loudly if that ever stops holding.
                    if layer_num != layer_end:
                        print(f"Warning! pipeline stage {npu_group} walked to "
                              f"layer {layer_num}, expected {layer_end}")
                        layer_end = layer_num

                    if npu_group == (num_npu_group - 1):
                        # Store output (for the last layer)
                        # The last layer is the sampler: what crosses back to
                        # the host is its OUTPUT, the sampled token ids, 4 bytes
                        # per sequence.
                        #
                        # Do not switch this back to input_memory_size. That read
                        # was a deliberate down-scaling when lm_head was the last
                        # trace layer: its output is the logits
                        # (num_seqs * vocab_size * dtype) and billing those to CPU
                        # memory every iteration was a large overcharge, so the
                        # smaller input (the hidden state) stood in. Once sampler
                        # was appended as the new last layer the two swapped
                        # roles -- sampler's output is the token ids and its input
                        # is the logits -- so reading the input landed back on
                        # exactly the tensor the workaround existed to avoid.
                        output_store_node = self.get_memory_store_node(
                            layers[layer_end - 1].name,
                            "OUTPUT",
                            layers[layer_end - 1].output_memory_loc,
                            layers[layer_end - 1].output_memory_size,
                        )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(output_store_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(output_store_node, comp_node)
                        else:
                            self.add_parent(output_store_node, layers[layer_end - 2].comp_node)
                        encode_message(g, output_store_node)
                    else:
                        if layers[layer_end - 1].is_expert or layers[layer_end - 1].is_pim:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
    
    def convert_prefill(self, f: TextIOWrapper, num_layers: int, num_npu_group: int,
                        stage_boundaries: List[int] = None):
        layers: list[Layer] = self.get_layers(f)
        # There will be no pim operation in prefill (PIM cannot perform GEMM)

        # vllm: check eviction or load
        evict = None
        load = None
        ev_ld_cnt = 0
        for i in range(2):
            if 'kv_load' in layers[i].name:
                load = self.get_memory_load_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size, # already per npu kv_cache size
                        )
            elif 'kv_evict' in layers[i].name:
                evict = self.get_memory_store_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size,
                        )
            else:
                continue
            ev_ld_cnt += 1
            
        layers = layers[ev_ld_cnt:]
        num_layers -= ev_ld_cnt

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        stage_edges = self.get_stage_edges(num_layers, num_npu_group,
                                           stage_boundaries or [])

        for npu_group in range(num_npu_group):
            for npu_offset in range(npus_per_group):
                # Re-read the authoritative edges per rank: the walk below may
                # rebind layer_end if it ever overruns.
                layer_start, layer_end = stage_edges[npu_group]
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                output_filename1 = "%s.%d.et" % (self.output_filename, npu_id)
                output_filename2 = "%s.%d.et" % (self.output_filename, npu_id + self.num_npus) # sender for prefill-decode
                first_comp_node = True
                with open(output_filename1, "wb") as g, open(output_filename2, "wb") as s:
                    global_metadata = self.get_global_metadata()
                    encode_message(g, global_metadata)
                    encode_message(s, global_metadata)
                    if evict != None:
                        encode_message(g, evict)
                    if load != None:
                        encode_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        encode_message(g, input_load_node)                  
                    else:
                        if layers[layer_start].is_expert:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)

                    expert_start = False
                    layer_num = layer_start
                    while expert_start or layer_num < layer_end:
                        if not layers[layer_num].is_expert:
                            if (self.local_offloading or layers[layer_num].weight_memory_loc != "LOCAL") and layers[layer_num].weight_memory_size > 0:
                                # Load weight (for weight offloading)
                                weight_load_node = self.get_memory_load_node(
                                    layers[layer_num].name,
                                    "WEIGHT",
                                    layers[layer_num].weight_memory_loc,
                                    layers[layer_num].weight_memory_size,
                                )
                                layers[layer_num].weight_memory_node = weight_load_node
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node) # dependent to previous comp_node due to gate function
                                encode_message(g, weight_load_node)
                            
                            # Compute
                            if layers[layer_num].comp_time != 0:
                                gpu_time = max(1, (layers[layer_num].comp_time * 3) // 4)
                                cpu_time = max(1, layers[layer_num].comp_time - gpu_time)

                                gpu_comp_node = self.get_comp_node(
                                    layers[layer_num].name + "_GPU",
                                    layers[layer_num].comp_time)

                                encode_message(g, gpu_comp_node)

                                comp_node = gpu_comp_node
                                layers[layer_num].comp_node = comp_node

                                if first_comp_node:
                                    if npu_group == 0:
                                        self.add_parent(comp_node, input_load_node)
                                    else:
                                        self.add_parent(comp_node, receive_input_node)
                                    if evict != None:
                                        self.add_parent(comp_node, evict)
                                    if load != None:
                                        self.add_parent(comp_node, load)
                                    if layers[layer_num].weight_memory_node != None:
                                        self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                    first_comp_node = False
                                else:
                                    if layers[layer_num].weight_memory_node != None:
                                        self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                    if layers[layer_num - 1].comm_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                    else:
                                        self.add_parent(comp_node, layers[layer_num - 2].comp_node)
                                
                                encode_message(g, comp_node)

                                # Send KV cache after each kv_proj.
                                # comm_size comes from the trace's comm_size
                                # column, which the frontend fills with the
                                # per-layer, per-rank K+V bytes. It used to read
                                # output_memory_size, i.e. the whole QKV
                                # activation, which shipped Q as well and
                                # overstated the transfer by
                                # (q_dim + 2*kv_dim) / (2*kv_dim) -- 3x for
                                # Llama-3.1-8B -- and ignored kv_cache_dtype.
                                if "v_proj" in layers[layer_num].name and layers[layer_num].comm_size > 0:
                                    send_kv_node = self.get_comm_node(
                                        is_send=True,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].comm_size,
                                        comm_src=npu_id,
                                        comm_dst=npu_id + self.num_npus, # to the paired npu in decode
                                        id=layer_num
                                    )
                                    self.add_parent(send_kv_node, comp_node)
                                    encode_message(g, send_kv_node)

                                    recv_kv_node = self.get_comm_node(
                                        is_send=False,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].comm_size,
                                        comm_src=npu_id,
                                        comm_dst=npu_id + self.num_npus,
                                        id=layer_num
                                    )
                                    encode_message(s, recv_kv_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            # communication can happen even with one NPU in the group, for example, expert input gathering in data parallel
                            if expert_start == False and layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE": 
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                # End of expert, add ALLTOALL communication after expert computation
                                if layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    encode_message(g, comm_coll_node)
                                layer_num += 1
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1

                    # The frontend cuts stages on transformer-block boundaries,
                    # so the walk above lands exactly on the stage edge. Expert
                    # and PIM blocks live inside a block and can no longer be
                    # straddled; warn loudly if that ever stops holding.
                    if layer_num != layer_end:
                        print(f"Warning! pipeline stage {npu_group} walked to "
                              f"layer {layer_num}, expected {layer_end}")
                        layer_end = layer_num

                    if npu_group == (num_npu_group - 1):
                        # Send output (for the last layer, to the paired decode npu)
                        send_output_node = self.get_comm_node(
                            is_send=True,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
                        # paired decode npu receive output
                        recv_output_node = self.get_comm_node(
                            is_send=False,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        encode_message(s, recv_output_node)
                    else:
                        if layers[layer_end - 1].is_expert:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)

    def convert_event(self, f: TextIOWrapper, num_layers: int):
        layers: list[Layer] = self.get_layers(f)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for idx, layer in enumerate(layers):
                    comp_node = self.get_comp_node(
                    layer.name, 
                    layer.comp_time)
                    layer.comp_node = comp_node
                    encode_message(g, comp_node)

    # -- TOOLDAG -------------------------------------------------------------

    @staticmethod
    def parse_header_fields(fields: List[str]) -> Dict[str, str]:
        """Strict ``key: value`` pairs, for the TOOLDAG header.

        Stricter than the LLM header parsing on purpose: a stray token there
        is a formatting slip, here it could be a silently dropped VRAM budget.
        """
        if len(fields) % 2:
            raise ValueError(f"TOOLDAG header must be 'key: value' pairs, got {' '.join(fields)!r}")
        header = {}
        for i in range(0, len(fields), 2):
            if not fields[i].endswith(":") or len(fields[i]) == 1:
                raise ValueError(f"TOOLDAG header expected 'key:' but found {fields[i]!r}")
            header[fields[i][:-1]] = fields[i + 1]
        return header

    @staticmethod
    def tool_dag_options(header: Dict[str, str]) -> Dict[str, Any]:
        """Typed convert_tool_dag keyword arguments from a TOOLDAG header."""
        unknown = sorted(set(header) - set(TOOL_DAG_HEADER_KEYS))
        if unknown:
            raise ValueError(f"unknown TOOLDAG header key(s) {', '.join(unknown)}; "
                             f"expected {', '.join(TOOL_DAG_HEADER_KEYS)}")
        return {key: TOOL_DAG_HEADER_KEYS[key](value) for key, value in header.items()}

    @staticmethod
    def _toposort(parents: Dict[str, List[str]], row: Dict[str, int]) -> List[str]:
        # Kahn's algorithm. Among ready tasks the earliest row goes first, so
        # node ids -- and with them ASTRA-Sim's smallest-id-first issue order
        # among ready nodes -- follow the trace.
        children = {x: [] for x in parents}
        indegree = {x: len(ps) for x, ps in parents.items()}
        for x, ps in parents.items():
            for p in ps:
                children[p].append(x)
        ready = [(row[x], x) for x in parents if indegree[x] == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            _, x = heapq.heappop(ready)
            order.append(x)
            for c in children[x]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    heapq.heappush(ready, (row[c], c))
        if len(order) != len(parents):
            stuck = sorted((x for x in parents if indegree[x] > 0), key=row.get)
            raise ValueError(f"TOOLDAG dependencies form a cycle through {', '.join(stuck)}")
        return order

    @staticmethod
    def _descendants(parents: Dict[str, List[str]], order: List[str]) -> Dict[str, set]:
        children = {x: [] for x in order}
        for x in order:
            for p in parents[x]:
                children[p].append(x)
        desc = {}
        for x in reversed(order):
            reach = set()
            for c in children[x]:
                reach.add(c)
                reach |= desc[c]
            desc[x] = reach
        return desc

    @staticmethod
    def _max_weight_antichain(candidates: List[str], weight: Dict[str, float],
                              desc: Dict[str, set]):
        """Heaviest set of mutually unordered tasks: the worst co-residency.

        Exact branch and bound. The bound (weight not yet considered) prunes
        every other branch once the first one has taken an antichain covering
        all candidates, so a fan-out costs one pass; tool DAGs are small.
        """
        order = sorted(candidates, key=lambda x: -weight[x])
        suffix = [0.0] * (len(order) + 1)
        for i in range(len(order) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + weight[order[i]]
        best = [0.0, []]
        chosen = []

        def search(i: int, total: float) -> None:
            if total > best[0]:
                best[0], best[1] = total, list(chosen)
            if i == len(order) or total + suffix[i] <= best[0]:
                return
            x = order[i]
            if all(x not in desc[c] and c not in desc[x] for c in chosen):
                chosen.append(x)
                search(i + 1, total + weight[x])
                chosen.pop()
            search(i + 1, total)

        search(0, 0.0)
        return best[0], best[1]

    def analyze_tool_dag(self, tasks: List[ToolTask], vram_budget_mb: float = 0.0) -> ToolDagPlan:
        """Validate a mapping and resolve its precedence and VRAM peak.

        The queue follows the paper's two-phase timeline: the immediate phase
        is every gpu_now task that does not itself wait on a queued task, and
        queued tasks start once that phase is over, one at a time, in
        topological order. Residency follows from that precedence: two tasks
        can hold VRAM together only if neither is ordered before the other,
        so the peak is the heaviest such set over the gpu_now and gpu_queue
        tasks -- for a fan-out, the plain sum of the gpu_now tasks, which is
        the paper's constraint. The queue chain keeps any two queued tasks
        ordered, so at most one of them is ever counted.
        """
        by_id: Dict[str, ToolTask] = {}
        for t in tasks:
            if t.task in by_id:
                raise ValueError(f"duplicate TOOLDAG task id '{t.task}'")
            by_id[t.task] = t
        if not by_id:
            raise ValueError("a TOOLDAG trace needs at least one task")
        for t in tasks:
            for d in t.deps:
                if d not in by_id:
                    raise ValueError(f"task '{t.task}' depends on unknown task '{d}'")
        row = {t.task: i for i, t in enumerate(tasks)}
        dag_parents = {t.task: list(dict.fromkeys(t.deps)) for t in tasks}
        dag_order = self._toposort(dag_parents, row)
        desc = self._descendants(dag_parents, dag_order)

        queued = [x for x in dag_order if by_id[x].placement == "gpu_queue"]
        waits_on_queue = set()
        for q in queued:
            waits_on_queue |= desc[q]
        immediate = [x for x in dag_order
                     if by_id[x].placement == "gpu_now" and x not in waits_on_queue]
        aug_parents = {x: list(dag_parents[x]) for x in dag_order}
        previous = None
        for q in queued:
            extra = [p for p in immediate if q not in desc[p]]
            if previous is not None and q not in desc[previous]:
                extra.append(previous)
            aug_parents[q] += [p for p in extra if p not in aug_parents[q]]
            previous = q
        order = self._toposort(aug_parents, row)
        aug_desc = self._descendants(aug_parents, order)
        level: Dict[str, int] = {}
        for x in order:
            level[x] = 1 + max((level[p] for p in aug_parents[x]), default=-1)

        resident = [x for x in order
                    if by_id[x].placement in ("gpu_now", "gpu_queue") and by_id[x].vram_mb > 0]
        peak, members = self._max_weight_antichain(
            resident, {x: by_id[x].vram_mb for x in resident}, aug_desc)
        members.sort(key=row.get)
        return ToolDagPlan(by_id, dag_parents, aug_parents, order, level, peak, members,
                           float(vram_budget_mb or 0.0))

    def convert_tool_dag(self, tasks: List[ToolTask], vram_budget_mb: float = 0.0,
                         time_units_per_ms: float = 1000.0, contention: str = "simulator",
                         slice_ms: float = 0.0, cpu_flag: str = "bool") -> Dict[str, Any]:
        """Write one ET per NPU for a tool DAG under a fixed device mapping.

        Every NPU file carries the same graph -- an independent replica, as
        in convert_event -- so any network configuration runs it and each sys
        reports the same finish time.

        Nodes, in topological order of the queue-augmented graph:

          cpu        COMP_NODE_TOOL_<task>_<tool>_CPU   is_cpu_op=1, cpu_ms
          gpu_now    COMP_NODE_TOOL_<task>_<tool>_GPU   is_cpu_op=0, gpu_ms
          gpu_queue  COMP_NODE_TOOL_<task>_<tool>_LOAD  is_cpu_op=0, load_ms
                     COMP_NODE_TOOL_<task>_<tool>_GPU   is_cpu_op=0, gpu_ms

        A task's first node depends on the last node of each of its
        queue-augmented parents (see analyze_tool_dag). The load occupies the
        GPU resource because the queue is serialized on the GPU: a queued
        tool is admitted, loads, computes, and only then is the next one
        admitted.

        ``time_units_per_ms`` converts milliseconds into duration_micros
        units. Chakra defines that field in microseconds (1000 per ms), which
        current ASTRA-Sim scales to its nanosecond tick; a build that takes
        it as raw ticks needs 1e6. Durations are rounded to whole units, at
        least one, so no node is ever skipped as empty.

        ``contention`` "simulator" leaves co-placed tools to ASTRA-Sim's
        per-NPU CPU and GPU resources (one in-flight op each in stock
        ASTRA-Sim). "serialize" adds edges chaining each resource's nodes in
        emission order, for a simulator that would otherwise run compute
        nodes concurrently; on the paper's diamond and fan-out topologies the
        two give the same schedule.

        ``slice_ms`` > 0 splits every phase into chained slices of about that
        length, interleaved round-robin across the tasks of a level, so tools
        sharing a resource progress together -- processor sharing, as
        concurrent kernels do -- instead of one after another. E2E is
        unchanged, the resources stay busy either way, but per-tool latencies
        then look like shared execution.

        ``cpu_flag`` writes is_cpu_op as bool_val ("bool", as the LLM paths
        do) or int32_val ("int32") for a feeder that reads it that way.

        Raises VramBudgetExceeded, before writing anything, when the mapping
        does not fit ``vram_budget_mb`` (0 = unconstrained). Returns a summary
        holding the files, the VRAM peak and every node's id, task and phase:
        what a caller needs to map the simulator's per-node log back to tools.
        """
        if self.num_npus < 1:
            raise ValueError(f"num_npus must be >= 1, got {self.num_npus}")
        if not time_units_per_ms > 0:
            raise ValueError(f"time_units_per_ms must be > 0, got {time_units_per_ms}")
        if contention not in ("simulator", "serialize"):
            raise ValueError(f"contention must be 'simulator' or 'serialize', got {contention!r}")
        if not slice_ms >= 0:
            raise ValueError(f"slice_ms must be >= 0, got {slice_ms}")
        if cpu_flag not in ("bool", "int32"):
            raise ValueError(f"cpu_flag must be 'bool' or 'int32', got {cpu_flag!r}")
        plan = self.analyze_tool_dag(tasks, vram_budget_mb)
        if not plan.feasible:
            raise VramBudgetExceeded(plan.peak_vram_mb, plan.vram_budget_mb, plan.peak_tasks)

        # Each task's node sequence as (phase, resource, units, slice, slices).
        sequences = {}
        for x in plan.order:
            t = plan.by_id[x]
            if t.placement == "cpu":
                phases = [("CPU", "cpu", t.cpu_ms)]
            elif t.placement == "gpu_now":
                phases = [("GPU", "gpu", t.gpu_ms)]
            else:
                phases = [("LOAD", "gpu", t.load_ms)] if t.load_ms > 0 else []
                phases.append(("GPU", "gpu", t.gpu_ms))
            seq = []
            for phase, resource, ms in phases:
                units = max(1, int(round(ms * time_units_per_ms)))
                n = 1
                if slice_ms > 0:
                    n = max(1, min(units, int(math.ceil(ms / slice_ms - 1e-9))))
                base, extra = divmod(units, n)
                for k in range(n):
                    seq.append((phase, resource, base + (1 if k < extra else 0), k, n))
            sequences[x] = seq

        # Emit level by level, round-robin over the tasks of a level. Every
        # parent sits on a lower level, so parents are always written before
        # their children; without slicing this is just the topological order.
        levels: Dict[int, List[str]] = {}
        for x in plan.order:
            levels.setdefault(plan.level[x], []).append(x)
        nodes, info = [], []
        first, last = {}, {}
        resource_last = {"cpu": None, "gpu": None}
        for lvl in sorted(levels):
            cursor = {x: 0 for x in levels[lvl]}
            active = list(levels[lvl])
            while active:
                pending = []
                for x in active:
                    t = plan.by_id[x]
                    phase, resource, units, k, n = sequences[x][cursor[x]]
                    name = f"TOOL_{x}_{t.tool}_{phase}" + (f"_s{k + 1}of{n}" if n > 1 else "")
                    node = self._tool_comp_node(name, units, resource == "cpu", cpu_flag)
                    if cursor[x] == 0:
                        deps = [last[p] for p in plan.aug_parents[x]]
                        first[x] = node.id
                    else:
                        deps = [last[x]]
                    if (contention == "serialize" and resource_last[resource] is not None
                            and resource_last[resource] not in deps):
                        deps.append(resource_last[resource])
                    for d in deps:
                        node.data_deps.append(d)
                    resource_last[resource] = node.id
                    last[x] = node.id
                    nodes.append(node)
                    info.append({"id": node.id, "name": node.name, "task": x, "tool": t.tool,
                                 "phase": phase, "resource": resource, "units": units,
                                 "slice": k + 1, "slices": n, "deps": list(deps)})
                    cursor[x] += 1
                    if cursor[x] < len(sequences[x]):
                        pending.append(x)
                active = pending

        files = []
        for i in range(self.num_npus):
            npu_id = i + self.npu_offset
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                encode_message(g, self.get_global_metadata())
                for node in nodes:
                    encode_message(g, node)
            files.append(output_filename)

        return {
            "files": files,
            "peak_vram_mb": plan.peak_vram_mb,
            "peak_tasks": plan.peak_tasks,
            "vram_budget_mb": plan.vram_budget_mb,
            "time_units_per_ms": time_units_per_ms,
            "nodes": info,
            "first_node": first,
            "last_node": last,
            "dag_parents": plan.dag_parents,
            "aug_parents": plan.aug_parents,
        }

    def _tool_comp_node(self, name: str, units: int, is_cpu: bool, cpu_flag: str) -> Any:
        if cpu_flag == "bool":
            return self.get_comp_node(name, units, is_cpu=is_cpu)
        node = self.get_node("COMP_NODE_" + name, COMP_NODE)
        node.duration_micros = units
        node.attr.append(ChakraAttr(name="is_cpu_op", int32_val=int(is_cpu)))
        return node

    def convert(self):
        with open(self.input_filename, "r") as f:
            first_line = f.readline().strip().split()
            execution_type = first_line[0]

            # The first line carries "key: value" pairs after the execution
            # type, e.g. "model_parallel_NPU_group: 4  pp_stage_boundaries: 73,145,217".
            header = {}
            fields = first_line[1:]
            for i in range(0, len(fields) - 1, 2):
                if fields[i].endswith(":"):
                    header[fields[i][:-1]] = fields[i + 1]

            num_npu_group = int(header.get("model_parallel_NPU_group", 0))
            boundary_str = header.get("pp_stage_boundaries", "")
            stage_boundaries = (
                [int(b) for b in boundary_str.split(",")] if boundary_str else []
            )

            second_line = f.readline().strip()
            num_layers = int(second_line)

            third_line = f.readline() # This is for the table header, so just ignore it

            if execution_type == "COLOCATED":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_common(f, num_layers, num_npu_group, stage_boundaries)
            elif execution_type == "PREFILL":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_prefill(f, num_layers, num_npu_group, stage_boundaries)
            elif execution_type == "DECODE":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_common(f, num_layers, num_npu_group, stage_boundaries)
            elif execution_type == "EVENT":
                self.convert_event(f, num_layers)
            elif execution_type == "TOOLDAG":
                tasks = [ToolTask(line) for line in f if line.strip()]
                if len(tasks) != num_layers:
                    raise ValueError(f"TOOLDAG trace declares {num_layers} tasks but carries "
                                     f"{len(tasks)} rows")
                return self.convert_tool_dag(
                    tasks, **self.tool_dag_options(self.parse_header_fields(first_line[1:])))
            else:
                raise ValueError(f"Unsupported execution type, {execution_type}")
