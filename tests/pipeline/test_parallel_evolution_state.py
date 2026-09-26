"""Parallel transport regressions, without LLM calls or market data.

Load the actual mining module with its external collaborators replaced. The
spawn tests use real OS processes/queues and a fresh module in each child;
only the expensive research loop, controller and import-time services are
stand-ins. No production scheduling or serialization functions are copied.
"""

import importlib.util
import multiprocessing
import os
import pickle
import queue
import sys
import threading
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


class Phase(str, Enum):
    ORIGINAL = "original"
    MUTATION = "mutation"
    CROSSOVER = "crossover"


QUALITY_GATE = {
    "consistency_enabled": True,
    "complexity_enabled": False,
    "redundancy_enabled": False,
    "max_correction_attempts": 7,
}


def make_task(phase=Phase.CROSSOVER, direction_id=0):
    count = {Phase.ORIGINAL: 0, Phase.MUTATION: 1, Phase.CROSSOVER: 2}[phase]
    return {
        "phase": phase,
        "direction_id": direction_id,
        "round_idx": 2,
        "strategy_suffix": "keep this guidance",
        # Locks deliberately cannot cross a spawn boundary: only IDs should.
        "parent_trajectories": [
            SimpleNamespace(trajectory_id=f"parent-{i}", lock=threading.Lock())
            for i in range(count)
        ],
    }


class RecordingLoop:
    instances = []

    def __init__(self, settings, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def run(self, step_n, stop_event):
        self.steps = step_n

    def _get_trajectory_data(self):
        return {"hypothesis": {
            "parents": self.kwargs.get("parent_trajectory_ids", []),
            "quality_gate": self.kwargs["quality_gate_config"],
            "direction": self.kwargs["potential_direction"],
            "guidance": self.kwargs.get("strategy_suffix"),
            "user_direction": self.user_initial_direction,
            "pid": os.getpid(),
            "steps": self.steps,
        }}


class RecordingController:
    instances = []

    def __init__(self, config):
        self.tasks = [make_task(direction_id=i) for i in range(2)]
        self.results = []
        self.instances.append(self)
        self.pool = SimpleNamespace(get_statistics=lambda: {})

    def is_complete(self):
        return len(self.results) == len(self.tasks)

    def get_all_tasks_for_current_phase(self):
        return self.tasks

    def get_next_task(self):
        return self.tasks[len(self.results)]

    def create_trajectory_from_loop_result(self, task, hypothesis, **kwargs):
        self.results.append((task, hypothesis))
        return SimpleNamespace(trajectory_id="child", get_primary_metric=lambda: 0.1)

    def report_task_complete(self, task, trajectory):
        pass

    def advance_phase_after_parallel_completion(self, tasks):
        assert len(tasks) == len(self.tasks)

    def save_state(self, path):
        pass

    def get_best_trajectories(self, top_n):
        return []


@contextmanager
def load_mining(log_root):
    """Isolate import-time dependencies without changing the mining code."""
    services = {
        "fire": {},
        "quantaalpha.pipeline.settings": {"ALPHA_AGENT_FACTOR_PROP_SETTING": None},
        "quantaalpha.pipeline.planning": {
            "generate_parallel_directions": lambda **kw: ["direction-0", "direction-1"],
            "load_run_config": lambda path: {},
        },
        "quantaalpha.pipeline.loop": {"AlphaAgentLoop": RecordingLoop},
        "quantaalpha.pipeline.evolution": {
            "EvolutionController": RecordingController,
            "EvolutionConfig": SimpleNamespace,
            "StrategyTrajectory": SimpleNamespace(generate_id=lambda *args: "child"),
            "RoundPhase": Phase,
        },
        "quantaalpha.core.exception": {"FactorEmptyError": RuntimeError},
        "quantaalpha.core.conf": {"RD_AGENT_SETTINGS": SimpleNamespace(
            workspace_path=log_root, pickle_cache_folder_path_str=str(log_root),
            cache_with_pickle=False, use_file_lock=True,
        )},
        "quantaalpha.log": {"logger": Mock(log_trace_path=log_root)},
        "quantaalpha.log.time": {"measure_time": lambda fn: fn},
        "quantaalpha.llm.config": {
            "LLM_SETTINGS": SimpleNamespace(factor_mining_timeout=30),
        },
    }
    modules = {}
    for name, attrs in services.items():
        modules[name] = ModuleType(name)
        vars(modules[name]).update(attrs)
    path = Path(__file__).resolve().parents[2] / "quantaalpha/pipeline/factor_mining.py"
    spec = importlib.util.spec_from_file_location("_parallel_mining_test", path)
    mining = importlib.util.module_from_spec(spec)
    modules[spec.name] = mining
    # Restore only these entries, not modules lazily imported by multiprocessing.
    with pytest.MonkeyPatch.context() as imports:
        for name, module in modules.items():
            imports.setitem(sys.modules, name, module)
        spec.loader.exec_module(mining)
        yield mining


class InlineProcess:
    """Exercise the actual production target with its actual args/kwargs."""
    def __init__(self, target, args=(), kwargs=None):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)

    def join(self):
        pass


@pytest.fixture
def mining(tmp_path, monkeypatch):
    RecordingLoop.instances = []
    RecordingController.instances = []
    with load_mining(tmp_path) as module:
        monkeypatch.setattr(module, "Process", InlineProcess)
        monkeypatch.setattr(module, "Queue", queue.Queue)
        yield module


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.parametrize("serialized", [False, True], ids=["sequential", "serialized"])
def test_parent_ids_reach_loop(mining, phase, serialized):
    task = make_task(phase)
    parents = task["parent_trajectories"]
    expected = [p.trajectory_id for p in parents]
    if serialized:
        task = pickle.loads(pickle.dumps(mining._serialize_task_for_parallel(task)))
        assert task["parent_trajectories"] == []
        assert task["parent_trajectory_ids"] == expected
        assert [p.trajectory_id for p in parents] == expected
    result = mining._run_evolution_task(task, ["direction-0"], 5, True, "research", "", None)
    assert result["hypothesis"]["parents"] == expected
    assert result["hypothesis"]["guidance"] == "keep this guidance"
    assert result["hypothesis"]["user_direction"] == "research"
    assert result["hypothesis"]["direction"] == (None if phase == Phase.CROSSOVER else "direction-0")


def run_evolution(mining, parallel, config):
    mining.run_evolution_loop(
        initial_direction="research", evolution_cfg={"parallel_enabled": parallel},
        exec_cfg={"steps_per_loop": 5}, planning_cfg={}, quality_gate_cfg=config,
    )
    return RecordingController.instances[-1]


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
@pytest.mark.parametrize("config", [None, {}, QUALITY_GATE], ids=["default", "empty", "custom"])
def test_evolution_configuration_reaches_every_loop(mining, parallel, config):
    before = deepcopy(config)
    controller = run_evolution(mining, parallel, config)
    assert len(controller.results) == 2
    for task, result in controller.results:
        assert result["quality_gate"] == (config or {})
        assert result["parents"] == [p.trajectory_id for p in task["parent_trajectories"]]
        assert result["steps"] == 5
    assert config == before


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_main_planning_configuration_reaches_every_branch(mining, monkeypatch, parallel):
    config = {
        "planning": {"enabled": True, "num_directions": 2},
        "execution": {"parallel_execution": parallel, "step_n": 5},
        "quality_gate": deepcopy(QUALITY_GATE),
    }
    monkeypatch.setattr(mining, "load_run_config", lambda path: config)
    # Skip only the wall-clock/signal decorator, not the CLI routing logic.
    mining.main.__wrapped__(direction="research", evolution_mode=False)
    assert len(RecordingLoop.instances) == 2
    for loop in RecordingLoop.instances:
        assert loop.kwargs["quality_gate_config"] == QUALITY_GATE
        assert loop.steps == 5


def test_worker_default_and_failure_payload(mining, monkeypatch):
    result_queue = queue.Queue()
    args = (make_task(Phase.ORIGINAL), ["direction-0"], 5, True, "research", "", result_queue, 3)
    mining._parallel_task_worker(*args)
    result = result_queue.get_nowait()
    assert result["success"] is True
    assert result["traj_data"]["hypothesis"]["quality_gate"] == {}
    monkeypatch.setattr(mining, "AlphaAgentLoop", Mock(side_effect=ValueError("test failure")))
    mining._parallel_task_worker(*args)
    result = result_queue.get_nowait()
    assert result["success"] is False
    assert result["task_idx"] == 3
    assert result["error"] == "test failure"


def test_empty_parallel_batch_starts_nothing(mining, monkeypatch):
    factory = Mock(side_effect=AssertionError("No process or queue should be created"))
    monkeypatch.setattr(mining, "Process", factory)
    monkeypatch.setattr(mining, "Queue", factory)
    assert mining._run_tasks_parallel([], [], 5, True, None, "") == []


def spawn_worker(*args, **kwargs):
    """Install dependency stand-ins after spawning, then run the real worker."""
    with load_mining(Path(args[5])) as module:
        module._parallel_task_worker(*args, **kwargs)


class BoundedQueue:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def put(self, value):
        self.wrapped.put(value)

    def get(self):
        return self.wrapped.get(timeout=15)


@pytest.mark.parametrize("config", [{}, QUALITY_GATE], ids=["empty", "custom"])
def test_real_spawn_transport_preserves_state(mining, monkeypatch, config):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []

    def process_factory(target, args=(), kwargs=None):
        assert target is mining._parallel_task_worker
        process = context.Process(target=spawn_worker, args=args, kwargs=kwargs or {})
        processes.append(process)

        def join():
            process.join(timeout=15)
            assert not process.is_alive(), "Worker failed to exit"
            assert process.exitcode == 0

        return SimpleNamespace(start=process.start, join=join)

    monkeypatch.setattr(mining, "Process", process_factory)
    monkeypatch.setattr(mining, "Queue", lambda: BoundedQueue(result_queue))
    try:
        controller = run_evolution(mining, True, config)
        assert len(processes) == len(controller.results) == 2
        for task, result in controller.results:
            assert result["pid"] != os.getpid()
            assert result["parents"] == [p.trajectory_id for p in task["parent_trajectories"]]
            assert result["quality_gate"] == config
            assert result["guidance"] == "keep this guidance"
            assert result["user_direction"] == "research"
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
