from __future__ import annotations

import os

import pytest
from accelerate.state import PartialState
from accelerate.utils import DistributedType

from model.src.config import enable_tpu_data_parallel, is_distributed_launch


@pytest.fixture
def patched_partial_state(monkeypatch):
    # monkeypatch restores the original _prepare_backend after each test
    monkeypatch.setattr(PartialState, "_prepare_backend", PartialState._prepare_backend)
    monkeypatch.delenv("ACCELERATE_BYPASS_DEVICE_MAP", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    enable_tpu_data_parallel()
    return monkeypatch


def prepare_backend(cpu=False, backend=None):
    state = object.__new__(PartialState)
    return PartialState._prepare_backend(state, cpu, False, backend)


def test_distributed_launch_needs_world_size_and_local_rank(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert not is_distributed_launch()
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert is_distributed_launch()
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not is_distributed_launch()


def test_torchrun_launch_uses_tpu_dist_ddp(patched_partial_state):
    assert prepare_backend() == ("tpu_dist", DistributedType.MULTI_GPU)


def test_ddp_is_built_without_device_ids(patched_partial_state):
    assert os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] == "true"


def test_explicit_ddp_backend_is_kept(patched_partial_state):
    assert prepare_backend(backend="gloo") == ("gloo", DistributedType.MULTI_GPU)


def test_cpu_and_single_process_runs_keep_accelerate_defaults(patched_partial_state):
    backend, distributed_type = prepare_backend(cpu=True)
    assert distributed_type == DistributedType.MULTI_CPU

    patched_partial_state.delenv("LOCAL_RANK")
    patched_partial_state.setenv("WORLD_SIZE", "1")
    assert prepare_backend() == (None, DistributedType.NO)


def test_enabling_twice_does_not_stack_patches(patched_partial_state):
    patched = PartialState._prepare_backend
    enable_tpu_data_parallel()
    assert PartialState._prepare_backend is patched
