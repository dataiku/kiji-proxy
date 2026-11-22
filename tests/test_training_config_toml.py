"""TrainingConfig.from_toml and the PII-free over-mask metric."""

from __future__ import annotations

import tomllib
from pathlib import Path

from model.src.config import TrainingConfig
from model.src.trainer import pii_free_overmask_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
MIX_V4 = REPO_ROOT / "model/flows/experiments/mdeberta_german_mix_v4.toml"


def test_mix_v4_toml_reaches_the_config(tmp_path):
    cfg = tomllib.loads(MIX_V4.read_text())
    cfg["paths"]["output_dir"] = str(tmp_path)
    config = TrainingConfig.from_toml(cfg, environ={})

    assert config.model_name == "microsoft/mdeberta-v3-base"
    assert config.batch_size == 32
    assert config.gradient_accumulation_steps == 1
    assert config.bf16 is True
    assert config.seed == 42
    assert config.max_eval_samples == 0
    assert config.early_stopping_threshold == 0.005
    assert config.static_padding is None  # on when a TPU is present
    assert config.training_samples_dir.endswith("german_mix_v4/samples")


def test_every_training_key_in_the_tomls_is_read(tmp_path):
    # A key the mapping ignores silently falls back to the dataclass default.
    for toml in [
        REPO_ROOT / "model/flows/training_config.toml",
        *sorted((REPO_ROOT / "model/flows/experiments").glob("*.toml")),
    ]:
        cfg = tomllib.loads(toml.read_text())
        cfg.setdefault("paths", {})["output_dir"] = str(tmp_path)
        config = TrainingConfig.from_toml(cfg, environ={})
        for key, value in cfg.get("training", {}).items():
            assert getattr(config, key) == value, (toml.name, key)


def test_environment_overrides_seed_and_ai4privacy(tmp_path):
    cfg = {"paths": {"output_dir": str(tmp_path)}, "training": {"seed": 42}}
    config = TrainingConfig.from_toml(
        cfg, environ={"TRAINING_SEED": "13", "NUM_AI4PRIVACY_SAMPLES": "5"}
    )
    assert config.seed == 13
    assert config.num_ai4privacy_samples == 5


def test_overmask_metrics_count_only_pii_free_rows():
    true = [["O", "O", "O"], ["O", "O"], ["B-FIRSTNAME", "O"], []]
    pred = [["O", "B-PASSWORD", "O"], ["O", "O"], ["O", "O"], []]
    metrics = pii_free_overmask_metrics(true, pred)
    assert metrics["eval_pii_free_rows"] == 2
    assert metrics["eval_pii_free_overmask_rate"] == 0.5
    assert metrics["eval_pii_free_masked_per_1k_tokens"] == 200.0
    assert pii_free_overmask_metrics([["B-CITY"]], [["B-CITY"]]) == {}


def test_model_initialization_is_seeded(tmp_path, monkeypatch):
    import torch

    from model.src import trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "PIIDetectionModel",
        lambda **kwargs: torch.nn.Linear(8, kwargs["num_pii_labels"]),
    )

    def init_weights(seed: int, process_state: int) -> torch.Tensor:
        pii_trainer = trainer_module.PIITrainer.__new__(trainer_module.PIITrainer)
        pii_trainer.config = TrainingConfig(
            output_dir=str(tmp_path), seed=seed, use_custom_loss=False
        )
        pii_trainer.pii_label2id = {"O": 0, "B-CITY": 1, "I-CITY": 2}
        pii_trainer.pii_id2label = {0: "O", 1: "B-CITY", 2: "I-CITY"}
        torch.manual_seed(process_state)  # whatever state the process is in
        pii_trainer.initialize_model()
        return pii_trainer.model.weight.detach().clone()

    assert torch.equal(init_weights(42, 1), init_weights(42, 2))
    assert not torch.equal(init_weights(42, 1), init_weights(13, 1))


def _load_train_tpu():
    import importlib.util

    path = REPO_ROOT / "model/flows/experiments/train_tpu.py"
    spec = importlib.util.spec_from_file_location("train_tpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_tpu_seed_flag_beats_training_seed_env(tmp_path, monkeypatch):
    import argparse

    train_tpu = _load_train_tpu()
    monkeypatch.setenv("TRAINING_SEED", "13")

    def args(seed):
        return argparse.Namespace(
            config=str(MIX_V4),
            output_dir=str(tmp_path),
            num_epochs=None,
            max_eval_samples=None,
            seed=seed,
        )

    assert train_tpu.build_config(args(7)).seed == 7
    # Without the flag the environment still overrides the file.
    assert train_tpu.build_config(args(None)).seed == 13
