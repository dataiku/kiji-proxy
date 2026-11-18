"""Eval scripts rebuild the encoder from the saved config, not a per-type default."""

from __future__ import annotations

import json

import pytest
import torch
from transformers import DebertaV2Config

from model.src import compare_models, eval_model_detailed
from model.src.model import PIIDetectionModel

# Deliberately not the 128,100-token vocabulary of microsoft/deberta-v3-base.
VOCAB_SIZE = 300
LABELS = ["O", "B-EMAIL", "I-EMAIL"]


@pytest.fixture
def checkpoint(tmp_path):
    """A saved checkpoint whose config.json has no `_name_or_path`."""
    config = DebertaV2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    id2label = dict(enumerate(LABELS))
    model = PIIDetectionModel(
        model_name="unused",
        num_pii_labels=len(LABELS),
        id2label_pii=id2label,
        encoder_config=config,
    )
    config.save_pretrained(tmp_path)
    assert "_name_or_path" not in json.loads((tmp_path / "config.json").read_text())
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    mappings = {
        "pii": {
            "label2id": {label: i for i, label in id2label.items()},
            "id2label": {str(i): label for i, label in id2label.items()},
        }
    }
    (tmp_path / "label_mappings.json").write_text(json.dumps(mappings))
    return tmp_path


def _embedding_rows(model: PIIDetectionModel) -> int:
    return model.encoder.get_input_embeddings().weight.shape[0]


def test_eval_model_detailed_loads_saved_architecture(checkpoint, monkeypatch):
    monkeypatch.setattr(
        eval_model_detailed.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )
    loader = eval_model_detailed.DetailedPIIModelLoader(str(checkpoint))
    loader.load_model()
    assert _embedding_rows(loader.model) == VOCAB_SIZE


def test_compare_models_loads_saved_architecture(checkpoint, monkeypatch):
    monkeypatch.setattr(
        compare_models.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )
    model, _tokenizer, _labels = compare_models.load_pytorch_model(str(checkpoint))
    assert _embedding_rows(model) == VOCAB_SIZE
