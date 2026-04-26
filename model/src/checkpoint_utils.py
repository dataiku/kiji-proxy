"""Utilities for loading trained model checkpoints safely."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

STATE_DICT_PREFIXES = ("model.", "_orig_mod.", "module.")
CRITICAL_WEIGHT_PREFIXES = ("encoder.", "pii_classifier.", "crf.")


def normalize_state_dict_keys(
    state_dict: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Strip common wrapper prefixes from a model state dict."""
    normalized = dict(state_dict)

    changed = True
    while changed:
        changed = False
        for prefix in STATE_DICT_PREFIXES:
            prefixed = {
                key[len(prefix) :]: value
                for key, value in normalized.items()
                if key.startswith(prefix)
            }
            if prefixed and len(prefixed) >= len(normalized) / 2:
                normalized = prefixed
                changed = True
                break

    return normalized


def load_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    source: str,
) -> Any:
    """Load a checkpoint and fail if critical PII model weights are missing."""
    normalized = normalize_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(normalized, strict=False)

    critical_missing = [
        key
        for key in incompatible.missing_keys
        if key.startswith(CRITICAL_WEIGHT_PREFIXES)
    ]
    if critical_missing:
        missing_preview = ", ".join(critical_missing[:20])
        if len(critical_missing) > 20:
            missing_preview += ", ..."
        unexpected_preview = ", ".join(incompatible.unexpected_keys[:20])
        if len(incompatible.unexpected_keys) > 20:
            unexpected_preview += ", ..."
        detail = f" Missing critical weights: {missing_preview}."
        if unexpected_preview:
            detail += f" Unexpected checkpoint keys include: {unexpected_preview}."
        raise RuntimeError(
            f"Checkpoint {source} is incompatible with PIIDetectionModel.{detail}"
        )

    return incompatible
