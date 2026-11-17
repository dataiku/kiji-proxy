from __future__ import annotations

import inspect

from model.src.model import PIIDetectionModel


def test_model_forward_takes_no_loss_kwargs():
    # The HF Trainer divides the loss by gradient_accumulation_steps only when
    # the model's forward() has no **kwargs. Adding **kwargs would silently make
    # accumulated gradients gradient_accumulation_steps times too large.
    params = inspect.signature(PIIDetectionModel.forward).parameters.values()
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params)
