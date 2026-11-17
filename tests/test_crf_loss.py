"""The vectorised CRF log-likelihood matches torchcrf's, in a few ops."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torchcrf import CRF
from transformers import DebertaV2Config

from model.src.model import PIIDetectionModel, crf_log_likelihood

NUM_TAGS = 53


def _crf(scale: float = 1.0) -> CRF:
    torch.manual_seed(0)
    crf = CRF(NUM_TAGS, batch_first=True)
    for parameter in crf.parameters():
        torch.nn.init.normal_(parameter, std=scale)
    return crf


def _batch(seq_len: int, scale: float = 1.0, *, gaps: bool = True):
    """Emissions, tags and a right-padded mask, optionally with gaps."""
    generator = torch.Generator().manual_seed(seq_len)
    batch_size = 6
    emissions = scale * torch.randn(batch_size, seq_len, NUM_TAGS, generator=generator)
    tags = torch.randint(0, NUM_TAGS, (batch_size, seq_len), generator=generator)
    lengths = torch.randint(1, seq_len + 1, (batch_size,), generator=generator)
    lengths[0] = seq_len  # one unpadded row
    lengths[1] = 1  # one row with only the first position
    mask = torch.arange(seq_len)[None, :] < lengths[:, None]
    if gaps:  # positions labelled -100 inside a sequence (special tokens)
        mask &= torch.rand(batch_size, seq_len, generator=generator) > 0.2
    mask[:, 0] = True
    return emissions, tags, mask


def _grads(output: torch.Tensor, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    return list(
        torch.autograd.grad(
            output.sum(), inputs, allow_unused=True, materialize_grads=True
        )
    )


def _float64_reference(crf: CRF, emissions, tags, mask):
    """torchcrf's log-likelihood and gradients, computed in float64."""
    crf64 = copy.deepcopy(crf).double()
    emissions64 = emissions.detach().double().requires_grad_()
    expected = crf64(emissions64, tags, mask=mask, reduction="none")
    return expected, _grads(expected, [emissions64, *crf64.parameters()])


@pytest.mark.parametrize("seq_len", [1, 2, 3, 17, 64, 255, 384])
@pytest.mark.parametrize("gaps", [False, True])
def test_matches_torchcrf_with_padding(seq_len, gaps):
    crf = _crf()
    emissions, tags, mask = _batch(seq_len, scale=3.0, gaps=gaps)
    emissions.requires_grad_()
    inputs = [emissions, *crf.parameters()]

    actual = crf_log_likelihood(crf, emissions, tags, mask)
    torch.testing.assert_close(
        actual, crf(emissions, tags, mask=mask, reduction="none"), rtol=1e-5, atol=1e-3
    )

    # torchcrf's float32 loop drifts by up to ~3e-4 in the gradient over 384
    # tokens; this is within ~1e-6 of float64, so compare against float64.
    expected, expected_grads = _float64_reference(crf, emissions, tags, mask)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-3)
    for got, want in zip(_grads(actual, inputs), expected_grads, strict=True):
        torch.testing.assert_close(got.double(), want, rtol=1e-4, atol=1e-5)


def test_bf16_emissions_match_torchcrf():
    crf = _crf()
    emissions, tags, mask = _batch(256, scale=3.0)
    emissions = emissions.bfloat16().requires_grad_()
    inputs = [emissions, *crf.parameters()]

    actual = crf_log_likelihood(crf, emissions, tags, mask)
    assert actual.dtype == torch.float32  # as torchcrf's, by type promotion
    torch.testing.assert_close(
        actual, crf(emissions, tags, mask=mask, reduction="none"), rtol=1e-5, atol=1e-3
    )

    expected, expected_grads = _float64_reference(crf, emissions, tags, mask)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-3)
    got_grads = _grads(actual, inputs)
    assert got_grads[0].dtype == torch.bfloat16
    for got, want in zip(got_grads, expected_grads, strict=True):
        # The emission gradient is rounded back to bf16 (8 significant bits).
        torch.testing.assert_close(got.double(), want, rtol=1e-2, atol=1e-5)


def test_bf16_autocast_keeps_float32_math():
    """Under bf16 mixed precision the matmuls must not drop to bf16."""
    crf = _crf()
    emissions, tags, mask = _batch(256, scale=3.0)
    emissions = emissions.bfloat16()
    expected = crf(emissions, tags, mask=mask, reduction="none")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = crf_log_likelihood(crf, emissions, tags, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-3)


class _MatmulPrecisions(TorchDispatchMode):
    """Records the float32 matmul precision in force at each matmul."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket.__name__ in {"bmm", "mm", "matmul", "mv"}:
            self.seen.add(torch.get_float32_matmul_precision())
        return func(*args, **(kwargs or {}))


def test_matmuls_run_at_highest_precision_in_both_passes():
    """TorchTPU defaults to "medium" (bf16 passes); the CRF must not use it."""
    crf = _crf()
    emissions, tags, mask = _batch(64)
    emissions.requires_grad_()
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        with _MatmulPrecisions() as forward:
            loss = crf_log_likelihood(crf, emissions, tags, mask).mean()
        with _MatmulPrecisions() as backward:
            loss.backward()
        assert torch.get_float32_matmul_precision() == "medium"  # restored
    finally:
        torch.set_float32_matmul_precision(previous)
    assert forward.seen == {"highest"}
    assert backward.seen == {"highest"}


def test_large_scores_stay_finite_and_match():
    """Trained CRFs have large logits and transitions; nothing may overflow."""
    crf = _crf(scale=5.0)
    emissions, tags, mask = _batch(384, scale=20.0)
    emissions.requires_grad_()
    expected = crf(emissions, tags, mask=mask, reduction="none")
    actual = crf_log_likelihood(crf, emissions, tags, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-2)
    (gradient,) = _grads(actual, [emissions])
    assert torch.isfinite(gradient).all()


def test_model_loss_matches_torchcrf_mean():
    """The training loss is still -crf(..., reduction="mean") on safe labels."""
    torch.manual_seed(0)
    config = DebertaV2Config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = PIIDetectionModel(
        "unused", NUM_TAGS, {i: str(i) for i in range(NUM_TAGS)}, encoder_config=config
    ).eval()
    input_ids = torch.randint(0, 100, (4, 20))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, NUM_TAGS, (4, 20))
    labels[:, 0] = -100  # [CLS]
    labels[0, 12:] = -100  # [SEP] and padding
    labels[2, 5] = -100  # a token without characters

    outputs = model(input_ids, attention_mask, pii_labels=labels)

    mask = labels != -100
    mask[:, 0] = True
    expected = -model.crf(
        outputs["pii_logits"], labels.clamp(min=0), mask=mask, reduction="mean"
    )
    torch.testing.assert_close(outputs["crf_loss"], expected)


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


def _ops_per_step(loss_fn, seq_len: int) -> int:
    crf = _crf()
    emissions, tags, mask = _batch(seq_len)
    emissions.requires_grad_()
    with _OpCounter() as counter:
        loss_fn(crf, emissions, tags, mask).mean().backward()
    return counter.count


def test_op_count_does_not_grow_with_sequence_length():
    """torchcrf issues ~20 ops per token; this must stay near-constant."""

    def ours(crf, emissions, tags, mask):
        return crf_log_likelihood(crf, emissions, tags, mask)

    def torchcrf(crf, emissions, tags, mask):
        return crf(emissions, tags, mask=mask, reduction="none")

    short, long = _ops_per_step(ours, 64), _ops_per_step(ours, 512)
    assert long - short <= 120  # three more pairwise levels, about 35 ops each
    assert long < 600
    assert _ops_per_step(torchcrf, 512) > 20 * long
