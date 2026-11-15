"""CRF decoding runs on CPU and matches torchcrf's own decode."""

from __future__ import annotations

import pytest
import torch
from torchcrf import CRF

from model.src.model import viterbi_decode_on_cpu


def _batch(dtype=torch.float32):
    torch.manual_seed(0)
    crf = CRF(53, batch_first=True)
    torch.nn.init.normal_(crf.transitions)
    emissions = torch.randn(8, 40, 53).to(dtype)
    lengths = torch.tensor([40, 1, 7, 33, 40, 12, 2, 25])
    mask = torch.arange(40)[None, :] < lengths[:, None]
    return crf, emissions, mask.long()


def test_matches_torchcrf_on_cpu():
    crf, emissions, mask = _batch()
    expected = crf.decode(emissions, mask=mask.bool())
    assert viterbi_decode_on_cpu(crf, emissions, mask) == expected
    assert [len(p) for p in expected] == mask.sum(1).tolist()


def test_bf16_emissions_decode_like_float32_ones():
    crf, emissions, mask = _batch(torch.bfloat16)
    expected = crf.decode(emissions.float(), mask=mask.bool())
    assert viterbi_decode_on_cpu(crf, emissions, mask) == expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_a_crf_on_an_accelerator_is_decoded_on_cpu():
    crf, emissions, mask = _batch()
    expected = crf.decode(emissions, mask=mask.bool())
    device = torch.device("mps")
    crf_on_device = crf.to(device)
    paths = viterbi_decode_on_cpu(crf_on_device, emissions.to(device), mask.to(device))
    assert paths == expected
    assert crf_on_device.transitions.device.type == "mps"  # model left in place
