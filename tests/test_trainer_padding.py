from __future__ import annotations

import pytest

from model.src.trainer import bucketed_length


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, 128), (127, 128), (128, 128), (129, 256), (300, 384), (511, 512), (512, 512)],
)
def test_bucketed_length_rounds_up_to_bucket(length, expected):
    assert bucketed_length(length, 128, 512) == expected


def test_bucketed_length_caps_at_max_length():
    assert bucketed_length(450, 128, 480) == 480


def test_bucketed_length_never_shortens_a_batch():
    assert bucketed_length(600, 128, 512) == 600
