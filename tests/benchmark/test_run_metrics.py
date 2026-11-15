"""Offline checks for the benchmark's scoring helpers (no Hub access, no model)."""

from tests.benchmark.run import (
    AnyLabelOverlapMetrics,
    RelaxedOverlapMetrics,
    drop_unscored,
)


def test_any_label_counts_a_mask_with_the_wrong_label():
    gold = [(0, 9, "NATIONALID"), (10, 14, "FIRSTNAME")]
    predicted = [(0, 9, "IDCARDNUM"), (20, 25, "SSN")]

    same_label = RelaxedOverlapMetrics()
    same_label.update(gold, predicted)
    assert sum(same_label.tp.values()) == 0

    any_label = AnyLabelOverlapMetrics()
    any_label.update(gold, predicted)
    summary = any_label.summary()
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["masked_share_per_gold_label"] == {
        "FIRSTNAME": 0.0,
        "NATIONALID": 1.0,
    }


def test_any_label_matches_one_to_one():
    # Two fragments of one entity: one is a match, the other a false positive.
    any_label = AnyLabelOverlapMetrics()
    any_label.update([(0, 19, "USERNAME")], [(0, 7, "FIRSTNAME"), (8, 19, "USERNAME")])
    assert sum(any_label.tp.values()) == 1
    assert sum(any_label.fp.values()) == 1


def test_drop_unscored_removes_overlapping_predictions_only():
    predicted = [(11, 19, "SECURITYTOKEN"), (23, 27, "USERNAME"), (30, 33, "ZIP")]
    kept, dropped = drop_unscored(predicted, [(11, 19), (31, 36)])
    assert kept == [(23, 27, "USERNAME")]
    assert dropped == 2
    assert drop_unscored(predicted, []) == (predicted, 0)
