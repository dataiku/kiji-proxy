"""Tests for the train/validation split in DatasetProcessor."""

from __future__ import annotations

from collections import Counter

from model.src.config import TrainingConfig
from model.src.preprocessing import DatasetProcessor


def _processor(tmp_path, **overrides) -> DatasetProcessor:
    processor = DatasetProcessor.__new__(DatasetProcessor)
    processor.config = TrainingConfig(output_dir=str(tmp_path), **overrides)
    # The split only needs each row's entity types; tests store them directly.
    processor._sample_entity_types = lambda sample: sample["types"]
    return processor


def _rows() -> tuple[list[dict], list[tuple]]:
    """900 rows: 3 sources x 2 languages, a third PII-free, one rare label."""
    rows, strata = [], []
    for i in range(900):
        source = ("kiji", "nym", "syn")[i % 3]
        language = ("German", "English")[(i // 3) % 2]
        pii_free = source == "syn" and i % 2 == 0
        types = set() if pii_free else {"FIRSTNAME"}
        if i % 150 == 1:  # 6 rows with a rare label
            types = types | {"IBAN"}
        rows.append({"id": i, "types": types})
        strata.append((source, language, pii_free))
    return rows, strata


def test_validation_keeps_the_mix_of_sources_languages_and_negatives(tmp_path):
    rows, strata = _rows()
    processor = _processor(tmp_path, eval_size_ratio=0.1, seed=42)
    train, val = processor._split_train_validation(rows, strata)

    assert len(train) + len(val) == len(rows)
    assert {r["id"] for r in train}.isdisjoint(r["id"] for r in val)
    all_groups = Counter(strata)
    val_groups = Counter(strata[r["id"]] for r in val)
    for group, count in all_groups.items():
        assert abs(val_groups[group] - count * 0.1) <= 3, group
    assert sum(1 for r in val if not r["types"]) >= 0.8 * 0.1 * sum(
        1 for r in rows if not r["types"]
    )


def test_rare_labels_are_topped_up_but_half_stays_in_training(tmp_path):
    rows, strata = _rows()
    processor = _processor(tmp_path, eval_size_ratio=0.1, seed=42)
    train, val = processor._split_train_validation(rows, strata)

    assert any("IBAN" in r["types"] for r in val)
    assert sum("IBAN" in r["types"] for r in train) >= 3


def test_split_is_deterministic_and_seed_dependent(tmp_path):
    rows, strata = _rows()
    first = _processor(tmp_path, seed=7)._split_train_validation(rows, strata)
    again = _processor(tmp_path, seed=7)._split_train_validation(rows, strata)
    other = _processor(tmp_path, seed=8)._split_train_validation(rows, strata)

    ids = [{r["id"] for r in part} for part in (first[1], again[1], other[1])]
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_eval_cap_moves_extra_rows_to_training(tmp_path):
    rows, strata = _rows()
    processor = _processor(tmp_path, eval_size_ratio=0.1, max_eval_samples=30, seed=42)
    train, val = processor._split_train_validation(rows, strata)

    # 30 stratified rows, plus at most a few for the rare label.
    assert 30 <= len(val) <= 33
    assert len(train) == len(rows) - len(val)


def test_without_balancing_the_split_is_only_stratified(tmp_path):
    rows, strata = _rows()
    processor = _processor(
        tmp_path, eval_size_ratio=0.1, balanced_validation_split=False, seed=42
    )
    train, val = processor._split_train_validation(rows, strata)

    assert len(val) == 90
    assert len(train) == 810


def test_template_groups_land_on_one_side(tmp_path):
    rows, strata = _rows()
    groups = [r["id"] // 5 for r in rows]  # 180 groups of 5 rows
    processor = _processor(tmp_path, eval_size_ratio=0.1, seed=42)
    train, val = processor._split_train_validation(rows, strata, groups)

    train_groups = {groups[r["id"]] for r in train}
    val_groups = {groups[r["id"]] for r in val}
    assert train_groups.isdisjoint(val_groups)
    assert 80 <= len(val) <= 110


def test_template_key_ignores_pii_values_and_digits():
    def sample(text: str, spans: list[tuple[int, int, str]]) -> dict:
        return {
            "text": text,
            "privacy_mask": [
                {"start": s, "end": e, "label": lab, "value": text[s:e]}
                for s, e, lab in spans
            ],
        }

    a = sample("Hallo Anna, Auftrag 1234.", [(6, 10, "FIRSTNAME")])
    b = sample("Hallo  Peter, Auftrag 9876.", [(7, 12, "FIRSTNAME")])
    c = sample("Hallo Anna, Rechnung 1234.", [(6, 10, "FIRSTNAME")])
    key = DatasetProcessor._template_key
    assert key(a) == key(b) == "hallo <firstname>, auftrag 0000."
    assert key(a) != key(c)


def test_a_big_template_group_does_not_overshoot_its_stratum(tmp_path):
    # 1,000 rows in one stratum: one template with 300 rows and 70 templates
    # with 10. Validation gets its 10%, never the 300-row group, whichever
    # group the shuffle puts first.
    rows = [{"id": i, "types": set()} for i in range(1000)]
    groups = ["big"] * 300 + [i // 10 for i in range(700)]
    for seed in range(200):
        processor = _processor(
            tmp_path, eval_size_ratio=0.1, balanced_validation_split=False, seed=seed
        )
        train, val = processor._split_train_validation(rows, None, groups)
        assert len(val) == 100, seed
        assert len(train) == 900, seed


def test_a_stratum_of_only_big_groups_still_gets_one_group(tmp_path):
    # Stratum "b" is one 40-row template; its 10% quota is 4 rows.
    rows = [{"id": i, "types": set()} for i in range(400)]
    strata = ["a"] * 360 + ["b"] * 40
    groups = list(range(360)) + ["b-template"] * 40
    processor = _processor(
        tmp_path, eval_size_ratio=0.1, balanced_validation_split=False, seed=1
    )
    train, val = processor._split_train_validation(rows, strata, groups)
    assert sum(1 for r in val if r["id"] >= 360) == 40
    assert len(train) + len(val) == 400
