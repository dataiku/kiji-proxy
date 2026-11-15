"""Build the training mix (German-focused, with multilingual negatives) as Label
Studio JSON files.

Sources, written to one directory the training pipeline reads:

1. DataikuNLP/kiji-pii-training-data (Apache-2.0), all languages, all splits,
   re-converted with the fixed converter (repeated mentions are labelled).
   With ``--kiji-dir``, the same rows are read from a cleaned Label Studio
   export instead (garbled characters repaired from the raw generator output),
   and go through the same converter.
2. German rows of Wismut/nym-pii-multilingual-data train (MIT).
3. German rows of ai4privacy/pii-masking-openpii-1m train (CC-BY-4.0), a
   DE/AT/CH-balanced subsample. Kept small (8,000 by default): its IDs often
   sit in odd contexts and teach "random code = ID", the over-masking this
   mix works against.
4. Template-generated negative and contrast rows
   (model/dataset/synthetic_negatives.py) in German, English, French,
   Spanish, Dutch and Danish, about half of them PII-free.

Not included on purpose: naeyn/nobody-pii-synth-de train, because its test
split is the main held-out German benchmark and shares templates with train.

Samples go to ``<output-dir>/samples/`` (point ``paths.training_samples_dir``
there); ``manifest.json`` (counts, labels, licences) and ``ATTRIBUTION.md`` go
to ``<output-dir>/``.

Usage:
    uv run python -m model.dataset.build_training_mix \
        --output-dir model/dataset/data_samples/german_mix_v3

    # v4: cleaned Kiji export; nym, openpii and template rows copied from v3
    uv run python -m model.dataset.build_training_mix \
        --output-dir model/dataset/data_samples/german_mix_v4 \
        --kiji-dir <cleaned export>/training_samples_clean \
        --reuse-from model/dataset/data_samples/german_mix_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from model.dataset.huggingface.german_sources import (
    NYM_REPO,
    OPENPII_REPO,
    convert_nym_row,
    convert_openpii_row,
)
from model.dataset.labelstudio.labelstudio_format import convert_to_labelstudio
from model.dataset.synthetic_negatives import LOCALES
from model.dataset.synthetic_negatives import generate as generate_synthetic

KIJI_REPO = "DataikuNLP/kiji-pii-training-data"

LICENCES = {
    "kiji": (KIJI_REPO, "Apache-2.0"),
    "nym": (NYM_REPO, "MIT"),
    "openpii": (OPENPII_REPO, "CC-BY-4.0, credit: Ai4Privacy / Ai Suisse SA"),
}
# File prefix per template language (synde_, synen_, ...).
SYNTHETIC_PREFIXES = {lang: f"syn{loc.code}" for lang, loc in LOCALES.items()}
for _prefix in SYNTHETIC_PREFIXES.values():
    LICENCES[_prefix] = (
        "model/dataset/synthetic_negatives.py",
        "generated in this repo",
    )

ATTRIBUTION = f"""# Attribution

This training set contains material from the following datasets.

- **{OPENPII_REPO}** - OpenPII 1M by Ai4Privacy / Ai Suisse SA,
  https://huggingface.co/datasets/{OPENPII_REPO}, licensed under CC-BY-4.0
  (https://creativecommons.org/licenses/by/4.0/). Changes: German rows only,
  labels mapped to the Kiji label set, DATE split into DATEOFBIRTH / unlabelled,
  rows with ambiguous birth dates removed.
- **{NYM_REPO}** - MIT licence, https://huggingface.co/datasets/{NYM_REPO}.
  Changes: German rows only, labels mapped to the Kiji label set, rows with
  labels Kiji does not have (customer/account IDs, CVV, ...) removed.
- **{KIJI_REPO}** - Apache-2.0.
"""


def _load(repo: str, **kwargs):
    from datasets import load_dataset

    try:
        return load_dataset(repo, **kwargs)
    except Exception:  # expired tokens: retry anonymously
        return load_dataset(repo, token=False, **kwargs)


def load_kiji() -> list[dict]:
    ds = _load(KIJI_REPO)
    rows = []
    for split in ds:
        for row in ds[split]:
            # About 4% of rows have non-ASCII letters replaced by NUL
            # ("Z\x00rich"); the original letter cannot be recovered.
            if "\x00" in row["text"]:
                continue
            rows.append(
                {
                    "text": row["text"],
                    "privacy_mask": row.get("privacy_mask") or [],
                    "coreferences": row.get("coreferences") or [],
                    "language": row.get("language"),
                    "country": row.get("country"),
                    "source": KIJI_REPO,
                }
            )
    return rows


# Coreference spans in Kiji's Label Studio exports. The trainer ignores them,
# and keeping them would stop repeated names from being labelled.
_COREFERENCE_LABELS = {"MENTION", "PRONOUN", "REFERENCE"}


def kiji_row_from_labelstudio(sample: dict) -> dict | None:
    """Turn an exported Label Studio sample back into a privacy-mask row.

    Only PII spans are kept, with their offsets, so ``convert_to_labelstudio``
    places them where they were and then labels their later repeats.
    """
    data = sample.get("data") or {}
    text = data.get("text") or ""
    if not text or "\x00" in text:
        return None
    source = (sample.get("annotations") or sample.get("predictions") or [{}])[0]
    mask = []
    for item in source.get("result", []):
        if item.get("type") != "labels":
            continue
        value = item.get("value") or {}
        labels = value.get("labels") or []
        if not labels or labels[0].upper() in _COREFERENCE_LABELS:
            continue
        start, end = value.get("start"), value.get("end")
        if not (
            isinstance(start, int) and isinstance(end, int) and 0 <= start < end
        ) or end > len(text):
            continue
        mask.append(
            {"value": text[start:end], "label": labels[0], "start": start, "end": end}
        )
    mask.sort(key=lambda m: (m["start"], m["end"]))
    return {
        "text": text,
        "privacy_mask": mask,
        "coreferences": [],
        "language": data.get("language"),
        "country": data.get("country"),
        "source": f"{KIJI_REPO} (cleaned export)",
    }


def load_kiji_dir(path: str) -> list[dict]:
    rows = []
    for file in sorted(Path(path).glob("*.json")):
        try:
            sample = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = kiji_row_from_labelstudio(sample)
        if row:
            rows.append(row)
    return rows


def copy_samples(mix_dir: str, out_dir: Path, skip_prefix: str) -> dict[str, int]:
    """Copy an earlier mix's samples, except those with ``skip_prefix``."""
    src = Path(mix_dir) / "samples"
    copied: Counter = Counter()
    for file in sorted(src.glob("*.json")):
        prefix = file.name.split("_", 1)[0]
        if prefix == skip_prefix:
            continue
        shutil.copyfile(file, out_dir / file.name)
        copied[prefix] += 1
    if not copied:
        raise SystemExit(f"No samples to reuse in {src}")
    return dict(copied)


def load_nym() -> list[dict]:
    ds = _load(NYM_REPO, data_files={"train": "train.jsonl"}, split="train")
    return [s for s in (convert_nym_row(r) for r in ds) if s]


def load_openpii(local_jsonl: str | None) -> list[dict]:
    if local_jsonl:
        with open(local_jsonl, encoding="utf-8") as f:
            rows = (json.loads(line) for line in f)
            return [s for s in (convert_openpii_row(r) for r in rows) if s]
    ds = _load(OPENPII_REPO, split="train", streaming=True)
    out, since_de = [], 0
    for row in ds:
        if row.get("language") != "de":
            since_de += 1
            if out and since_de > 50000:  # the file is grouped by language
                break
            continue
        since_de = 0
        sample = convert_openpii_row(row)
        if sample:
            out.append(sample)
    return out


def balanced_sample(rows: list[dict], num: int, key: str, seed: int) -> list[dict]:
    """Take ``num`` rows spread evenly over the values of ``rows[i][key]``."""
    if num <= 0 or num >= len(rows):
        return rows
    groups: dict[str | None, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get(key)].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    out, i = [], 0
    while len(out) < num:
        for group in groups.values():
            if i < len(group) and len(out) < num:
                out.append(group[i])
        i += 1
    return out


def write_samples(rows: list[dict], prefix: str, out_dir: Path) -> int:
    written = 0
    for row in rows:
        ls_sample = convert_to_labelstudio(row)
        ls_sample["data"]["source"] = row.get("source")
        if row.get("pii_free"):
            ls_sample["data"]["pii_free"] = True
        digest = hashlib.sha256(row["text"].encode()).hexdigest()[:20]
        name = f"{prefix}_{digest}.json"
        ls_sample["data"]["file_name"] = name
        path = out_dir / name
        if path.exists():
            continue  # exact duplicate text within a source
        with path.open("w", encoding="utf-8") as f:
            json.dump(ls_sample, f, ensure_ascii=False)
        written += 1
    return written


def summarize(out_dir: Path) -> dict:
    per_source: Counter = Counter()
    per_language: Counter = Counter()
    labels: dict[str, Counter] = defaultdict(Counter)
    pii_free: Counter = Counter()
    for path in out_dir.glob("*.json"):
        prefix = path.name.split("_", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        per_source[prefix] += 1
        per_language[data["data"].get("language") or "unknown"] += 1
        result = data["predictions"][0]["result"]
        has_pii = False
        for item in result:
            for label in item.get("value", {}).get("labels", []):
                labels[prefix][label] += 1
                has_pii = True
        if not has_pii:
            pii_free[prefix] += 1
    return {
        "rows_per_source": dict(per_source),
        "rows_per_language": dict(per_language.most_common()),
        "pii_free_rows_per_source": dict(pii_free),
        "entities_per_source_and_label": {
            k: dict(v.most_common()) for k, v in labels.items()
        },
    }


def build_other_sources(
    args: argparse.Namespace, out_dir: Path, written: dict[str, int]
) -> None:
    """Write the nym, openpii and template rows."""
    print(f"Loading German rows of {NYM_REPO} ...")
    nym = load_nym()
    if args.nym > 0:
        nym = random.Random(args.seed).sample(nym, min(args.nym, len(nym)))
    written["nym"] = write_samples(nym, "nym", out_dir)

    print(f"Loading German rows of {OPENPII_REPO} ...")
    openpii = balanced_sample(
        load_openpii(args.openpii_jsonl), args.openpii, "country", args.seed
    )
    written["openpii"] = write_samples(openpii, "openpii", out_dir)

    for language, prefix in SYNTHETIC_PREFIXES.items():
        num = args.synthetic if language == "German" else args.synthetic_per_language
        print(f"Generating {num} {language} negative and contrast rows ...")
        rows = generate_synthetic(num, seed=args.seed, language=language)
        written[prefix] = write_samples(rows, prefix, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", default="model/dataset/data_samples/german_mix_v3")
    ap.add_argument("--nym", type=int, default=0, help="German nym rows (0 = all)")
    ap.add_argument(
        "--openpii", type=int, default=8000, help="German openpii rows (0 = all)"
    )
    ap.add_argument("--synthetic", type=int, default=4000, help="German template rows")
    ap.add_argument(
        "--synthetic-per-language",
        type=int,
        default=1500,
        help="Template rows for each of English, French, Spanish, Dutch, Danish",
    )
    ap.add_argument(
        "--openpii-jsonl",
        default=None,
        help="Local German openpii rows (skips streaming)",
    )
    ap.add_argument(
        "--kiji-dir",
        default=None,
        help="Cleaned Label Studio export of the Kiji data (instead of the Hub)",
    )
    ap.add_argument(
        "--reuse-from",
        default=None,
        help="Earlier mix directory: copy its non-Kiji samples instead of "
        "rebuilding nym, openpii and template rows",
    )
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    root = Path(args.output_dir)
    out_dir = root / "samples"  # training_samples_dir; metadata stays one level up
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.json")):
        raise SystemExit(f"{out_dir} already has samples; use an empty directory.")

    written = {}
    if args.kiji_dir:
        print(f"Loading the cleaned {KIJI_REPO} export from {args.kiji_dir} ...")
        written["kiji"] = write_samples(load_kiji_dir(args.kiji_dir), "kiji", out_dir)
    else:
        print(f"Loading {KIJI_REPO} ...")
        written["kiji"] = write_samples(load_kiji(), "kiji", out_dir)

    if args.reuse_from:
        print(f"Copying nym, openpii and template rows from {args.reuse_from} ...")
        written.update(copy_samples(args.reuse_from, out_dir, skip_prefix="kiji"))
    else:
        build_other_sources(args, out_dir, written)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "built_from_commit": commit,
        "args": vars(args),
        "written": written,
        "sources": {k: {"dataset": v[0], "licence": v[1]} for k, v in LICENCES.items()},
        **summarize(out_dir),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    (root / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    print(json.dumps(written, indent=2))
    print(f"Done: {sum(written.values())} samples in {out_dir}")


if __name__ == "__main__":
    main()
