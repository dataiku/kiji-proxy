"""Smoke test for the backend's /predict endpoint, independent of Label Studio.

    .venv/bin/python test_predict.py
    .venv/bin/python test_predict.py --text "Some custom text"
"""

import argparse
import json
import sys
from pathlib import Path

import requests

LABEL_CONFIG_PATH = Path(__file__).parent.parent / "label_config.xml"

DEFAULT_TEXT = (
    "Hello, my name is Margaret L. Chen and I work at Riverbend Analytics. "
    "You can reach me at margaret.l.chen@email.com or (503) 555-0148."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9090/predict")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    label_config = LABEL_CONFIG_PATH.read_text() if LABEL_CONFIG_PATH.exists() else ""
    payload = {
        "tasks": [{"id": 1, "data": {"text": args.text}}],
        "label_config": label_config,
        "project": "test.0",
        "params": {"context": {}},
    }

    print(f"POST {args.url}\ntext: {args.text!r}\n")
    response = requests.post(args.url, json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body, indent=2))

    results = body.get("results", [])
    span_count = sum(len(p.get("result", [])) for p in results)
    print(f"\n{span_count} predicted span(s) across {len(results)} task(s).")
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
