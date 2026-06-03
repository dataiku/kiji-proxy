"""WSGI entrypoint for the PII pre-annotation ML backend.

Run locally:
    set -a; source ../../../.env; set +a          # OPENAI_API_KEY (openai provider)
    PII_PROVIDER=openai .venv/bin/python _wsgi.py  # serves on :9090

Then connect Label Studio: Project > Settings > Model > Connect Model
(URL http://localhost:9090), or POST /api/ml/ (see README.md).
"""

import argparse
import logging
import os

from label_studio_ml.api import init_app

from model import PIIPreAnnotator

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="PII pre-annotation ML backend")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9090")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = init_app(model_class=PIIPreAnnotator)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
