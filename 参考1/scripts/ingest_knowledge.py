from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.ingest import ingest_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a Markdown knowledge zip into the EduAgentX RAG index.")
    parser.add_argument("--zip", dest="zip_path", default="data/raw/knowledge.zip", help="Path to a knowledge zip file.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy the source zip to data/raw/knowledge.zip before extraction.",
    )
    args = parser.parse_args()

    summary = ingest_knowledge(zip_path=args.zip_path, data_dir=args.data_dir, copy_zip=not args.no_copy)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
