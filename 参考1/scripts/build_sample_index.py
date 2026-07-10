from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.ingest import build_sample_course_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EduAgentX demo sample-course RAG index.")
    parser.add_argument("--sample-dir", default="data/sample_course", help="Directory containing sample Markdown files.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Demo index output directory. Defaults to data/processed/sample_index.",
    )
    args = parser.parse_args()
    summary = build_sample_course_index(sample_dir=args.sample_dir, data_dir=args.data_dir, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
