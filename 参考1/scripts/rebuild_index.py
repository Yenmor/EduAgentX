from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.chunker import load_chunks_jsonl
from backend.rag.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the vector index from processed chunks.jsonl.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl", help="Path to chunks.jsonl.")
    parser.add_argument("--index-dir", default="data/processed/faiss_index", help="Output FAISS/fallback index directory.")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    chunks = load_chunks_jsonl(chunks_path)
    store = VectorStore()
    store.build_index(chunks)
    store.save_index(args.index_dir)
    print(json.dumps({"chunk_count": len(chunks), "index_dir": str(Path(args.index_dir).resolve()), "backend": store.backend}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
