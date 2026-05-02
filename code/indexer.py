"""
indexer.py — Build and cache FAISS index from the data/ corpus.

Run directly to force a rebuild:
    python code/indexer.py --rebuild
"""

import json
import sys
import argparse
from pathlib import Path

# yaml is a lightweight dep — import at module level
import yaml

# Heavy ML deps (faiss, sentence_transformers, numpy) are imported lazily
# inside build_index() so that pure utility functions (parse_frontmatter,
# get_company, get_product_area) can be imported in tests without triggering
# model downloads.

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = Path(__file__).parent / ".cache"
INDEX_FILE = CACHE_DIR / "index.faiss"
METADATA_FILE = CACHE_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CONTENT_CHARS = 2000   # used for embedding text (keep stable)
MAX_DISPLAY_CHARS = 5000   # stored in metadata for LLM context display
EMBED_BATCH_SIZE = 64


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                return meta, parts[2].strip()
            except yaml.YAMLError:
                pass
    return {}, text.strip()


def get_company(path: Path) -> str:
    """Derive company tag from file path (data/<company>/...)."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "data" and i + 1 < len(parts):
            return parts[i + 1].lower()
    return "unknown"


def get_product_area(path: Path, breadcrumbs: list) -> str:
    """Derive product area from breadcrumbs or subdirectory."""
    if breadcrumbs and len(breadcrumbs) >= 2:
        return breadcrumbs[-1].lower().replace(" ", "_")
    parts = path.parts
    for i, part in enumerate(parts):
        if part in ("hackerrank", "claude", "visa") and i + 1 < len(parts):
            return parts[i + 1].lower().replace("-", "_")
    return ""


def load_corpus() -> tuple[list[dict], list[str]]:
    """Walk data/ and parse all markdown files. Returns (docs, embed_texts)."""
    docs = []
    texts = []

    md_files = sorted(DATA_DIR.rglob("*.md"))
    print(f"  Found {len(md_files)} documents in corpus")

    for path in md_files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter(raw)

            company = get_company(path)
            breadcrumbs = meta.get("breadcrumbs") or []
            product_area = get_product_area(path, breadcrumbs)
            title = meta.get("title") or path.stem.replace("-", " ").replace("_", " ")
            source_url = meta.get("source_url") or ""

            content_for_embed = body[:MAX_CONTENT_CHARS]
            content_for_display = body[:MAX_DISPLAY_CHARS]
            embed_text = f"{title}\n{content_for_embed}"

            docs.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "title": title,
                    "company": company,
                    "product_area": product_area,
                    "source_url": source_url,
                    "breadcrumbs": breadcrumbs,
                    "content": content_for_display,
                }
            )
            texts.append(embed_text)

        except Exception as exc:
            print(f"  Warning: skipping {path.name}: {exc}")

    return docs, texts


def build_index(force: bool = False):
    """
    Build FAISS IndexFlatIP from corpus and cache to disk.
    On subsequent calls, load from cache unless force=True.
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and INDEX_FILE.exists() and METADATA_FILE.exists():
        print("Loading cached FAISS index...")
        index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Loaded {index.ntotal} vectors")
        return index, metadata

    print("Building FAISS index from corpus...")
    docs, texts = load_corpus()

    if not texts:
        raise RuntimeError(f"No documents found under {DATA_DIR}")

    print(f"  Embedding {len(texts)} documents with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize → inner product = cosine similarity
        convert_to_numpy=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return index, docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from corpus")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cache exists")
    args = parser.parse_args()
    build_index(force=args.rebuild)
    print("Done.")
