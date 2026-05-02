"""
retriever.py — Semantic search over the FAISS corpus index.

Optional cross-encoder reranking for higher precision at the cost of latency.
"""

import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from indexer import MODEL_NAME, build_index

# Minimum cosine similarity to consider a result useful.
LOW_CONFIDENCE_THRESHOLD = 0.25

# Candidates fetched before domain-filtering and optional reranking.
FILTER_MULTIPLIER = 20

# Cross-encoder model for optional reranking (significant quality boost)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    def __init__(self, rebuild: bool = False, use_reranker: bool = False):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index, self.metadata = build_index(force=rebuild)
        self.reranker = None

        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANK_MODEL)
                print(f"  Cross-encoder reranker loaded: {RERANK_MODEL}")
            except Exception as exc:
                print(f"  Warning: failed to load reranker: {exc}")

    def retrieve(
        self, query: str, company: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        """
        Return up to top_k docs most relevant to query.

        Pipeline:
          1. FAISS broad retrieval (top_k * FILTER_MULTIPLIER candidates)
          2. Domain filter by company (falls back to all if too few results)
          3. Optional cross-encoder reranking
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        # Broad FAISS retrieval
        search_k = max(1, min(top_k * FILTER_MULTIPLIER, self.index.ntotal))
        scores, indices = self.index.search(query_vec, search_k)

        company_lower = company.lower().strip() if company else None

        def _collect(filter_company: Optional[str]) -> list[dict]:
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = self.metadata[idx]
                if filter_company and doc["company"] != filter_company:
                    continue
                results.append({**doc, "score": float(score)})
                if len(results) >= (top_k * 4 if self.reranker else top_k):
                    break
            return results

        candidates = _collect(company_lower)

        # Fall back to all companies if domain filter yields too few
        if len(candidates) < top_k and company_lower:
            candidates = _collect(None)

        if not candidates:
            return []

        # Cross-encoder reranking
        if self.reranker and len(candidates) > 1:
            pairs = [(query, doc["content"][:512]) for doc in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for doc, rs in zip(candidates, rerank_scores):
                doc["score"] = float(rs)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:top_k]

    def max_score(self, results: list[dict]) -> float:
        if not results:
            return 0.0
        return max(r["score"] for r in results)

    def is_low_confidence(self, results: list[dict]) -> bool:
        return self.max_score(results) < LOW_CONFIDENCE_THRESHOLD
