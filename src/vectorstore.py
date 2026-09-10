"""
vectorstore.py
Handles: embedding chunks, storing in FAISS, semantic + hybrid (BM25) retrieval.
"""

import os
# Prevent transformers from trying to import TensorFlow/Flax — we only need PyTorch,
# and a stray TF install can cause protobuf version conflicts.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict

MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks: List[Dict] = []
        self.bm25 = None

    def build(self, chunks: List[Dict]):
        """Embed all chunks and build both the FAISS index and BM25 index."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine since normalized
        self.index.add(embeddings)

        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def add(self, chunks: List[Dict]):
        """Add more chunks to an existing store (rebuilds indices for simplicity)."""
        self.build(self.chunks + chunks)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Combine semantic (cosine) and BM25 (keyword) scores.
        alpha = weight given to semantic score (1-alpha goes to BM25).
        """
        # semantic scores over ALL chunks
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        all_scores, all_indices = self.index.search(q_emb, len(self.chunks))
        semantic_scores = {idx: score for score, idx in zip(all_scores[0], all_indices[0])}

        # bm25 scores over all chunks
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1

        combined = []
        for i, chunk in enumerate(self.chunks):
            sem = semantic_scores.get(i, 0)
            bm25_norm = bm25_scores[i] / bm25_max
            final_score = alpha * sem + (1 - alpha) * bm25_norm
            combined.append((final_score, chunk))

        combined.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in combined[:top_k]:
            c = dict(chunk)
            c["score"] = float(score)
            results.append(c)
        return results
