from src.utils import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json
import os
import re

class RetrievalService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        self.collection = os.getenv("COLLECTION_NAME", "passages")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    def _tokenize(self, text: str):
        tokens = re.findall(r"\w+", text.lower())
        return set(tokens)

    def retrieve_vector(self, question: str, top_k: int = 5, alpha: float = 0.75):
        """Hybrid search: vector search in Qdrant + lightweight text re-ranking.
        alpha: weight for vector score (0..1). (1-alpha) is text boost weight.
        """
        try:
            query_emb = self.model.encode(question).tolist()

            # 1) vector search in qdrant
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_emb,
                limit=top_k,
                with_payload=True,
            )

            results = []
            for h in hits:
                payload = h.payload or {}
                text = payload.get("text", "")
                vec_score = getattr(h, "score", None)
                results.append({"text": text, "vec_score": vec_score, "raw": h})

            if not results:
                return json.dumps([], ensure_ascii=False)

            # 2) normalize vector scores (min-max)
            vec_scores = [r["vec_score"] if r["vec_score"] is not None else 0.0 for r in results]
            v_min, v_max = min(vec_scores), max(vec_scores)
            if v_max - v_min > 1e-12:
                for r in results:
                    r["vec_norm"] = (r["vec_score"] - v_min) / (v_max - v_min)
            else:
                for r in results:
                    r["vec_norm"] = 1.0

            # 3) compute text overlap score
            q_tokens = self._tokenize(question)
            for r in results:
                p_tokens = self._tokenize(r["text"])
                if not q_tokens:
                    r["text_score"] = 0.0
                else:
                    overlap = q_tokens.intersection(p_tokens)
                    r["text_score"] = len(overlap) / len(q_tokens)

            # 4) combine scores
            for r in results:
                r["combined_score"] = alpha * r["vec_norm"] + (1 - alpha) * r["text_score"]

            # 5) sort and return
            results_sorted = sorted(results, key=lambda x: x["combined_score"], reverse=True)
            out = [{"text": r["text"], "score": float(r["combined_score"])} for r in results_sorted[:top_k]]
            return json.dumps(out, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error during vector retrieval: {e}")
            return json.dumps([], ensure_ascii=False)