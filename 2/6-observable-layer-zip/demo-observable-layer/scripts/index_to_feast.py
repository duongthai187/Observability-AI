"""
Index passages from parquet into Feast (Milvus online store).

Usage:
    uv run python scripts/index_to_feast.py

Prerequisites:
    - Docker infra running (postgres, milvus)
    - feast database created in PostgreSQL
    - feast apply has been run (feature_view.py registered)
"""

import hashlib
import pandas as pd
from datetime import datetime, timezone
from feast import FeatureStore

PARQUET_PATH = "scripts/rag-mini-wikipedia-embeddings.parquet"
FEATURE_VIEW_NAME = "docs_embeddings"
BATCH_SIZE = 128


def _stable_id(text: str) -> str:
    """Generate a stable string ID from passage text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def index_parquet(path: str):
    # 1. Read parquet
    df = pd.read_parquet(path)
    print(f"Read {len(df)} passages from {path}")

    if "passage" not in df.columns:
        raise SystemExit("Parquet must contain 'passage' column")

    # 2. Prepare columns to match FeatureView schema
    #    FeatureView expects: passage_id, passage, passage_embedding, event_timestamp
    rows = []
    for _, row in df.iterrows():
        text = str(row["passage"])
        emb = row.get("embedding")
        ts = row.get("event_timestamp", datetime.now(timezone.utc))

        # Compute embedding if not present in parquet
        if emb is None:
            # Fallback: encode using all-MiniLM-L6-v2 (same as retrieval)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(text).tolist()
        elif isinstance(emb, (list, tuple)):
            emb = list(emb)
        else:
            # Already a list
            emb = emb

        rows.append({
            "passage_id": _stable_id(text),
            "passage": text,
            "passage_embedding": emb,
            "event_timestamp": ts,
            "created": pd.Timestamp.now(),
        })

    out_df = pd.DataFrame(rows)
    print(f"Prepared {len(out_df)} rows with columns: {list(out_df.columns)}")
    print(f"Embedding dimension: {len(out_df['passage_embedding'].iloc[0])}")

    # 3. Write to Feast online store (Milvus)
    store = FeatureStore(repo_path="feature_repo")
    store.write_to_online_store(
        feature_view_name=FEATURE_VIEW_NAME,
        df=out_df,
    )
    print(f"✅ Indexed {len(out_df)} passages into Feast → Milvus ({FEATURE_VIEW_NAME})")


if __name__ == "__main__":
    index_parquet(PARQUET_PATH)