from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import pandas as pd
import json
import hashlib
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models as rest

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None
PARQUET_PATH = "./rag-mini-wikipedia-embeddings.parquet"
COLLECTION_NAME = "passages"
BATCH_SIZE = 128

if not PARQUET_PATH:
    raise SystemExit("Set PARQUET_PATH env or pass path as first arg")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

def _ensure_collection(vector_size: int):
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=types.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            )
        )

def _stable_id(text: str) -> int:
    # stable 64-bit-ish id
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)

def index_parquet(path: str):
    df = pd.read_parquet(path)
    if "passage" not in df.columns:
        raise SystemExit("Parquet must contain 'passage' column")

    # prepare embeddings (use existing column if present)
    if "embedding" in df.columns:
        first_embedding = next((e for e in df["embedding"] if e is not None), None)
        if first_embedding is None:
            # compute embeddings if column empty
            df["embedding"] = df["passage"].apply(lambda x: model.encode(x).tolist())
            vector_size = len(df["embedding"].iloc[0])
        else:
            vector_size = len(first_embedding)
    else:
        df["embedding"] = df["passage"].apply(lambda x: model.encode(x).tolist())
        vector_size = len(df["embedding"].iloc[0])

    _ensure_collection(vector_size)

    points_batch = []
    for _, row in df.iterrows():
        passage = str(row["passage"])
        emb = row["embedding"]
        ts = row.get("event_timestamp", None)
        pid = _stable_id(passage)
        payload = {"text": passage}
        if ts is not None:
            payload["event_timestamp"] = str(ts)
        points_batch.append({"id": pid, "vector": emb, "payload": payload})

        if len(points_batch) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            points_batch = []

    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)

    print(f"Indexed {len(df)} passages into collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    index_parquet(PARQUET_PATH)