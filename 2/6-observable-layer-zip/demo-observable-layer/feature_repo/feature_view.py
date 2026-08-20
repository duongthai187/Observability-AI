from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Array, Float32, String, UnixTimestamp

# ──────────────────────────────────────────────
# Entity — "primary key" for each passage
# ──────────────────────────────────────────────
passage = Entity(
    name="passage_id",
    description="Unique identifier for each passage",
    join_keys=["passage_id"],
)

# ──────────────────────────────────────────────
# Offline data source (parquet file)
# ──────────────────────────────────────────────
passages_source = FileSource(
    name="passages_source",
    path="scripts/rag-mini-wikipedia-embeddings.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ──────────────────────────────────────────────
# Feature View — schema for vector search
# ──────────────────────────────────────────────
# The field with vector_index=True is the embedding used for similarity search.
# Feast will create a Milvus collection with IVF_FLAT index on this field.
docs_embeddings = FeatureView(
    name="docs_embeddings",          # ⚠️ MUST match the name used in retrieval.py
    entities=[passage],
    ttl=timedelta(days=365),
    schema=[
        Field(name="passage", dtype=String),
        Field(name="passage_id", dtype=String),
        Field(name="event_timestamp", dtype=UnixTimestamp),
        Field(
            name="passage_embedding",        # embedding vector field
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="COSINE",
        ),
    ],
    source=passages_source,
    online=True,
)