# Qdrant, Milvus và Feast — So sánh và giải thích

Tài liệu này giải thích sự khác biệt giữa **Qdrant**, **Milvus** và **Feast**, cũng như lý do tại sao project `demo-observable-layer` lại có hai phiên bản: một dùng Qdrant và một dùng Feast + Milvus.

---

## 1. Qdrant là gì?

**Qdrant** là một **vector database** mã nguồn mở, được thiết kế riêng cho semantic search và RAG. Nó lưu trữ các vector embeddings (biểu diễn số của văn bản, ảnh, âm thanh...) và tìm kiếm các vector tương tự nhanh chóng.

### Đặc điểm chính

- Viết bằng **Rust**, nhanh và ổn định dưới tải cao.
- Dùng thuật toán **HNSW** (Hierarchical Navigable Small World) để tìm kiếm láng giềng gần nhất trong thời gian sublinear (không cần scan toàn bộ DB).
- Hỗ trợ **hybrid search** — kết hợp vector similarity + metadata filtering, với cơ chế gộp kết quả như Reciprocal Rank Fusion (RRF) và Distribution-Based Score Fusion (DBSF).
- Hỗ trợ **dense, sparse, multi-vector** search; mỗi vector có thể gắn **payload** (metadata) và filter theo payload.
- Có thể chạy local, Docker, hoặc dùng **Qdrant Cloud** (managed); ngoài ra còn có **Qdrant Edge** cho thiết bị biên (edge device).
- **License**: Open Source Edition theo **Apache 2.0** (repo GitHub); ngoài ra có Qdrant Enterprise Edition với license thương mại. Dự án đang phát triển rất tích cực (ra mắt liên tục, v1.19.0 là bản mới nhất).
- Client Python đơn giản: `QdrantClient`.

### Ví dụ trong project v1

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
hits = client.search(
    collection_name="passages",
    query_vector=query_emb,
    limit=5,
)
```

---

## 2. Milvus là gì?

**Milvus** cũng là một **vector database** mã nguồn mở, do Zilliz phát triển, tập trung vào khả năng **mở rộng (scalability)** và xử lý **tỉ tỉ vectors** trong môi trường distributed.

### Đặc điểm chính

- Kiến trúc **cloud-native, distributed**, có **3 chế độ triển khai**: Milvus Lite (local, file-based), Milvus Standalone (một máy), và Milvus Distributed (cluster Kubernetes, xử lý tỉ tỉ vectors).
- Hỗ trợ nhiều loại index: IVF_FLAT, HNSW, DISKANN...
- Tích hợp tốt với Spark, S3, và các data pipeline lớn.
- **License**: **Apache 2.0**, dự án do **LF AI & Data Foundation** quản lý, phát triển bởi cộng đồng từ Zilliz, NVIDIA, Intel, Meta, Microsoft. Phần lõi 100% miễn phí; bản managed là **Zilliz Cloud**.
- Client Python: `pymilvus`.

### Ví dụ trong project v2

```python
from feast import FeatureStore

store = FeatureStore("src")
docs = store.retrieve_online_documents_v2(
    features=["docs_embeddings:passage"],
    query=embedding,
    top_k=3,
    distance_metric="COSINE",
)
```

---

## 3. Qdrant vs Milvus

| Tiêu chí | **Qdrant** | **Milvus** |
|---|---|---|
| **Mục tiêu** | Dễ dùng, nhẹ, tốc độ cao cho RAG | Mở rộng lớn, distributed, enterprise |
| **Kiến trúc** | Đơn giản, dễ deploy | Cloud-native, modular, phức tạp hơn |
| **Quy mô** | Tốt cho vừa và nhỏ | Tốt cho lớn (tỉ tỉ vectors) |
| **Deployment** | Docker, local, Qdrant Cloud | Milvus Lite, Standalone, Distributed, Zilliz Cloud |
| **Index** | HNSW, binary quantization... | IVF_FLAT, HNSW, DISKANN... |
| **Hybrid search** | Hỗ trợ tốt | Hỗ trợ |
| **License** | Apache 2.0 (Open Source Edition) | Apache 2.0 |
| **Quản lý dự án** | Qdrant (công ty) | LF AI & Data Foundation |
| **Dễ cài đặt** | Rất dễ | Cần nhiều thành phần hơn |
| **Managed cloud** | Qdrant Cloud | Zilliz Cloud |

### Khi nào chọn cái nào?

- **Chọn Qdrant** khi bạn muốn nhanh, gọn, dễ triển khai RAG, ít component.
- **Chọn Milvus** khi bạn cần hệ thống lớn, distributed, hoặc tích hợp vào data platform phức tạp.

---

## 4. Feast là gì? Tại sao cần Feast?

**Feast** là một **feature store** mã nguồn mở. Nó **không phải** là vector database, mà là **lớp trung gian quản lý features** cho ML.

### Vấn đề Feast giải quyết

Trong ML, bạn cần dùng **cùng một bộ features** cho cả **training** và **serving**. Nếu không có feature store, dễ xảy ra **training-serving skew** — dữ liệu train và dữ liệu khi predict khác nhau.

Ngoài ra, Feast còn giúp **tránh data leakage** bằng cách tạo các feature set đúng theo thời điểm (point-in-time correct) — đảm bảo giá trị feature tương lai không bị lọt vào quá trình training.

Feast là dự án mã nguồn mở (Apache 2.0) **vẫn đang phát triển tích cực**, do LF AI & Data Foundation quản lý, được dùng bởi nhiều công ty lớn (Shopify, Adyen, Cloudflare, Red Hat, Affirm...).

#### Training-serving skew là gì?

Đây là tình huống code xử lý feature trong quá trình **train model** khác với code xử lý feature trong quá trình **serving (predict)**. Kết quả là model thấy dữ liệu "không giống nhau" giữa hai giai đoạn, dẫn đến dự đoán kém chính xác khi đưa vào production.

#### Ví dụ cụ thể

Giả sử bạn làm hệ thống gợi ý sản phẩm, có một feature là `"tuổi tài khoản"`:

- **Khi train**: bạn tính `tuổi_tài_khoản = năm_nay - năm_tạo_tài_khoản`.
- **Khi serving**: developer khác lại tính `tuổi_tài_khoản = tháng_nay - tháng_tạo_tài_khoản` (đơn vị tháng thay vì năm).

Model đã học theo đơn vị năm, nhưng khi chạy thực tế lại nhận giá trị theo tháng → dự đoán sai.

#### Ví dụ khác với embeddings

Trong RAG, bạn dùng embedding để retrieve tài liệu:

- **Khi train/index**: team A dùng model `all-MiniLM-L6-v2` để tạo embeddings.
- **Khi serving**: team B lại dùng model `all-mpnet-base-v2` khác để encode câu truy vấn.

Hai model cho embedding khác không gian vector, khi tìm kiếm sẽ không khớp → retrieve sai tài liệu.

#### Feast giải quyết như thế nào?

Feast định nghĩa feature một lần duy nhất trong **feature definition**, rồi dùng chung cho cả training và serving. Nhờ đó:

- Cùng một định nghĩa feature được dùng ở cả hai nơi.
- Cùng một phiên bản dữ liệu được serve.
- Giảm thiểu lỗi do code xử lý feature bị lặp lại hoặc không đồng bộ.

### Thành phần của Feast

| Thành phần | Mục đích | Ví dụ |
|---|---|---|
| **Offline store** | Lưu trữ lịch sử để train model | File, BigQuery, Snowflake, Redshift, Spark |
| **Online store** | Phục vụ low-latency inference | Redis, PostgreSQL, SQLite, **Milvus** |
| **Registry** | Quản lý định nghĩa features | SQLite, PostgreSQL |

### Tại sao lại dùng Feast + Milvus trong project v2?

- Feast đóng vai trò **feature store**, Milvus đóng vai trò **online store** cho vector embeddings.
- Quan trọng: theo docs chính thức, Feast hỗ trợ nhiều vector store làm online store, gồm **Qdrant**, **Milvus** và **Faiss**. Như vậy v1 (dùng Qdrant trực tiếp) và v2 (dùng Feast + Milvus) đều hợp lệ — v2 chỉ chọn Milvus làm backend cho Feast.
- Khi cần retrieve documents, code gọi `store.retrieve_online_documents_v2(...)` thay vì gọi trực tiếp Qdrant/Milvus. Điều này giúp:
  - Quản lý features tập trung.
  - Dễ dàng thay đổi backend vector DB (đổi từ Milvus sang Qdrant/Faiss) mà không cần sửa nhiều code, vì app chỉ giao tiếp qua Feast.
  - Đảm bảo consistency giữa training và serving.

### Ví dụ `feature_store.yaml`

```yaml
project: mlops5
provider: gcp
registry:
  registry_type: sql
  path: postgresql+psycopg2://airflow:airflow@127.0.0.1:5432/feast
  cache_ttl_seconds: 60
online_store:
  type: milvus
  host: http://milvus-standalone
  port: 19530
  vector_enabled: true
  text_search_enabled: true
  embedding_dim: 384
  index_type: "IVF_FLAT"
  metric_type: "L2"
offline_store:
  type: file
entity_key_serialization_version: 3
auth:
  type: no_auth
```

---

## 5. Tóm tắt

| Thành phần | Vai trò | Trong project |
|---|---|---|
| **Qdrant** | Vector database đơn giản, dễ dùng | Dùng ở v1, gọi trực tiếp `QdrantClient` |
| **Milvus** | Vector database scalable, enterprise | Dùng ở v2 làm online store cho Feast |
| **Feast** | Feature store — quản lý và serve features | Dùng ở v2, abstract việc truy vấn vector |

### Tại sao v2 chuyển sang Feast + Milvus?

Vì muốn áp dụng **feature store pattern** — tách biệt quản lý features khỏi ứng dụng, dễ mở rộng và bảo trì hơn trong production. Qdrant vẫn tốt cho RAG đơn giản, nhưng nếu hệ thống lớn hoặc cần quản lý features nghiêm ngặt thì **Feast + Milvus** là lựa chọn phổ biến hơn.

---

## 6. Tham khảo

- Qdrant docs: https://qdrant.tech/documentation/
- Milvus docs: https://milvus.io/docs
- Feast docs: https://docs.feast.dev/
  - Feast + Milvus online store: https://docs.feast.dev/reference/online-stores/milvus
  - Feast + Qdrant online store: https://docs.feast.dev/reference/online-stores/qdrant
  - Feast + Faiss online store: https://docs.feast.dev/reference/online-stores/faiss
