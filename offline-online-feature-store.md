# Offline Store và Online Store — Feature Store tác động thế nào đến ML?

> Tổng hợp từ tài liệu chính thức của Feast: https://docs.feast.dev/

---

## 1. Vì sao phải tách Offline / Online store?

Trong ML, có hai nhu cầu dữ liệu **trái ngược nhau**:

| | **Training** | **Serving (predict)** |
|---|---|---|
| Cần gì | Toàn bộ **lịch sử** dữ liệu | Giá trị features **mới nhất** |
| Khối lượng | Rất lớn (nhiều năm, nhiều dòng) | Nhỏ (chỉ 1 entity tại 1 thời điểm) |
| Độ trễ | Không quan trọng (chạy batch) | **Cực kỳ quan trọng** (ms) |
| Công cụ phù hợp | Data warehouse / file lớn | Key-value store / vector DB in-memory |

Nếu dùng **một** cơ sở dữ liệu cho cả hai việc thì sẽ bị một trong hai: hoặc query lịch sử quá chậm, hoặc phục vụ real-time không nhanh. Vì vậy feature store tách thành **offline store** và **online store**.

---

## 2. Offline Store là gì?

**Offline store** là nơi lưu **dữ liệu lịch sử** của features — dùng để **xây dataset huấn luyện** và làm nguồn để **materialize** sang online store.

### Đặc điểm

- Tối ưu cho **batch processing** — đọc/aggregate lượng lớn dữ liệu.
- Lưu features theo **từng thời điểm (event timestamp)**.
- Không yêu cầu độ trễ thấp.

### Ví dụ công nghệ

- File (Parquet), DuckDB, SQLite
- BigQuery, Snowflake, Redshift
- Spark
- PostgreSQL (contrib plugin), Athena, Clickhouse, ...

### Cách dùng

```python
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_hourly_stats:conv_rate"],
).to_df()  # dataset để train model
```

---

## 3. Online Store là gì?

**Online store** là nơi lưu **giá trị features mới nhất** — dùng để phục vụ **real-time inference** với độ trễ rất thấp.

### Đặc điểm

- Tối ưu cho **low-latency lookup** — giống query key-value.
- Lưu giá trị gần nhất, không cần lịch sử.
- Hỗ trợ **vector search** (cho RAG / semantic retrieval).
- Dữ liệu được đưa vào qua **materialization**.

### Ví dụ công nghệ

- **Key-value**: Redis, DynamoDB, SQLite, PostgreSQL, Cassandra, MySQL...
- **Vector DB**: Milvus, Qdrant, Faiss, Elasticsearch ✅

### Cách dùng

```python
feature_vector = store.get_online_features(
    features=["driver_hourly_stats:conv_rate"],
    entity_rows=[{"driver_id": 1001}],
).to_dict()

model.predict(feature_vector)
```

Hoặc vector search trong RAG:

```python
docs = store.retrieve_online_documents_v2(
    features=["docs_embeddings:passage"],
    query=embedding,
    top_k=3,
    distance_metric="COSINE",
).to_df()
```

---

## 4. Bảng so sánh Offline vs Online Store

| Tiêu chí | **Offline Store** | **Online Store** |
|---|---|---|
| Mục đích | Build training data | Real-time serving |
| Loại dữ liệu | Lịch sử (point-in-time) | Giá trị mới nhất |
| Khối lượng | Rất lớn | Nhỏ (1 entity/lần) |
| Độ trễ | Thấp ưu tiên | Cực thấp (ms) |
| Truy vấn | `get_historical_features` | `get_online_features` |
| Ví dụ | File, BigQuery, Snowflake, Spark | Redis, SQLite, **Milvus, Qdrant, Faiss** |
| Data flow | Nguồn dữ liệu | Được **materialize** từ offline store |

---

## 5. Materialization — cầu nối giữa hai store

**Materialization** là quá trình query offline store (batch source) trong một khoảng thời gian, rồi **copy giá trị feature mới nhất** vào online store.

```
Offline Store ──feast materialize──▶ Online Store
(file, BigQuery)                       (Redis, Milvus, Qdrant)
```

```bash
# Toàn bộ feature views
feast materialize 2021-04-07T00:00:00 2021-04-08T00:00:00

# Tăng dần — chạy định kỳ bằng scheduler
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

> ⚠️ Muốn online serving chạy được thì **phải materialize trước**. Nếu quên, `get_online_features` sẽ trả về rỗng.

---

## 6. Feast tác động như thế nào đến quy trình ML?

### ✅ 1. Chống Training-Serving Skew

- Training và serving dùng **cùng một định nghĩa feature** (từ cùng FeatureView/FeatureService).
- Không còn tình trạng "code xử lý feature khi train khác khi serve".

### ✅ 2. Chống Data Leakage (point-in-time join)

- `get_historical_features` nối features **theo event timestamp**: chỉ lấy feature có thời điểm ≤ thời điểm của dòng dữ liệu.
- Dữ liệu "tương lai" không thể lọt vào training → model học đúng, không bị ảo tưởng độ chính xác.

### ✅ 3. Consistent giữa các model / team

- Features được định nghĩa tập trung, nhiều team, nhiều model dùng chung.
- Không mỗi nơi tự tính một kiểu.

### ✅ 4. Giảm độ trễ serving

- Online store (Redis/Milvus/Qdrant) trả feature trong vài ms thay vì query batch warehouse.

### ✅ 5. Tách rời ML khỏi data infrastructure (decoupling)

- App chỉ giao tiếp với Feast, không gắn chặt vào database cụ thể.
- Muốn đổi backend (vd từ Milvus sang Qdrant) chỉ cần đổi config, không sửa code.
- Model chạy được trên batch → realtime → nhiều infra khác nhau.

### ✅ 6. Tiêu chuẩn hóa qui trình

- `apply → materialize → train / serve` trở thành pipeline chuẩn, dễ tự động hóa bằng Airflow/K8s.

---

## 7. Liên hệ với project demo-observable-layer (v2)

```
┌─────────────────────────────────────────────────────┐
│                    Feast                            │
│  feature_store.yaml                                  │
│  ├── offline_store: file        (training data)     │
│  └── online_store: milvus       (vector retrieval)  │
└─────────────────────────────────────────────────────┘
                          │
            retrieve_online_documents_v2(...)
                          ▼
                   RAG Chatbot
```

- **Offline store** = file: nơi chứa embeddings gốc (train data / index).
- **Online store** = Milvus: serve vector search theo câu hỏi, trả top-k tài liệu liên quan.
- App chỉ gọi `store.retrieve_online_documents_v2(...)` — không cần biết Milvus host/port — đúng tinh thần decoupling của Feast.

---

## 8. Tóm tắt

- **Offline store** phục vụ **training** (lịch sử, lớn, batch, point-in-time).
- **Online store** phục vụ **serving** (mới nhất, nhỏ, low-latency, vector search).
- **Materialization** nối hai nơi lại với nhau.
- Feast đảm bảo **consistency**, **chống leakage**, **giảm latency**, và **decoupling** — đó là toàn bộ "tác động" của feature store tới ML workflow.

---

## 9. Tham khảo

- Offline store: https://docs.feast.dev/reference/offline-stores/
- Online store: https://docs.feast.dev/reference/online-stores/
- Vector store (Milvus): https://docs.feast.dev/reference/online-stores/milvus
- Vector store (Qdrant): https://docs.feast.dev/reference/online-stores/qdrant