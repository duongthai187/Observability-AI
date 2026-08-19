# Feast — Feature Store chi tiết

> Tóm tắt từ tài liệu chính thức: https://docs.feast.dev/

---

## 1. Feast là gì?

**Feast (Feature Store)** là một feature store mã nguồn mở (Apache 2.0) do **LF AI & Data Foundation** quản lý. Feast cung cấp cho team ML:

- Nơi định nghĩa features **một lần duy nhất** (single source of truth).
- Quản lý **offline store** (dữ liệu lịch sử để train) và **online store** (dữ liệu low-latency để serve).
- Giúp đưa features từ data infrastructure tới model **training** và **online inference** một cách nhất quán.

> **Cốt lõi**: làm cho features "luôn nhất quán và đúng lúc" giữa giai đoạn huấn luyện và giai đoạn chạy thật, đồng thời **tránh data leakage**.

---

## 2. Kiến trúc Feast

```
┌─────────────────────────────────────────────────────────────┐
│                    Feature Repository                       │
│     (định nghĩa: Entity, FeatureView, FeatureService)        │
└──────────────────────────┬──────────────────────────────────┘
                           │ feast apply
                           ▼
                    ┌──────────────┐
                    │   Registry   │  ← lưu định nghĩa features
                    └──────┬───────┘
                           │
              ┌────────────┴────────────────────────────┐
              ▼                                         ▼
   ┌────────────────────┐                   ┌────────────────────┐
   │   Offline Store     │    materialize    │    Online Store    │
   │  (file, BigQuery,   │ ───────────────▶  │  (Redis, SQLite,   │
   │   Snowflake, Spark) │                   │   Milvus, Qdrant)  │
   └────────────────────┘                   └────────────────────┘
         get_historical_features                  get_online_features
         (train model)                            (predict low-latency)
```

### Thành phần chính

| Thành phần | Vai trò |
|---|---|
| **Feature Repository** | Thư mục chứa định nghĩa features (Entity, FeatureView, FeatureService) bằng Python |
| **Registry** | Nơi lưu metadata/định nghĩa features (SQLite, SQL, file...) |
| **Offline Store** | Lưu dữ liệu lịch sử → phục vụ training + materialize |
| **Online Store** | Lưu giá trị feature mới nhất → phục vụ inference low-latency |
| **Feature Server** | Serve features qua HTTP, tách deployment |
| **Materialization** | Quá trình copy dữ liệu từ offline store → online store |

---

## 3. Core concepts

### 3.1 Entity

**Entity** là "khóa" dùng để nối features — ví dụ `driver_id`, `user_id`. Mỗi feature gắn với một entity.

```python
from feast import Entity

driver = Entity(
    name="driver",
    join_keys=["driver_id"],
)
```

### 3.2 FeatureView

**FeatureView** là một nhóm features logic, kết nối với một data source, có TTL, và gắn với entity.

```python
from feast import FeatureView, Field
from feast.types import Float32, Int32
from datetime import timedelta

driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=["driver"],
    ttl=timedelta(days=1),       # giữ data trong bao lâu
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int32),
    ],
    source=driver_stats_source,
)
```

**Khái niệm quan trọng** — FeatureView có flag:
- `online=True`: feature được đưa vào online store để serve.
- `offline=True`: feature chỉ dùng cho training.

### 3.3 FeatureService

**FeatureService** gom một nhóm features lại, dùng chung cho cả training và serving — đảm bảo bạn luôn query đúng cùng một bộ features.

```python
from feast import FeatureService

driver_ranking_service = FeatureService(
    name="driver_ranking",
    features=[driver_hourly_stats],
)
```

### 3.4 Data Source

Nguồn dữ liệu cho FeatureView: file parquet, Kafka, PostgreSQL, BigQuery, Snowflake, Redshift, Spark...

---

## 4. Vòng đời làm việc với Feast

### Bước 1 — Khởi tạo repository

```bash
pip install feast
feast init my_feature_repo
```

### Bước 2 — Định nghĩa features (Python) rồi apply vào registry

```bash
feast apply
```

CLI sẽ **đồng bộ registry** với định nghĩa trong feature repository.

### Bước 3 — Materialize (copy offline → online)

```bash
# Toàn bộ feature views trong khoảng thời gian
feast materialize 2021-04-07T00:00:00 2021-04-08T00:00:00

# Tăng dần (recommended) — chạy định kỳ bằng scheduler (Airflow...)
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME

# Chỉ một feature view cụ thể
feast materialize ... --views driver_hourly_stats
```

> **Lưu ý**: muốn `get_online_features` chạy được thì features **phải được materialize trước**.

### Bước 4 — Train model với historical features

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path=".")

entity_df = pd.DataFrame({
    "driver_id": [1001, 1002, 1003],
    "event_timestamp": [
        datetime(2023, 6, 1),
        datetime(2023, 6, 15),
        datetime(2023, 7, 1),
    ],
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
    ],
).to_df()
```

### Bước 5 — Serve online features với độ trễ thấp

```python
feature_vector = store.get_online_features(
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
    ],
    entity_rows=[{"driver_id": 1001}],
).to_dict()

model.predict(feature_vector)
```

---

## 5. Hai loại retrieval quan trọng

### `get_historical_features` (cho training)

- Lấy **lịch sử** features tại đúng từng thời điểm (event timestamp) của mỗi entity.
- Thực hiện **point-in-time join**: với mỗi dòng entity, **chỉ** lấy giá trị feature có `event_timestamp <= timestamp` của dòng đó.
- Mục đích: **chống data leakage** — dữ liệu "tương lai" không được lọt vào quá trình training.
- Có thể trả về DataFrame (`to_df()`), PyArrow (`to_arrow()`), hoặc SQL string (`to_sql_string()`).

### `get_online_features` (cho serving)

- Lấy **giá trị mới nhất** của features, phục vụ inference với độ trễ rất thấp.
- Chạy nhanh như một lookup key-value trên Redis/SQLite/Milvus/Qdrant...
- Trả về dạng dict (`to_dict()`) hoặc DataFrame (`to_df()`).

---

## 6. Point-in-time correct joins — Chống data leakage chi tiết

> Đây là **tính năng quan trọng nhất** của Feast giúp tránh data leakage.
> Nguồn: [Feast Docs — Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)

### 6.1 Vấn đề: Data leakage là gì?

**Data leakage** xảy ra khi dữ liệu từ **tương lai** (future data) bị lọt vào quá trình training. Khi đó model học được những thông tin mà nó **không thể có** tại thời điểm dự đoán thực tế, dẫn đến:

- ✅ Accuracy trên validation/test rất cao
- ❌ Khi deploy → performance thực tế rất tệ (vì model đã "gian lận")

**Ví dụ cụ thể:**

| event_timestamp | driver_id | conv_rate (thực tế) | conv_rate (bị leak) |
|---|---|---|---|
| 2024-06-01 08:00 | 1001 | 0.45 | 0.45 ✅ |
| 2024-06-01 10:00 | 1001 | 0.52 | 0.52 ✅ |
| 2024-06-01 12:00 | 1001 | 0.38 | **0.61** ❌ ← giá trị từ 2024-06-02 |

Giá trị `0.61` là conv_rate của ngày hôm sau (2024-06-02) nhưng bị join nhầm vào dòng training của ngày 2024-06-01 → data leakage.

### 6.2 Cách Feast giải quyết: Point-in-time join

Feast dùng **event timestamp** làm mốc chặn trên (upper bound) khi join features:

```
 entity_df                              feature data
┌──────────────────────┐             ┌──────────────────────────────┐
│ driver_id │ timestamp │             │ driver_id │ timestamp │ value│
├──────────────────────┤             ├──────────────────────────────┤
│   1001    │ 2024-06-01│──┐         │   1001    │ 2024-06-01 │ 0.45 │◄── lấy
│   1001    │ 2024-06-02│──┤         │   1001    │ 2024-06-02 │ 0.61 │◄── lấy
│   1001    │ 2024-06-03│──┤         │   1001    │ 2024-06-03 │ 0.55 │
└──────────────────────┘  │         │   1001    │ 2024-06-04 │ 0.70 │── không lấy
                          │         └──────────────────────────────┘
                          │                    ▲
                          └────────────────────┘
                    Feast chỉ lấy feature có
                    event_timestamp <= entity timestamp
```

**Cơ chế hoạt động:**

1. Với mỗi dòng trong `entity_df` (dữ liệu training), Feast nhìn vào cột `event_timestamp`.
2. Feast truy vấn `offline store` và chỉ lấy bản ghi feature có `event_timestamp <= entity_timestamp`.
3. Feast lấy bản ghi **gần nhất** (latest value) trước hoặc bằng mốc thời gian đó.
4. Kết quả: **không có giá trị "tương lai" nào lọt vào training set.**

### 6.3 Code minh họa

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path=".")

# entity_df chứa cột event_timestamp — mốc thời gian của từng sự kiện
entity_df = pd.DataFrame({
    "driver_id": [1001, 1001, 1001],
    "event_timestamp": [
        datetime(2024, 6, 1, 8, 0),   # sáng 1/6
        datetime(2024, 6, 1, 10, 0),  # trưa 1/6
        datetime(2024, 6, 1, 12, 0),  # chiều 1/6
    ],
})

# Feast tự động point-in-time join — không leak!
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_hourly_stats:conv_rate"],
).to_df()
```

### 6.4 Lọc thêm bằng `created_timestamp` — Chống backfill leakage

Ngoài `event_timestamp`, Feast còn hỗ trợ thêm một lớp bảo vệ: **`filter_by_created_timestamp`**.

Vấn đề: Dữ liệu có thể bị **backfill** (sửa/cập nhật sau). Ví dụ:

```
┌──────────┬──────────────┬──────────────┬──────────────┐
│ driver_id│ event_timestamp │ value │ created_timestamp │
├──────────┼──────────────┼──────────────┼──────────────┤
│   1001   │ 2024-06-01   │   0.45       │ 2024-06-01   │ ← ghi đúng hôm đó
│   1001   │ 2024-06-01   │   0.55       │ 2024-06-03   │ ← backfill sau 2 ngày!
└──────────┴──────────────┴──────────────┴──────────────┘
```

Nếu không có `filter_by_created_timestamp`, Feast vẫn lấy giá trị `0.55` (dù nó được ghi sau). Nếu bật flag:

```python
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_hourly_stats:conv_rate"],
    filter_by_created_timestamp=True,  # ← chỉ lấy feature có created_timestamp <= event_timestamp
).to_df()
```

Kết quả: Feast chỉ lấy `0.45` vì `created_timestamp (2024-06-01) <= event_timestamp (2024-06-01)`.

> **Lưu ý:** Tính năng này yêu cầu feature view có cột `created_timestamp_column` được định nghĩa, và offline store phải hỗ trợ.

### 6.5 Tại sao point-in-time join quan trọng với Observability-AI?

Trong hệ thống observability/monitoring, dữ liệu thường đến **không đều** (irregular intervals):

```
03:00 — CPU 45%
03:05 — CPU 52%
03:12 — CPU 48%   ← bạn muốn predict tại mốc này
03:20 — CPU 61%
```

Khi join với các feature sources khác (số lượng request, memory usage, disk I/O...), nếu không có point-in-time join:

- Feature của **03:20** có thể bị lấy nhầm cho dự đoán ở **03:12** → model học sai.
- Feature của **03:12** là "tương lai" của feature source khác đến trễ → training sai.

Feast đảm bảo mỗi mẫu training chỉ nhìn thấy dữ liệu **tại đúng thời điểm nó xảy ra** — giống hệt những gì model sẽ thấy khi inference thực tế.

### 6.6 Tóm tắt

| Khái niệm | Vai trò chống leakage |
|---|---|
| `event_timestamp` | Upper bound — chỉ lấy feature có timestamp ≤ entity timestamp |
| Point-in-time join | Với mỗi entity row, lấy **latest value** trước mốc đó |
| `filter_by_created_timestamp` | Loại bỏ feature bị **backfill** sau thời điểm event |
| `get_historical_features` | Tự động thực hiện point-in-time join |
| Consistency | Cùng một FeatureService cho cả training và serving |

---

## 7. Vector search trong Feast

Feast hỗ trợ **vector search** qua các online store vector: **Qdrant, Milvus, Faiss**, và Elasticsearch.

### Cấu hình ví dụ (Milvus)

```yaml
project: local_rag
provider: local
registry: data/registry.db
online_store:
  type: milvus
  path: data/online_store.db
  vector_enabled: true
  embedding_dim: 384
  index_type: "IVF_FLAT"
offline_store:
  type: file
entity_key_serialization_version: 3
auth:
  type: no_auth
```

### Truy vấn vector (retrieve tài liệu)

```python
store = FeatureStore("src")
docs = store.retrieve_online_documents_v2(
    features=["docs_embeddings:passage"],
    query=embedding,
    top_k=3,
    distance_metric="COSINE",
).to_df()
```

Đây chính là pattern project `demo-observable-layer` (phiên bản v2) sử dụng.

---

## 8. Feast hoạt động tốt khi nào?

| Nên dùng Feast | Không cần Feast |
|---|---|
| Nhiều model dùng chung features | 1 model đơn giản, 1 bảng data |
| Cần consistency training/serving | Không có pipeline trực tuyến |
| Có streaming + batch | Chỉ chạy batch, không cần real-time |
| Team ML/data nhiều người, nhiều nhóm | Đồ án nhỏ, một người quản lý |

---

## 9. Tham khảo

- Docs: https://docs.feast.dev/
- Point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Data integrity: https://docs.feast.dev/reference/data-integrity
- Vector DB integration: https://docs.feast.dev/reference/online-stores/milvus
- RAG example: https://github.com/feast-dev/feast/tree/master/examples/rag
- GitHub: https://github.com/feast-dev/feast