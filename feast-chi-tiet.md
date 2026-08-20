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

### 3.1 Entity — "Khoá chính" để nối features

**Entity** là định danh cho một đối tượng trong thế giới thực — `driver_id`, `user_id`, `sensor_id`...
Nó giống **primary key** trong database: dùng để nối features từ nhiều FeatureView khác nhau.

```python
from feast import Entity

# Entity: driver (join key = driver_id)
driver = Entity(
    name="driver",
    join_keys=["driver_id"],       # tên cột dùng để JOIN
    description="A driver in the fleet",
    value_type=ValueType.INT64,    # kiểu dữ liệu của join key
)
```

> **Ví dụ trực quan:** Entity giống bảng `drivers(id, name, phone)` trong SQL.
> Các bảng feature khác (`driver_trips`, `driver_ratings`) đều có cột `driver_id` → JOIN qua đó.

### 3.2 FeatureView — "Bảng ảo" chứa features

**FeatureView** là một nhóm features có liên quan, gắn với một data source, có TTL, và thuộc về một entity.

```python
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int32
from datetime import timedelta

# Nguồn dữ liệu (offline store)
driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
)

# FeatureView — giống như 1 VIEW trong SQL
driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=["driver"],                # belongs to entity "driver"
    ttl=timedelta(days=1),              # dữ liệu cũ hơn 1 ngày bỏ qua
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int32),
    ],
    source=driver_stats_source,
    online=True,                        # ← đưa lên online store để serve
)
```

**Tham số `online` và `offline`** — mặc định:

```python
online:  True   # FeatureView có sẵn trong online store → get_online_features()
offline: False  # FeatureView KHÔNG có sẵn trong offline store → get_historical_features()
```

Ý nghĩa:

| Flag | `True` | `False` |
|---|---|---|
| `online=True` | Feature được **materialize vào online store** (Milvus, Redis...) → `get_online_features()` hoạt động | Feature **không có trong online store** → `get_online_features()` lỗi |
| `offline=True` | Feature có thể được dùng để **train model** qua `get_historical_features()` | Feature **không dùng được cho training** |

**Ràng buộc:**
- `online=True` → bắt buộc phải có `source` (data source) và `entities` (trừ feature view không entity)
- `online=True` → dữ liệu phải có **event timestamp** (Feast cần để xác định giá trị mới nhất)
- `online=True` mà không `materialize` hoặc `write_to_online_store` → online store rỗng → `get_online_features()` trả về rỗng
- Có thể bật cả `online=True` và `offline=True` — feature vừa serve được vừa train được

> Nguồn: [Feast docs — Feature view](https://docs.feast.dev/getting-started/concepts/feature-view.md) | Source code: `FeatureView.__init__(online=True, offline=False)`

### 3.3 FeatureService — "Gói features" cho một model

**FeatureService** gom FeatureView lại thành 1 nhóm, dùng chung cho cả training và serving.
Đảm bảo **cùng một bộ features** được dùng ở cả 2 phase.

```python
from feast import FeatureService

# Gói features cho model "driver_activity_v1"
driver_activity_fs = FeatureService(
    name="driver_activity_v1",
    features=[
        driver_hourly_stats,                     # lấy tất cả fields
        driver_ratings_fv[["lifetime_rating"]],   # chỉ lấy 1 field
    ],
)
```

**Dùng FeatureService giúp:**

```python
# Training — dùng FeatureService
training_df = store.get_historical_features(
    feature_service="driver_activity_v1",   # ← gọi bằng tên
    entity_df=entity_df,
).to_df()

# Serving — cũng FeatureService đó
features = store.get_online_features(
    feature_service="driver_activity_v1",   # ← y hệt
    entity_rows=[{"driver_id": 1001}],
).to_dict()
```

> **Lợi ích:** Không cần nhớ "features gồm những cột nào" — cứ gọi tên service là đủ.
> Khi thêm/bớt feature, chỉ sửa 1 chỗ (FeatureService definition), không phải sửa code train + serve riêng.
>
> Nguồn: [Feast docs — FeatureService](https://docs.feast.dev/project/adr/adr-0001-feature-services.md)

```python
from feast import FeatureService

driver_ranking_service = FeatureService(
    name="driver_ranking",
    features=[driver_hourly_stats],
)
```

### 3.4 `online` / `offline` — Materialize vào online store

Tham số `online` và `offline` trong FeatureView quyết định feature có sẵn ở đâu:

| Flag | Mặc định | `True` | `False` |
|---|---|---|---|
| `online` | `True` | Feature được materialize vào **online store** → `get_online_features()` hoạt động | Feature **không có trong online store** → serve lỗi |
| `offline` | `False` | Feature có thể dùng **train model** qua `get_historical_features()` | Feature **không dùng được cho training** |

> Nguồn: [Feast docs — Feature view](https://docs.feast.dev/getting-started/concepts/feature-view.md) | Source code: `FeatureView.__init__(online=True, offline=False)`

**Ràng buộc khi `online=True`:**
1. Phải có `source` (data source) — Feast cần biết lấy dữ liệu từ đâu
2. Phải có `entities` (trừ feature view không entity)
3. Dữ liệu phải có **event timestamp** — Feast cần để xác định giá trị mới nhất
4. Phải **materialize** hoặc `write_to_online_store` — nếu không online store rỗng

#### Cách đưa dữ liệu vào online store

**Cách 1 — `feast materialize` (stateless):**

```bash
# Materialize tất cả feature views trong khoảng [start, end]
feast materialize 2024-01-07T00:00:00 2024-01-08T00:00:00

# Chỉ materialize 1 feature view cụ thể
feast materialize ... -v driver_hourly_stats
```

> Nguồn: [Feast docs — Load data into online store](https://docs.feast.dev/how-to-guides/feast-snowflake-gcp-aws/load-data-into-the-online-store)

**Cách 2 — `feast materialize-incremental` (stateful):**

```bash
# Lần 1: materialize từ đầu → 2024-01-08
feast materialize-incremental 2024-01-08T00:00:00
# Registry nhớ: "đã chạy đến 2024-01-08"

# Lần 2: tự động chạy từ 2024-01-08 → 2024-01-10 (chỉ data mới)
feast materialize-incremental 2024-01-10T00:00:00
```

> Nguồn: [Feast docs — materialize-incremental](https://docs.feast.dev/reference/feast-cli-commands.md)

**Cách 3 — `store.write_to_online_store(df)` (trực tiếp từ code):**

```python
store = FeatureStore("feature_repo")
store.write_to_online_store(
    feature_view_name="docs_embeddings",
    df=prepared_dataframe,
)
```

#### Cách Feast xử lý nhiều record cùng ID

Khi materialize trong khoảng thời gian, Feast **chỉ lấy giá trị mới nhất** của mỗi entity:

```
Offline store (parquet)                        Online store (Milvus)
┌──────────────────────────────────────┐       ┌──────────────────────┐
│ driver_id │ event_timestamp │ conv_rate│      │ driver_id │ conv_rate│
├──────────────────────────────────────┤       ├──────────────────────┤
│ 1001      │ 2024-01-07 08:00 │ 0.45  │       │ 1001      │ 0.72    │ ← latest
│ 1001      │ 2024-01-07 10:00 │ 0.52  │──materialize──▶│ 1002      │ 0.88    │
│ 1001      │ 2024-01-07 12:00 │ 0.72  │       └──────────────────────┘
│ 1002      │ 2024-01-07 09:00 │ 0.88  │
│ 1002      │ 2024-01-07 11:00 │ 0.91  │
└──────────────────────────────────────┘
```

Entity nào không có record trong khoảng thời gian → bị bỏ qua, không ảnh hưởng đến online store.

### 3.5 Data Source

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

**Tại sao không tự filter bằng code?**

Bạn nói đúng — về mặt lý thuyết viết:

```python
df_filtered = df_feature[df_feature.event_timestamp <= entity_ts]
```

cũng cho kết quả tương tự. Nhưng Feast làm thêm nhiều việc mà filter tay dễ sai:

| Việc | Filter tay | Feast |
|---|---|---|
| **Lấy latest value** | Phải `groupby + sort + first` | ✅ Tự động |
| **Nhiều FeatureView** | Phải JOIN nhiều lần, dễ sai | ✅ 1 lệnh `get_historical_features` |
| **Backfill** | Dữ liệu thêm vào quá khứ → filter tay không biết | ✅ Dùng `created_timestamp` loại bỏ |
| **Distributed (BigQuery, Spark)** | Phải viết SQL riêng từng nền tảng | ✅ Tự động |
| **Point-in-time chính xác** | `event_timestamp <= entity_ts` là đúng, nhưng quên `created_timestamp` | ✅ Dùng cả 2 |

**Cơ chế hoạt động:**

1. Với mỗi dòng trong `entity_df` (dữ liệu training), Feast nhìn vào cột `event_timestamp`.
2. Feast truy vấn `offline store` và chỉ lấy bản ghi feature có `event_timestamp <= entity_timestamp`.
3. Feast lấy bản ghi **gần nhất** (latest value) trước hoặc bằng mốc thời gian đó.
4. Feast kiểm tra `created_timestamp` để loại bỏ backfill (dữ liệu được thêm vào sau).
5. Kết quả: **không có giá trị "tương lai" nào lọt vào training set.**

> Nguồn: [Feast Docs — Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)

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

### 7.1 Provider trong Feast — `local` vs `gcp` vs `aws`

**Provider** quyết định hạ tầng *mặc định* cho offline store, online store, và compute:

| Provider | Offline store mặc định | Online store mặc định | Dùng khi |
|---|---|---|---|
| **`local`** | File (parquet) | SQLite | Phát triển local, máy cá nhân |
| **`gcp`** | Google BigQuery | Google Datastore | Deploy trên Google Cloud |
| **`aws`** | AWS Redshift | AWS DynamoDB | Deploy trên AWS |

> Nguồn: [Feast docs — Providers](https://docs.feast.dev/getting-started/components/provider.md)

**Quan trọng:** provider chỉ là **mặc định**. Bạn có thể **override** từng thành phần:

```yaml
provider: local                # mặc định offline=file, online=SQLite
online_store:
  type: milvus                 # override online store thành Milvus
```

Vậy `provider: local` + `online_store: milvus` = "dùng local cho mọi thứ, nhưng online store thì chạy Milvus".

### 7.2 Feast tìm feature repository bằng cách nào?

**Feature repository** là thư mục chứa:
1. `feature_store.yaml` — cấu hình Feast
2. Các file `.py` — định nghĩa `FeatureView`, `Entity`, `DataSource`...

> Nguồn: [Feast docs — Feature Repository](https://docs.feast.dev/reference/feature-repository.md)

Feast tìm repository theo cách sau:

```bash
# Cách 1 — feast apply trong thư mục hiện tại
cd feature_repo
feast apply
# → Feast đọc feature_store.yaml trong thư mục hiện tại
# → Quét tất cả file .py trong thư mục (đệ quy)
# → Parse các đối tượng Feast (FeatureView, Entity...)
# → Đồng bộ registry với định nghĩa tìm được
```

**Cụ thể khi gõ `feast apply`:**

```
feast apply
    │
    ├── 1. Tìm feature_store.yaml trong thư mục hiện tại
    │
    ├── 2. Quét toàn bộ file *.py (đệ quy)
    │      ├── feature_view.py → parse FeatureView
    │      ├── entity.py → parse Entity
    │      └── ...
    │
    ├── 3. So sánh với registry hiện tại
    │      ├── FeatureView mới → tạo
    │      ├── FeatureView thay đổi → cập nhật
    │      └── FeatureView bị xoá → xoá
    │
    └── 4. Deploy infrastructure
           └── Tạo collection trong Milvus, table trong Redis...
```

> ⚠️ **Vì Feast quét tất cả file `.py`**, bạn **không được để code app chung với file Feast**.
> Nếu `feature_store.yaml` và `feature_view.py` nằm cùng thư mục với code FastAPI,
> Feast sẽ cố import cả code app → lỗi `ModuleNotFoundError`.
>
> **Giải pháp:** Tách riêng `feature_repo/` chỉ chứa file Feast.
> ```tree
> feature_repo/
> ├── feature_store.yaml     ← Feast
> └── feature_view.py        ← Feast
> src/
> ├── services/              ← code app
> ├── routers/               ← code app
> └── ...
> ```
>
> Cách gọi từ code Python:
> ```python
> # Đường dẫn tương đối hoặc tuyệt đối đến thư mục chứa feature_store.yaml
> store = FeatureStore("feature_repo")
> ```

### 7.3 ⚠️ Lưu ý quan trọng — `provider: local` + Milvus = Milvus Lite

> Nguồn: [Feast docs — Milvus online store](https://docs.feast.dev/reference/online-stores/milvus.md) | [Feast source code — milvus.py](https://github.com/feast-dev/feast/blob/master/sdk/python/feast/infra/online_stores/milvus_online_store/milvus.py)

Tài liệu Feast **không đề cập rõ ràng** về hạn chế này, nhưng code Feast v0.49.0 có logic cứng:

```python
def _connect(self, config):
    if config.provider == "local":
        # BẮT BUỘC dùng Milvus Lite (file local)
        self.client = MilvusClient(db_path)      # ← path từ config
    else:
        # Dùng remote Milvus (host:port)
        self.client = MilvusClient(url=...)       # ← host:port từ config
```

Khi `provider: local`:
- Feast **luôn** gọi `MilvusClient(db_path)` — dùng Milvus Lite (file SQLite local)
- `host`/`port` trong config **bị bỏ qua hoàn toàn**
- Dù bạn có ghi `host: localhost, port: 19530` cũng vô dụng

Khi `provider: gcp` (hoặc `aws`):
- Feast gọi `MilvusClient(url=...)` — kết nối đến Docker Milvus thật
- `host`/`port` được dùng để tạo URL kết nối

**Giải pháp:** dùng `provider: gcp` và override `online_store → type: milvus` + `offline_store → type: file`:

```yaml
provider: gcp                 # ← bắt buộc để dùng remote Milvus
offline_store:
  type: file                  # override: không dùng BigQuery mặc định của gcp
online_store:
  type: milvus                # override: không dùng Datastore mặc định của gcp
  host: localhost
  port: 19530
```

> **Tài liệu chính thức không nói gì về hạn chế này.** Nó chỉ được phát hiện khi đọc source code.
> Có thể Feast sẽ sửa trong tương lai, nhưng hiện tại (v0.49.0) đây là behavior thật.

### 7.4 Cấu hình ví dụ (Milvus)

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