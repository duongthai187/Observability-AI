# Embedding, Indexing & SentenceTransformer — Giải thích chi tiết

> Tài liệu này giải thích các khái niệm trong file `scripts/rag-mini-wikipedia-embeddings.parquet`
> và model `SentenceTransformer("all-MiniLM-L6-v2")` dùng trong project.

---

## 1. File `.parquet` là gì?

**Parquet** là định dạng file **cột (columnar)** do Apache phát triển, tối ưu cho lưu trữ và xử lý dữ liệu lớn.

### So sánh Parquet vs CSV vs JSON

```
CSV (hàng)              Parquet (cột)
┌──────┬──────┬──────┐  ┌──────────┬──────────┬──────────┐
│passage│embedding│...│  │ passage  │embedding │  ...     │
├──────┼──────┼──────┤  ├──────────┼──────────┼──────────┤
│"The.."│[0.1..]│...│  │"The.."   │[0.1..]   │          │
│"ML.." │[0.3..]│...│  │"ML.."    │[0.3..]   │          │
│"AI.." │[0.5..]│...│  │"AI.."    │[0.5..]   │          │
└──────┴──────┴──────┘  └──────────┴──────────┴──────────┘
  Lưu theo hàng           Lưu theo cột → nén tốt hơn
```

| Định dạng | Nén | Tốc độ đọc | Dùng cho |
|---|---|---|---|
| **Parquet** | ✅ Rất tốt (cột, snappy/zstd) | ✅ Nhanh (chỉ đọc cột cần) | Data pipeline, ML |
| CSV | ❌ Kém (văn bản thô) | ❌ Chậm (đọc toàn bộ) | Excel, debug |
| JSON | ❌ Trung bình | ❌ Chậm | API, web |

### Vì sao file này dùng Parquet?

File `rag-mini-wikipedia-embeddings.parquet` chứa **5332 passages** (đoạn văn từ Wikipedia) kèm **vector embeddings** đã được tính sẵn. Parquet được chọn vì:

1. **Mỗi embedding là một mảng 384 số float32** (1536 bytes) — Parquet nén cột này rất tốt nhờ các giá trị cùng kiểu.
2. **Chỉ cần đọc cột `passage`** khi cần xem nội dung, không cần load toàn bộ embedding vào RAM.
3. File nén chỉ **~11 MB** — nếu để CSV có thể gấp 3-5 lần.

### Hai cột trong file

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `passage` | `string` | Đoạn văn bản gốc từ Wikipedia (tiếng Anh) |
| `embedding` | `list[float32]` (384 chiều) | Vector biểu diễn đoạn văn đó, đã được tính sẵn bằng `all-MiniLM-L6-v2` |

```python
import pandas as pd

df = pd.read_parquet("scripts/rag-mini-wikipedia-embeddings.parquet")
print(df.shape)       # (5332, 2)
print(df.columns)     # ['passage', 'embedding']
print(df.iloc[0])     # passage: "The Amazon rainforest...", embedding: [0.014, -0.022, ...]
print(len(df.iloc[0]["embedding"]))  # 384
```

---

## 2. "Index dữ liệu" là làm gì?

### 2.1 Index trong database truyền thống (PostgreSQL)

> Nguồn: [PostgreSQL docs — Indexes](https://www.postgresql.org/docs/current/indexes.html) | [B-Tree implementation](https://www.postgresql.org/docs/current/btree.html)

**Index trong PostgreSQL là một cấu trúc dữ liệu riêng biệt, được lưu ở vùng riêng trên đĩa (index pages), tách biệt với dữ liệu gốc (heap table).**

Bạn nói đúng — nó không phải là "mục lục sách" đơn thuần, mà là một **cấu trúc dữ liệu riêng** được build từ dữ liệu gốc, giúp định vị bản ghi **mà không cần quét toàn bộ bảng (sequential scan)**.

#### Cấu trúc B-tree index trong PostgreSQL

```
Table (Heap):                             Index (B-tree):               
┌──────────────────────────────┐         ┌──────────────────────────────┐
│ Page 0:                      │         │  Root Page (metapage)        │
│  [id=1, name="Alice", ...]   │         │  ┌────────────────────────┐  │
│  [id=2, name="Bob", ...]     │         │  │ id=50 → page 100       │  │
│  ...                         │         │  │ id=100 → page 200      │  │
├──────────────────────────────┤         │  └────────────────────────┘  │
│ Page 100:                    │         ├──────────────────────────────┤
│  [id=50, name="Eve", ...]    │    ──▶  │  Internal Pages              │
│  [id=51, name="Frank", ...]  │         │  ┌────────────────────────┐  │
│  ...                         │         │  │ id=50-75 → page 150    │  │
├──────────────────────────────┤         │  │ id=75-100 → page 180   │  │
│ Page 200:                    │         │  └────────────────────────┘  │
│  [id=100, name="Grace", ...] │         ├──────────────────────────────┤
│  ...                         │         │  Leaf Pages (99%+)           │
└──────────────────────────────┘         │  ┌────────────────────────┐  │
                                          │  │ id=50 → TID=(100,1)   │  │
 Dữ liệu gốc được lưu trong heap          │  │ id=51 → TID=(100,2)   │  │
 theo dạng page (thường 8KB/page).        │  │ ...                   │  │
 Không có index → phải đọc tất cả          │  │ id=100 → TID=(200,1)  │  │
 pages để tìm 1 bản ghi.                  │  └────────────────────────┘  │
                                          └──────────────────────────────┘
                                          Index là cấu trúc riêng,
                                          lưu key + pointer (TID) đến
                                          dòng dữ liệu thực tế trong heap.
```

**Cách hoạt động:**

```sql
-- Không có index: Sequential Scan — đọc hết 10 triệu dòng
EXPLAIN ANALYZE SELECT * FROM users WHERE id = 50000;
-- Seq Scan on users  (cost=0.00..213744.00 rows=1)
--   (actual time=357.059..357.059 rows=1)
--   → Đọc toàn bộ bảng (63744 pages)

-- Có B-tree index: Index Scan — chỉ đọc ~3 pages
EXPLAIN ANALYZE SELECT * FROM users WHERE id = 50000;
-- Index Scan using users_pkey on users  (cost=0.42..8.44 rows=1)
--   (actual time=0.035..0.036 rows=1)
--   → root page → internal page → leaf page → 1 heap page
--   → Chỉ 3-4 pages thay vì 63744 pages!
```

**Tóm lại: Index trong PostgreSQL là một cấu trúc dữ liệu riêng (B-tree, Hash, GiST, GIN, v.v.)**:
- Được lưu trên **index pages riêng**, không nằm chung với dữ liệu gốc
- Chứa **giá trị key + con trỏ (TID = page number + offset)** trỏ đến dòng dữ liệu trong heap
- Khi tìm kiếm: đi từ root → internal → leaf → lấy TID → đọc đúng 1 page trong heap
- Index-only scan còn có thể trả kết quả **trực tiếp từ index mà không cần đọc heap**

### 2.2 Index trong vector database (Qdrant)

Trong Qdrant, có **2 loại index** riêng biệt:

#### a) Vector Index (HNSW) — cho tìm kiếm ngữ nghĩa

> Nguồn: [Qdrant docs — Vector Index](https://qdrant.tech/documentation/manage-data/indexing/)

Là cấu trúc **đồ thị đa tầng** (HNSW graph), giúp tìm vector gần nhất mà không cần quét hết toàn bộ dữ liệu. Index này được Qdrant tự động xây khi bạn upsert documents.

```
┌─────────────────────────────────────────────────────────────────┐
│  Qdrant Collection "passages"                                   │
│                                                                │
│  Points (dữ liệu gốc):                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ id=1 │ vector=[0.014, -0.022, ...] │ payload={text:...}  │   │
│  │ id=2 │ vector=[0.087, 0.033, ...]  │ payload={text:...}  │   │
│  │ ...  │ ...                         │ ...                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  HNSW Index (cấu trúc riêng, build từ vector):                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Layer 2: [A]────────[C]────────[E]                    │   │
│  │  Layer 1: [A]──[B]──[C]──[D]──[E]──[F]                │   │
│  │  Layer 0: [A]──[B]──[C]──[D]──[E]──[F]──[G]──[H]...   │   │
│  │  (Mỗi node lưu pointer đến các node lân cận)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  Payload Index (cho filter metadata):                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  "city:London" → [id=1, id=5, id=12]                   │   │
│  │  "city:Paris"  → [id=3, id=8, id=20]                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

> **ℹ️ Payload là gì?**
>
> **Payload** là metadata (dạng JSON) được gắn kèm với mỗi vector (point) trong Qdrant.
> Nó là nơi lưu **thông tin gốc của dữ liệu** — không phải vector, không phải index.
>
> Trong project này, mỗi payload chỉ gồm `{"text": passage}` — đó là **plain text của đoạn văn gốc** từ file parquet.
>
> ```python
> # Khi index, payload được tạo từ passage gốc:
> payload = {"text": passage}  # "The Amazon rainforest is a moist..."
> ```
>
> Khi search, Qdrant trả về cả vector lẫn payload:
> ```python
> hits = client.search(
>     collection_name="passages",
>     query_vector=query_emb,
>     limit=5,
>     with_payload=True,  # ← lấy luôn text gốc từ payload
> )
> # hits[0].payload["text"] → "The Amazon rainforest..."
> # → text này được đưa vào LLM làm context để trả lời
> ```
>
> **Vai trò của payload:**
> - Lưu **dữ liệu gốc** (text, JSON, số...) — những gì bạn cần sau khi tìm được vector phù hợp
> - Có thể filter/search theo payload (vd: chỉ lấy passage có category = "AI")
> - Không tham gia vào tính toán vector similarity, chỉ là thông tin đi kèm
>
> **Có thể lưu nhiều thứ hơn ngoài text:**
> ```python
> payload = {
>     "text": "The Amazon rainforest...",
>     "title": "Amazon Rainforest",
>     "category": "Geography",
>     "word_count": 245,
>     "source_url": "https://en.wikipedia.org/wiki/Amazon_rainforest",
>     "created_at": "2024-01-15",
> }
> ```

#### b) Payload Index — cho lọc metadata (giống index database truyền thống)

> Nguồn: [Qdrant docs — Payload Index](https://qdrant.tech/documentation/manage-data/payload/)

```python
# Tạo payload index trên field "category" — giống CREATE INDEX trong PostgreSQL
client.create_payload_index(
    collection_name="passages",
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
```

Payload index hoạt động giống index trong PostgreSQL: build một cấu trúc riêng (B-tree/map) để filter nhanh theo metadata, không cần quét hết tất cả points.

### 2.3 Tổng kết: Index KHÔNG phải chỉ là "mục lục sách"

| Khía cạnh | Mục lục sách | Index thật (PostgreSQL B-tree) | Index vector (Qdrant HNSW) |
|---|---|---|---|
| Là gì? | Danh sách liệt kê | Cấu trúc cây (B-tree) riêng | Cấu trúc đồ thị đa tầng riêng |
| Lưu ở đâu? | Trong sách | Index pages riêng trên đĩa | RAM/disk riêng, tách khỏi points |
| Nội dung | Từ khóa → số trang | Key + TID (pointer đến heap) | Vector node + edges (pointer đến node khác) |
| Cách tìm | Tra bảng → mở trang | Đi root → internal → leaf → heap | Đi layer 2 → layer 1 → layer 0 |
| Mục đích | Tra cứu từ khóa | Tìm bản ghi theo điều kiện | Tìm vector gần nhất (ANN) |

**Ví dụ cụ thể với PostgreSQL:**

```sql
CREATE INDEX idx_users_name ON users (name);
-- Index này là cấu trúc B-tree riêng, chứa:
--   "Alice"   → TID = (block=100, offset=1)
--   "Bob"     → TID = (block=100, offset=2)
--   "Charlie" → TID = (block=250, offset=5)

SELECT * FROM users WHERE name = 'Charlie';
-- 1. Không index: quét 100.000 rows (sequential scan)
-- 2. Có index: B-tree → root → internal → leaf → TID → 1 page
--    → Chỉ 3-4 page reads
```

**Ví dụ trong Qdrant:**

```python
client.search(collection_name="passages", query_vector=query_emb, limit=5)
-- 1. Không HNSW: so sánh với 5332 vectors (brute force)
-- 2. Có HNSW: đi graph đa tầng → ~20-50 lần so sánh
```

### 2.4 Trong project `index_to_qdrant.py`

File `scripts/index_to_qdrant.py` làm những việc sau:

1. **Đọc** file `.parquet` → lấy `passage` và `embedding`
2. **Tạo collection** trong Qdrant với cấu hình vector (384 chiều, distance = COSINE)
3. **Upsert** (insert/update) từng passage vào Qdrant dưới dạng `point`:
   - `id`: mã MD5 ổn định từ nội dung passage
   - `vector`: mảng 384 số float
   - `payload`: `{"text": passage}` — metadata kèm theo
4. Qdrant **tự động xây HNSW index** cho các vectors

```python
# Tóm tắt quá trình index:
# 1. Đọc parquet
df = pd.read_parquet("rag-mini-wikipedia-embeddings.parquet")

# 2. Tạo Qdrant collection
client.recreate_collection(
    collection_name="passages",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 3. Upsert từng batch
for batch in chunks:
    client.upsert(collection_name="passages", points=batch)
    # Qdrant tự động xây HNSW graph!
```

> **Kết quả**: Sau khi index, 5332 passages đã sẵn sàng để search. Khi có câu hỏi, code chỉ cần:
> 1. Encode câu hỏi → vector 384 chiều
> 2. `client.search(collection_name="passages", query_vector=query_emb, limit=5)`
> 3. Qdrant dùng HNSW graph tìm 5 passage gần nhất trong ~ms

---

## 3. Thuật toán HNSW (Hierarchical Navigable Small World) — Chi tiết

> Nguồn chính thức:
> - Paper gốc: [Malkov & Yashunin, 2018 — "Efficient and robust approximate nearest neighbor search using HNSW"](https://arxiv.org/abs/1603.09320)
> - Qdrant docs: https://qdrant.tech/documentation/manage-data/indexing/
> - Hnswlib: https://github.com/nmslib/hnswlib

### 3.1 Bài toán: Tìm kiếm láng giềng gần nhất (Nearest Neighbor Search)

Cho 5332 vectors, mỗi vector 384 chiều. Với một vector query, tìm **5 vector gần nhất**.

- **Cách ngây thơ (brute force)**: Tính khoảng cách từ query đến **cả 5332 vectors** → chọn 5 cái gần nhất → O(n) = 5332 lần tính.
- **Cách thông minh (HNSW)**: Xây một cấu trúc đa tầng → chỉ cần đi qua ~10-20 điểm → O(log n).

### 3.2 Nguồn gốc: Ý tưởng từ "Small World" (Thế giới nhỏ)

**Hiện tượng "Small World"** (còn gọi là **6 degrees of separation**):

> Trong một mạng xã hội, bất kỳ hai người nào cũng chỉ cách nhau tối đa 6 mối quan hệ.
> Bạn của bạn của bạn của bạn của bạn của bạn — chắc chắn có người bạn cần tìm!

```
Ví dụ: Bạn muốn tìm một kỹ sư AI ở Nhật
Bạn ── Bạn A ── Bạn B ── Bạn C ── Bạn D ── Kỹ sư AI
  (1)      (2)      (3)      (4)      (5)      (6)
Chỉ 6 bước là tới!
```

HNSW áp dụng ý tưởng này vào không gian vector: các vector được nối với nhau thành một **đồ thị (graph)**. Dù có 5332 vector, bạn chỉ cần đi qua vài bước là tới được vector gần nhất.

### 3.3 Kiến trúc: Đồ thị đa tầng (Multi-layer Graph)

> Nguồn: [Qdrant docs — Vector Index](https://qdrant.tech/documentation/manage-data/indexing/) | [Hnswlib docs](https://github.com/nmslib/hnswlib)

Hãy tưởng tượng bạn đang tìm một quán phở ngon ở Hà Nội. Bạn sẽ làm gì?

```
Cách 1 — Không có bản đồ (Brute Force):
  Đi bộ từng con phố, gõ cửa từng nhà hỏi "có bán phở không?"
  → Rất lâu, có thể mất cả ngày!

Cách 2 — Có Google Maps (HNSW):
  🔍 "phở ngon Hà Nội" → Zoom ra xem tổng quan
  → Thấy quận Hoàn Kiếm có nhiều quán → Zoom vào
  → Thấy phố Hàng Bông → Zoom vào chi tiết
  → Tìm được quán phở ưng ý trong 5 phút!
```

**HNSW cũng làm y hệt Google Maps — nó xây nhiều "tầng zoom" (layers) cho dữ liệu vector:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  NHÌN TỪ XA — Layer 2 (zoom xa nhất, ít điểm nhất)                 │
│  Chỉ thấy những điểm "nổi bật" nhất, cách xa nhau                   │
│                                                                     │
│         [A]──────────────────────────────[C]                        │
│              (chỉ 2-3 điểm trên toàn bộ dataset)                    │
│                                                                     │
│  NHÌN GẦN HƠN — Layer 1 (zoom vừa)                                  │
│  Thấy nhiều điểm hơn, có thể phân biệt các cụm                      │
│                                                                     │
│     [A]────[B]────[C]────[D]────[E]────[F]────[G]                  │
│              (vài chục điểm)                                        │
│                                                                     │
│  NHÌN CHI TIẾT — Layer 0 (zoom gần nhất, đầy đủ nhất)               │
│  Thấy TẤT CẢ các điểm, từng chi tiết nhỏ                           │
│                                                                     │
│  [A]─[B]─[C]─[D]─[E]─[F]─[G]─[H]─[I]─[J]─[K]─[L]─[M]─[N]─[O]─[P]  │
│  [Q]─[R]─[S]─[T]─[U]─[V]─[W]─[X]─[Y]─[Z]─[AA]─[AB]─[AC]─[AD]...  │
│              (tất cả 5332 điểm)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mỗi node là một vector, mỗi cạnh là một "đường đi" nhanh giữa 2 vector gần nhau.**

Quan trọng: **mỗi node có thể xuất hiện ở nhiều tầng, nhưng không nhất thiết phải là tất cả các tầng.** Node nào "quan trọng" (trung tâm của một cụm) thì xuất hiện ở nhiều tầng hơn.

### 3.4 Cách xây dựng (Indexing) — Từng bước cụ thể

> Nguồn: [HNSW paper — Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)

Khi thêm 1 vector mới vào HNSW, có 3 bước:

#### Bước 1: "May rủi" — Xác định vector này sẽ ở những tầng nào

HNSW dùng công thức xác suất để quyết định:

```
L = floor(-ln(rand(0,1)) * 1/ln(M))

Với M = 16:
  rand = 0.5 → L = floor(-ln(0.5) * 0.36) = floor(0.25) = 0  (chỉ layer 0)
  rand = 0.1 → L = floor(-ln(0.1) * 0.36) = floor(0.83) = 0  (chỉ layer 0)
  rand = 0.01 → L = floor(-ln(0.01) * 0.36) = floor(1.66) = 1 (layer 0 + 1)
  rand = 0.001 → L = floor(-ln(0.001) * 0.36) = floor(2.49) = 2 (layer 0 + 1 + 2)
```

**Kết quả với 5332 vectors:**
- ~87% vectors → chỉ ở layer 0 (bình thường)
- ~12% vectors → layer 0 + 1 (hơi quan trọng)
- ~1% vectors → layer 0 + 1 + 2 (rất quan trọng — "điểm mốc")

#### Bước 2: "Đi từ trên xuống" — Tìm vị trí cho vector mới

Giả sử thêm vector [X] vào, và [X] được phân vào layer 0 + 1:

```
Bắt đầu từ entry point (layer 2):
                                    
  Layer 2:    [A] ──────────── [C] ──────────── [E]
                  ^
                  Bắt đầu từ A, so sánh [X] với [A], [C], [E]
                  → [C] gần [X] nhất → xuống layer 1

  Layer 1:  [A] ── [B] ── [C] ── [D] ── [E] ── [F] ── [G]
                              │
                              Từ [C], đi đến các node lân cận
                              [C] → [B] → [D] → [E] → [F]
                              Luôn giữ ef_construction = 100 node
                              gần nhất trong hàng đợi
                              → [F] gần [X] nhất

  Layer 0:  [A]─[B]─[C]─[D]─[E]─[F]─[G]─[H]─[I]─[J]─[K]─[L]...
                                    │
                                    Từ [F], mở rộng tìm kiếm
                                    Giữ 100 node gần nhất
                                    → tìm ra M=16 node gần [X] nhất
```

#### Bước 3: "Kết bạn" — Nối cạnh

```
Kết quả: [X] gần nhất với [F], [G], [H], [K], [M]...

  Layer 1:  [F] ── [X] ── [G]   (nối với tối đa M cạnh)
                      │
                      [H]        (các cạnh này là "đường đi nhanh")

  Layer 0:  [E] ── [F] ── [X] ── [G] ── [H]
                      │      │
                      [K] ───┘
                      [M] ───┘

  Các node cũ [F], [G], [H] cũng được cập nhật:
  trước: [F]─[G]─[H]─[E]─[D]     (5 cạnh)
  sau:   [F]─[G]─[H]─[E]─[D]─[X] (6 cạnh, thêm [X])
  Nếu quá M cạnh → xóa cạnh xa nhất
```

### 3.5 Cách tìm kiếm (Search) — Ví dụ đi từ A đến Z

**Giả sử bạn search: "What is machine learning?"**

Trong 5332 passages, vector của câu hỏi là [Q] (384 chiều).

#### Giai đoạn 1: Định hướng nhanh (tầng trên cùng)

```
Layer 2 (3 điểm):
  [A] ───────────────── [C] ───────────────── [E]
  
  Bước 1: So sánh [Q] với [A]     → khoảng cách = 0.85
  Bước 2: So sánh [Q] với [C]     → khoảng cách = 0.42 ← gần hơn
  Bước 3: So sánh [Q] với [E]     → khoảng cách = 0.91

  → [C] là gần nhất → đi xuống layer 1, bắt đầu từ [C]
```

#### Giai đoạn 2: Tinh chỉnh (tầng giữa)

```
Layer 1 (30 điểm):
  [A]──[B]──[C]──[D]──[E]──[F]──[G]──[H]──[I]──[J]──[K]──[L]──[M]──[N]──[O]──[P]──[Q]──[R]──[S]──[T]

  Bắt đầu từ [C], nhìn vào các node kề cạnh của [C]: [B], [D]
  
  Priority queue (hàng đợi ưu tiên — luôn giữ node gần nhất ở đầu):
  ┌──────┬──────────────┐
  │ Node │ Khoảng cách  │  ← gần [Q] nhất
  ├──────┼──────────────┤
  │  [C] │    0.42      │
  │  [D] │    0.38      │  ← [D] gần hơn [C]!
  │  [B] │    0.51      │
  └──────┴──────────────┘

  Tiếp: từ [D] → [E], [F]
  ┌──────┬──────────────┐
  │  [F] │    0.29      │  ← [F] gần hơn nữa!
  │  [E] │    0.31      │
  │  [D] │    0.38      │
  │  [C] │    0.42      │
  │  [B] │    0.51      │
  └──────┴──────────────┘

  Tiếp: từ [F] → [G], [H]
  ┌──────┬──────────────┐
  │  [G] │    0.25      │  ← gần nhất hiện tại
  │  [H] │    0.27      │
  │  [F] │    0.29      │
  │  [E] │    0.31      │
  │  ... │    ...       │
  └──────┴──────────────┘

  Không còn node nào gần hơn [G] → xuống layer 0
```

#### Giai đoạn 3: Mở rộng chi tiết (tầng 0, với ef_search = 50)

```
Layer 0 (5332 điểm):
  Bắt đầu từ [G], mở rộng dần:
  
  ┌──────────┐
  │ Hàng đợi │ (luôn giữ 50 node gần nhất)
  ├──────────┤
  │ [G] 0.25 │ ← đầu queue
  │ [H] 0.27 │
  │ [F] 0.29 │
  │ [E] 0.31 │
  │ [K] 0.33 │ ← từ G đi đến K
  │ [M] 0.34 │ ← từ G đi đến M
  │ [L] 0.36 │ ← từ H đi đến L
  │ ...      │
  │ [J] 0.78 │ ← cuối queue (xa nhất trong 50 node)
  └──────────┘

  Tiếp tục mở rộng:
  - Lấy [G] ra khỏi queue → xem các node kề [G] → thêm vào queue
  - Lấy [H] ra → xem node kề [H] → thêm vào queue
  - Lấy [K] ra → xem node kề [K] → thêm vào queue
  - ...
  - Dừng khi node đầu queue không còn gần hơn được nữa

  Queue cuối cùng (top 5):
  ┌──────────┬──────────────┬──────────────────────┐
  │  Node    │ Khoảng cách  │  Passage tương ứng   │
  ├──────────┼──────────────┼──────────────────────┤
  │  [G]     │    0.25      │ "Machine learning is │
  │          │              │  a subset of AI..."  │
  │  [H]     │    0.27      │ "Deep learning uses  │
  │          │              │  neural networks..." │
  │  [K]     │    0.33      │ "Supervised learning │
  │          │              │  requires labeled..." │
  │  [M]     │    0.34      │ "Neural networks are │
  │          │              │  inspired by..."     │
  │  [F]     │    0.36      │ "Training data is   │
  │          │              │  essential for..."   │
  └──────────┴──────────────┴──────────────────────┘

  → Trả về 5 passages này cho LLM để tổng hợp câu trả lời!
```

### 3.6 Tại sao HNSW nhanh và chính xác?

**So sánh trực quan:**

```
Cách Brute Force (tìm trong 5332 passages):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Bạn phải ghé thăm TỪNG CĂN NHÀ trong 1 khu phố để tìm bạn     │
  │  → 5332 lần gõ cửa, 5332 lần "xin chào, bạn có phở không?"     │
  └─────────────────────────────────────────────────────────────────┘

Cách HNSW (tìm trong 5332 passages):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Bước 1: Leo lên đỉnh tòa nhà cao nhất, nhìn bao quát          │
  │          → thấy khu vực có nhiều quán ăn (layer 2)              │
  │  Bước 2: Chạy xe đến khu đó, nhìn bảng hiệu phố              │
  │          → thấy phố ẩm thực (layer 1)                          │
  │  Bước 3: Đi bộ vào từng quán, hỏi thử                          │
  │          → tìm được quán ngon (layer 0)                        │
  │  → Chỉ ~20-50 lần "gõ cửa" thay vì 5332 lần!                   │
  └─────────────────────────────────────────────────────────────────┘
```

**Độ chính xác recall (với M=16, Qdrant mặc định):**

| ef_search | Recall @10 | Số lần so sánh | Tương đương |
|---|---|---|---|
| 50 | ~95% | ~20-30 | Tìm gần đúng, rất nhanh |
| 100 | ~97% | ~40-60 | Cân bằng tốc độ/chính xác |
| 200 | ~99% | ~80-120 | Rất chính xác, hơi chậm |
| Brute force | 100% | 5332 | Chính xác tuyệt đối, chậm |

> **Trong RAG, recall ~95% là quá đủ** vì LLM sẽ tổng hợp câu trả lời từ context. Thiếu 1 passage "đúng" nhưng 4 passages kia vẫn đủ thông tin để LLM trả lời tốt.

### 3.7 Ví dụ thực tế: 5332 passages trong project

```
Dung lượng HNSW index với 5332 vectors 384D:
  - M = 16 (mỗi node có 16 cạnh)
  - Mỗi cạnh = 8 byte (pointer) + 4 byte (khoảng cách) ≈ 12 byte
  - Tổng cạnh ≈ 5332 × 16 × 0.5 (trung bình) ≈ 42.656 cạnh
  - Dung lượng index ≈ 42.656 × 12 byte ≈ 0.5 MB

  + dữ liệu gốc (vector thô): 5332 × 384 × 4 byte ≈ 8.2 MB
  → Tổng ~9 MB — rất nhẹ!

Tốc độ search:
  - Brute force: 5332 lần tính cosine ≈ 5-10 ms
  - HNSW (ef=100): ~40 lần tính cosine ≈ 0.05-0.1 ms
  → Nhanh hơn ~100 lần!

Ví dụ cụ thể:
  Bạn search "What is deep learning"
  → HNSW tìm được 5 passages:
    1. "Deep learning is a subset of machine learning..." (0.27)
    2. "Neural networks form the basis of deep learning..." (0.31)
    3. "Deep learning requires large amounts of data..." (0.35)
    4. "Convolutional neural networks are used in computer vision..." (0.38)
    5. "Training deep neural networks requires GPUs..." (0.42)
  
  → LLM nhận 5 passages này → tổng hợp câu trả lời hoàn chỉnh
```

### 3.8 Tham số HNSW trong Qdrant

Trong project, Qdrant dùng HNSW với các tham số mặc định:

```yaml
# Cấu hình mặc định của Qdrant cho HNSW
hnsw_index:
  m: 16                    # Mỗi node có tối đa 16 cạnh
  ef_construct: 100        # Xây index: xem xét 100 ứng viên
  full_scan_threshold: 10000  # Nếu < 10KB vector thì scan thay vì HNSW
```

Có thể override khi tạo collection:

```python
from qdrant_client.http import models

client.recreate_collection(
    collection_name="passages",
    vectors_config=models.VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(
        m=32,                 # nhiều cạnh hơn → chính xác hơn, tốn RAM hơn
        ef_construct=200,     # xây index kỹ hơn → chính xác hơn, lâu hơn
    ),
)
```

### 3.9 HNSW hoạt động với 5332 passages như thế nào?

Với 5332 vectors 384 chiều trong project:

```
Dung lượng index:
  - 5332 vectors × 384 số × 4 bytes (float32) = ~8.2 MB (dữ liệu thô)
  - HNSW graph: mỗi node có 16 cạnh × 8 bytes (pointer) = ~0.7 MB
  - Tổng cộng: ~9 MB trong RAM

Tốc độ search:
  - Brute force: 5332 lần tính cosine ≈ 5-10 ms
  - HNSW (ef=100): ~20-50 lần tính cosine ≈ 0.1-0.5 ms
  → Nhanh hơn ~50-100 lần!

Độ chính xác:
  - HNSW với ef=100: recall ≈ 97-99% (so với brute force)
  - 1-3% passages "sai" nhưng thường là passages gần tương đương
  → Trong RAG, LLM vẫn tổng hợp được câu trả lời tốt
```

### 3.10 Tóm tắt HNSW bằng một câu

> **HNSW xây một "kim tự tháp" nhiều tầng: tầng trên để định hướng nhanh, tầng dưới để tìm chính xác. Khi search, nó đi từ trên xuống dưới, mỗi bước một gần hơn — giống như bạn tra từ điển: mở mục lục trước, rồi đến trang, rồng đến dòng.**

---

## 4. `SentenceTransformer("all-MiniLM-L6-v2")` là gì?

### 4.1 Tổng quan

**SentenceTransformer** là một thư viện Python (từ [sbert.net](https://sbert.net/)) chuyên dùng để chuyển đổi **câu/văn bản thành vector embeddings** (dense vector) — tức là biến chữ thành số.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Đầu vào: văn bản
sentences = [
    "The Amazon rainforest is a moist broadleaf forest",
    "Machine learning is a subset of artificial intelligence",
]

# Đầu ra: vector embeddings (mảng số)
embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, 384)
print(embeddings[0])     # [0.014, -0.022, 0.087, ...] 384 số
```

### 4.2 "all-MiniLM-L6-v2" là gì?

Đây là tên của một **pre-trained model** cụ thể trên Hugging Face Hub. Hãy bóc tách từng phần:

```
all-MiniLM-L6-v2
├── all        → model đa năng (all-round), không chuyên biệt cho 1 lĩnh vực
├── MiniLM     → tên kiến trúc: MiniLM = phiên bản nhỏ gọn hơn của BERT
│                  (BERT: 340M params → MiniLM: 22.7M params)
├── L6         → 6 tầng Transformer (BERT-base có 12 tầng)
│                  → ít tầng hơn = nhanh hơn, nhẹ hơn
└── v2         → phiên bản thứ 2 (cải thiện so với v1)
```

**Thông số kỹ thuật:**

| Thông số | Giá trị | Ý nghĩa |
|---|---|---|
| Kích thước embedding | **384** (cố định) | Mỗi câu → vector 384 số |
| Số tầng Transformer | 6 | Nhẹ hơn BERT-base (12 tầng) |
| Số tham số | ~22.7M | Nhỏ hơn BERT-base (~110M) ~5 lần |
| Tốc độ encode | ~10.000 câu/giây (GPU) | Rất nhanh |
| Kích thước model | ~90MB | Dễ tải, dễ deploy |
| Ngôn ngữ | Tiếng Anh | Được train trên: 1B+ cặp câu từ nhiều nguồn (Reddit, Wikipedia, news...) |

> **💡 384 chiều là cố định, không phải cấu hình được.**
> Nó được quyết định bởi kiến trúc của model: `all-MiniLM-L6-v2` có **hidden size = 384** ở tầng Transformer cuối cùng, kết hợp với mean pooling → ra vector 384 chiều. Đây là con số "đóng hộp" từ nhà sản xuất.
>
> Nếu muốn embedding dimension khác, bạn phải **đổi model**:
>
> | Model | Embedding dimension | Kích thước |
> |---|---|---|
> | `all-MiniLM-L6-v2` | **384** ✅ | 22.7M params |
> | `all-mpnet-base-v2` | **768** | 109M params |
> | `text-embedding-3-small` (OpenAI) | **1536** | API |
> | `text-embedding-3-large` (OpenAI) | **3072** | API |
>
> **Lưu ý:** Khi đổi model, bạn phải **re-index toàn bộ dữ liệu** vì Qdrant collection đã cấu hình `size=384`. Nếu dùng model 768D, collection cũ không dùng được — phải tạo collection mới với `size=768`.

### 4.3 Cơ chế hoạt động bên trong

> Nguồn: [Hugging Face Tokenizers — WordPiece](https://huggingface.co/docs/tokenizers/en/components#models) | [Hugging Face Course — Tokenization](https://huggingface.co/learn/nlp-course/chapter2/4)

Khi bạn gọi `model.encode("The Amazon rainforest is a moist broadleaf forest")`, bên trong xảy ra 4 bước:

#### Bước 1: Tokenizer — Tách câu thành token

> Nguồn: [Hugging Face Course — WordPiece](https://huggingface.co/learn/nlp-course/chapter6/6)

Đây là bước **chia văn bản thành các mảnh nhỏ (token)**. Model `all-MiniLM-L6-v2` dùng thuật toán **WordPiece** (giống BERT).

**"Từ điển" (vocabulary) là gì?**

Khi model được train (huấn luyện), nó học một **bảng từ vựng (vocabulary)** gồm **~30.000 token phổ biến nhất** — đây là "từ điển" của model.

```
Ví dụ về vocabulary của BERT/MiniLM (trích):
┌──────────┬────────┐
│  Token   │  ID    │
├──────────┼────────┤
│ [CLS]    │ 101    │  ← token đặc biệt: đầu câu
│ [SEP]    │ 102    │  ← token đặc biệt: cuối câu
│ [UNK]    │ 100    │  ← token đặc biệt: không biết
│ the      │ 1996   │
│ amazon   │ 4249   │
│ rain     │ 4434   │  ← "rain" có trong từ điển
│ ##forest │ 3722   │  ← "##forest" có trong từ điển
│ is       │ 2003   │
│ a        │ 1037   │
│ moist    │ 16839  │
│ broad    │ 4810   │
│ ##leaf   │ 13668  │
│ forest   │ 1460   │
│ ...      │ ...    │  ← ~30.000 tokens tất cả
│ 😊       │ ???    │  ← KHÔNG có trong từ điển → [UNK]
└──────────┴────────┘
```

**Cách WordPiece dùng "từ điển" để tách từ:**

```
Thuật toán WordPiece — Greedy, từ trái sang phải:
─────────────────────────────────────────────────────────

Từ "rainforest":

  Bước 1: "rainforest" có trong từ điển không? → ❌ Không
  Bước 2: "rainfores" có trong từ điển không?  → ❌
  Bước 3: "rainfore" có trong từ điển không?   → ❌
  ... (lùi dần từng ký tự)
  Bước 7: "rain" có trong từ điển không?       → ✅ CÓ! (ID=4434)
  
  → Lấy "rain", phần còn lại: "forest"
  → Thêm "##" vào đầu: "##forest"
  
  Bước 8: "##forest" có trong từ điển không?   → ✅ CÓ! (ID=3722)
  
  → Kết quả: "rainforest" → ["rain", "##forest"]

Từ "broadleaf":

  Bước 1: "broadleaf" có trong từ điển không? → ❌
  Bước 2: "broadle" → "broad"✅ → còn "leaf" → "##leaf"✅
  → Kết quả: "broadleaf" → ["broad", "##leaf"]

Từ "😊" (emoji):

  Bước 1: "😊" có trong từ điển không? → ❌
  Bước 2: lùi mãi không tìm thấy gì → [UNK] (Unknown)
  → Kết quả: "😊" → ["[UNK]"]
```

**Dấu `##` có nghĩa là gì?**

```
"##" = "đây là phần tiếp theo của một từ, không phải từ riêng"

  rain      → từ hoàn chỉnh (có thể đứng một mình)
  ##forest  → mảnh ghép, phải đi với từ phía trước

Khi decode ngược lại:
  ["rain", "##forest"] → "rainforest" (bỏ ## và ghép lại)
  ["broad", "##leaf"]  → "broadleaf"
```

**Tóm lại: "có trong từ điển" = có trong vocabulary ~30.000 tokens của model.**
Nếu từ/tiếng không có → WordPiece cố tách thành các mảnh nhỏ hơn. Nếu vẫn không có → `[UNK]` (Unknown).

**Ví dụ cụ thể với câu "The Amazon rainforest is a moist broadleaf forest":**

```
Đầu vào: "The Amazon rainforest is a moist broadleaf forest"

Bước 1a — Pre-tokenize (tách từ bằng khoảng trắng):
  ["The", "Amazon", "rainforest", "is", "a", "moist", "broadleaf", "forest"]

Bước 1b — WordPiece (tách từ ghép thành subword):
  ["the", "amazon", "rain", "##forest", "is", "a", "moist", "broad", "##leaf", "forest"]
     1       2        3        4          5    6      7        8        9        10

Bước 1c — Thêm token đặc biệt [CLS] đầu câu, [SEP] cuối câu:
  ["[CLS]", "the", "amazon", "rain", "##forest", "is", "a", "moist", "broad", "##leaf", "forest", "[SEP]"]
     0        1       2        3        4        5     6     7        8        9        10       11

→ Tổng cộng: 12 tokens (bao gồm [CLS] và [SEP])
```

> **Số token KHÔNG cố định** — phụ thuộc vào độ dài câu và cách WordPiece tách từ.
> Câu ngắn → ít token, câu dài → nhiều token (tối đa 256 tokens do `max_seq_length=256`).

#### Bước 2: Transformer — Mỗi token → 1 vector 384 chiều

Sau khi tokenize, mỗi token được chuyển thành **một vector 384 chiều** qua 6 tầng Transformer:

```
Đầu ra của Tokenizer:        Đầu vào Transformer:        Đầu ra Transformer:
12 token IDs                 12 embedding vectors        12 vectors 384D
                              (trainable embedding)
┌──────────┐                ┌──────────────────┐        ┌──────────────────┐
│ [CLS]    │                │ [0.01, 0.02, ...]│        │ [0.12, 0.08, ...]│  ← vector 1
│ the      │                │ [0.03, 0.01, ...]│        │ [0.15, 0.03, ...]│  ← vector 2
│ amazon   │  ──── lookup ──▶│ [0.02, 0.04, ...]│─── 6 ─▶│ [0.09, 0.11, ...]│  ← vector 3
│ rain     │                │ [0.01, 0.03, ...]│ layers │ [0.07, 0.14, ...]│  ← vector 4
│ ##forest │                │ [0.04, 0.02, ...]│        │ [0.13, 0.06, ...]│  ← vector 5
│ ...      │                │ ...              │        │ ...              │
│ [SEP]    │                │ [0.02, 0.01, ...]│        │ [0.10, 0.05, ...]│  ← vector 12
└──────────┘                └──────────────────┘        └──────────────────┘
Token IDs                   Token embeddings            Contextual embeddings
(12 số)                     (12 × 384 = 4608 số)        (12 × 384 = 4608 số)
```

**Mỗi tầng Transformer làm gì? — Self-Attention từ thấp đến cao**

> Nguồn: [Hugging Face Course — Encoders & Self-Attention](https://huggingface.co/learn/nlp-course/chapter1/5) | [Transformer paper — Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

Điều kỳ diệu của Transformer nằm ở cơ chế **Self-Attention**: mỗi token có thể "nhìn" vào tất cả các token khác trong câu và quyết định token nào quan trọng với nó.

**Cơ chế Self-Attention trong 1 tầng:**

```
Ví dụ: Token "rain" đang cần hiểu ngữ cảnh của nó.

Bước 1 — Query: "rain" tự hỏi: "Tôi nên chú ý đến token nào?"
Bước 2 — Key:   Mỗi token khác giơ bảng: "tôi là the", "tôi là amazon", "tôi là ##forest"...
Bước 3 — Score: "rain" tính điểm cho từng token:
                ┌──────────┬──────────┬───────────┐
                │  Token   │  Score   │  Ý nghĩa  │
                ├──────────┼──────────┼───────────┤
                │  the     │   0.02   │ không liên quan │
                │  amazon  │   0.15   │ hơi liên quan   │
                │  rain    │   0.30   │ bản thân nó     │ ← highest
                │  ##forest│   0.28   │ rất liên quan!  │ ← gần highest
                │  is      │   0.05   │                   │
                │  ...     │   ...    │                   │
                └──────────┴──────────┴──────────────────┘
Bước 4 — Kết hợp: "rain" = 0.30×rain + 0.28×##forest + 0.15×amazon + ...
  → "rain" đã hiểu nó đi với "##forest" để tạo thành "rainforest"!
```

**Các tầng học được các cấp độ ngữ cảnh khác nhau:**

```
TẦNG 1 — Local syntax (cú pháp gần):
─────────────────────────────────────────────────────────────────────
  Mỗi token nhìn các token bên cạnh → hiểu quan hệ cục bộ.

  "rain" ──── "##forest"                "moist" ──── "broad"
      ╲       ╱                             ╲       ╱
      ╲     ╱                                ╲     ╱
      [rainforest]                        [moist broad...]
  
  "the" ── "amazon" ── "rainforest"     "a" ── "moist" ── "broadleaf"

  → Tầng 1 học: "rain" + "##forest" = "rainforest" (từ ghép)
                  "broad" + "##leaf" = "broadleaf" (từ ghép)
                  "the" + "amazon" = "the Amazon" (cụm danh từ)

TẦNG 2 — Phrase (cụm từ):
─────────────────────────────────────────────────────────────────────
  Token bắt đầu nhìn xa hơn → hiểu cụm từ.

  [the]──[amazon]──[rainforest]     [moist]──[broadleaf]──[forest]
     ╲      ╱      ╱                    ╲      ╱        ╱
      ╲    ╱    ╱                        ╲    ╱      ╱
       [the Amazon rainforest]     [moist broadleaf forest]

  → Tầng 2 học: "the Amazon rainforest" là 1 cụm
                  "moist broadleaf forest" là 1 cụm
                  "is a" là 1 cụm động từ

TẦNG 3 — Chunk (mệnh đề):
─────────────────────────────────────────────────────────────────────
  Token nhìn rộng hơn → hiểu mối quan hệ giữa các cụm.

  [the Amazon rainforest] ──── [is] ──── [a moist broadleaf forest]
          ╲                    ╱                    ╲
           ╲                  ╱                      ╲
              Chủ ngữ (subject)     Vị ngữ (predicate)
  
  → Tầng 3 học: "the Amazon rainforest" là chủ ngữ
                  "is a moist broadleaf forest" là vị ngữ
                  "rainforest" + "forest" = cùng chủ đề (rừng)

TẦNG 4 — Semantic role (vai trò ngữ nghĩa):
─────────────────────────────────────────────────────────────────────
  Token hiểu vai trò ngữ nghĩa trong câu.

  "Amazon" → "location" (địa danh)
  "rainforest" → "thing being described" (vật được mô tả)
  "moist broadleaf" → "property" (thuộc tính)

  → "The Amazon rainforest" = [LOCATION] [THING]
  → "moist broadleaf forest" = [PROPERTY] [PROPERTY] [THING]

TẦNG 5 — Global context (ngữ cảnh toàn cầu):
─────────────────────────────────────────────────────────────────────
  Token nhìn TOÀN BỘ câu → hiểu bức tranh tổng thể.

  "The Amazon rainforest is a moist broadleaf forest"
   ↑                                              ↑
   └──────────── Đây là câu mô tả ────────────────┘
   (không phải câu hỏi, không phải câu mệnh lệnh)
   (chủ đề: sinh thái / địa lý)
   (giọng văn: khách quan, khoa học)

  → Token "amazon" biết nó là một phần của câu mô tả địa lý
  → Token "forest" biết nó không phải forest ở cuối câu chuyện

TẦNG 6 — Fine-grained (tinh chỉnh cuối cùng):
─────────────────────────────────────────────────────────────────────
  Tinh chỉnh các biểu diễn, loại bỏ nhiễu, chuẩn hóa.

  Kết quả cuối: mỗi token có 1 vector 384 chiều chứa:
  - Thông tin về bản thân token (rain, forest...)
  - Thông tin về vị trí trong câu (chủ ngữ, vị ngữ...)
  - Thông tin về ngữ nghĩa toàn cầu (câu mô tả, chủ đề sinh thái...)
  - Thông tin về quan hệ với các token khác (rain ghép với forest...)
```

**Trực quan hóa sự thay đổi qua các tầng:**

```
Tầng 0 (đầu vào):  Mỗi token chỉ biết bản thân nó
  "the"  → [the]
  "rain" → [rain]
  "##forest" → [forest]

Tầng 1-2:          Token bắt đầu "thấy" hàng xóm
  "rain" → [rain + forest]       ← đã biết "rainforest"
  "##forest" → [forest + rain]   ← đã biết nó là đuôi của "rainforest"

Tầng 3-4:          Token "thấy" xa hơn, hiểu cấu trúc câu
  "rain" → [rainforest + Amazon + is + ...] ← đã biết nó là chủ ngữ

Tầng 5-6:          Token "thấy" toàn bộ câu, hiểu ngữ nghĩa tinh tế
  "rain" → [cả câu + ngữ cảnh + chủ đề] ← biểu diễn hoàn chỉnh
```

**Tóm lại: 6 tầng Transformer giống như 6 bước "đọc hiểu" của con người:**

```
Tầng 1: Nhận ra từ ghép           (rain + ##forest → rainforest)
Tầng 2: Nhận ra cụm từ            (the Amazon rainforest)
Tầng 3: Nhận ra chủ - vị          (subject → predicate)
Tầng 4: Hiểu vai trò ngữ nghĩa    (location, property, thing)
Tầng 5: Hiểu ngữ cảnh toàn cầu    (câu mô tả, chủ đề sinh thái)
Tầng 6: Tinh chỉnh + tổng hợp     (biểu diễn cuối cùng)
```

#### Bước 3: Pooling — Gộp 12 vectors thành 1 vector

Sau Transformer, ta có 12 vectors (mỗi token 1 vector). Cần **gộp** thành 1 vector duy nhất đại diện cho cả câu:

```
12 vectors tokens:                         1 vector câu:
┌──────────────────┐                      ┌──────────────────┐
│ [CLS] → 0.12 ... │                      │                  │
│ the   → 0.15 ... │                      │                  │
│ amazon→ 0.09 ... │                      │  0.11, 0.08,    │
│ rain  → 0.07 ... │  ──── Mean Pool ───▶ │  0.05, 0.12,    │
│ ##for → 0.13 ... │     (trung bình      │  ... 384 số     │
│ ...              │      tất cả token)   │                  │
│ [SEP] → 0.10 ... │                      │                  │
└──────────────────┘                      └──────────────────┘
  12 × 384                                 1 × 384
```

Công thức: `vector_câu = (v[CLS] + v[the] + v[amazon] + ... + v[SEP]) / 12`

#### Bước 4: Normalize — Chuẩn hóa độ dài vector = 1

Bước cuối, vector được chuẩn hóa để có độ dài (norm) = 1:

```
Trước normalize:  [0.11, 0.08, 0.05, ...]  độ dài = 2.5
Sau normalize:    [0.04, 0.03, 0.02, ...]  độ dài = 1.0
```

**Tại sao cần normalize?** — Để cosine similarity chạy đúng:

```
cosine(A, B) = (A·B) / (||A|| × ||B||)

Nếu ||A|| = ||B|| = 1 → cosine(A, B) = A·B (đơn giản hơn!)
```

#### Tổng kết luồng dữ liệu:

```
"The Amazon rainforest is a moist broadleaf forest"
  │
  ▼
Tokenizer (WordPiece)
  │
  ├── ["[CLS]", "the", "amazon", "rain", "##forest", "is", "a", 
  │    "moist", "broad", "##leaf", "forest", "[SEP]"]
  │    (12 tokens)
  │
  ▼
Transformer (6 tầng MiniLM)
  │
  ├── 12 vectors, mỗi vector 384 chiều  (12 × 384)
  │
  ▼
Mean Pooling
  │
  ├── 1 vector 384 chiều  (1 × 384)
  │
  ▼
Normalize
  │
  ├── 1 vector 384 chiều, độ dài = 1
  │
  ▼
Kết quả: [0.04, 0.03, 0.02, ...]  ← embedding của câu
```

### 4.4 Tại sao gọi là "Sentence Transformer"?

Có 3 giai đoạn phát triển:

```
1. Word Embeddings (Word2Vec, GloVe, 2013)
   Mỗi từ → 1 vector
   "king" → [0.1, 0.3, ...]
   "queen" → [0.2, 0.1, ...]
   ❌ Không biểu diễn được cả câu

2. Contextual Embeddings (BERT, 2019)
   Mỗi từ → 1 vector nhưng có ngữ cảnh
   "bank" (bờ sông) ≠ "bank" (ngân hàng)
   ✅ Hiểu ngữ cảnh
   ❌ Không tối ưu cho cả câu

3. Sentence Embeddings (Sentence-BERT, 2020)
   Cả câu → 1 vector duy nhất
   "The Amazon rainforest is a moist broadleaf forest"
     → [0.014, -0.022, 0.087, ...]  384 số
   ✅ Tối ưu cho so sánh ngữ nghĩa giữa các câu
   ✅ Nhanh hơn BERT gốc ~10.000 lần khi so sánh cặp câu
```

**Sentence-BERT (SBERT)** cải tiến BERT bằng cách thêm **pooling layer** và huấn luyện với **contrastive learning** (cặp câu tương tự/không tương tự). Kết quả: các câu có nghĩa giống nhau sẽ có vector gần nhau trong không gian 384 chiều.

### 4.5 Minh họa: cosine similarity

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Các câu về cùng chủ đề → vector gần nhau
sentences_a = ["A cat sits on the mat", "A feline is resting on a rug"]
emb_a = model.encode(sentences_a)

# Câu khác chủ đề → vector xa nhau
sentences_b = ["The stock market crashed today"]
emb_b = model.encode(sentences_b)

# Tính cosine similarity
sim = model.similarity(emb_a, emb_b)
print(sim)  # tensor([[0.85, 0.12]])
# 0.85 → "cat" gần "feline" (cùng nghĩa ✅)
# 0.12 → "cat" xa "stock market" (khác nghĩa ✅)
```

### 4.6 Vai trò trong project

```
┌──────────────────────────────────────────────────────────────┐
│                    QUY TRÌNH RAG ĐẦY ĐỦ                       │
│                                                              │
│  Bước 1: INDEX (làm 1 lần, offline)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Wikipedia passages  ──→  all-MiniLM-L6-v2  ──→  Qdrant │   │
│  │  (văn bản gốc)             (encode)           (vector DB)│   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Bước 2: SEARCH (mỗi lần hỏi)                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  "What is ML?"  ──→  all-MiniLM-L6-v2  ──→  Qdrant   │   │
│  │  (câu hỏi)            (encode 384D)     (search HNSW) │   │
│  │                                              │         │   │
│  │                                              ▼         │   │
│  │                                      Top 5 passages    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Bước 3: GENERATE                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Context (5 passages) + Question ──→  LLM → Answer    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**Tại sao dùng model nhỏ (all-MiniLM-L6-v2, 22.7M params) thay vì model lớn hơn?**

| Model | Kích thước | Tốc độ | Độ chính xác semantic |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22.7M params, 384D | ⚡️ Rất nhanh | 84.0 (Bảng xếp hạng) |
| `all-mpnet-base-v2` | 109M params, 768D | 🐢 Chậm hơn | 86.0 (cao hơn 2%) |
| `text-embedding-3-large` (OpenAI) | API | ⏳ Phụ thuộc network | 87.0 (cao hơn 3%) |

Trong RAG, **tốc độ** thường quan trọng hơn **độ chính xác tuyệt đối** vì:
- Bước retrieve chỉ là bước lọc sơ bộ → LLM sẽ tổng hợp câu trả lời sau đó
- Chênh lệch 2-3% accuracy ít ảnh hưởng hơn so với latency tăng gấp 3-5 lần

---

## 5. Tổng kết — Cả quy trình từ file đến câu trả lời

```
File .parquet                    SentenceTransformer             Qdrant
┌────────────────┐              ┌──────────────────┐          ┌──────────────┐
│ passage        │              │                  │          │              │
│ "Amazon..."    │──text──────▶ │ all-MiniLM-L6-v2  │─vector──▶│ HNSW Index   │
│ passage        │              │ (encode passage)  │   384D   │  ┌─┐ ┌─┐     │
│ "ML is..."     │──text──────▶ │                  │─vector──▶│  │ │ │ │     │
│ ...            │              └──────────────────┘   384D   │  └─┘ └─┘     │
│ embedding      │                                                      │
│ [0.014, ...]   │  (embedding đã tính sẵn, bỏ qua bước này)            │
│ [0.087, ...]   │                                                      │
└────────────────┘                                                      │
                                                                         │
Câu hỏi mới: "What is ML?"                                               │
       │                                                                 │
       ▼                                                                 │
┌──────────────────┐                                                    │
│ all-MiniLM-L6-v2  │─vector 384D ───────────────────────────────────────┤
│ (encode question) │                                                    │
└──────────────────┘                            ▼                        │
                                          ┌──────────────────┐          │
                                          │ Tìm 5 vector     │          │
                                          │ gần nhất (HNSW)  │          │
                                          │ → 5 passages     │          │
                                          └──────────────────┘          │
                                                   │                    │
                                                   ▼                    │
                                          ┌──────────────────┐          │
                                          │ LLM + Context    │          │
                                          │ → Answer         │          │
                                          └──────────────────┘          │
```

---

## 6. Tham khảo

- Sentence-Transformers docs: https://sbert.net/
- `all-MiniLM-L6-v2` model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Apache Parquet: https://parquet.apache.org/
- PostgreSQL Indexes: https://www.postgresql.org/docs/current/indexes.html
- PostgreSQL B-Tree implementation: https://www.postgresql.org/docs/current/btree.html
- PostgreSQL Pageinspect (xem cấu trúc index pages): https://www.postgresql.org/docs/current/pageinspect.html
- Qdrant Vector Index (HNSW): https://qdrant.tech/documentation/manage-data/indexing/
- Qdrant Payload Index: https://qdrant.tech/documentation/manage-data/payload/
- Qdrant HNSW configuration: https://qdrant.tech/documentation/ops-configuration/configuration
- HNSW algorithm paper (Malkov & Yashunin, 2018): https://arxiv.org/abs/1603.09320
- Hnswlib (C++ implementation reference): https://github.com/nmslib/hnswlib
- Navigable Small World explanation: https://en.wikipedia.org/wiki/Small-world_network