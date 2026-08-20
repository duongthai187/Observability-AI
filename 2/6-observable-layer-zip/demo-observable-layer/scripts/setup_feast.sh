#!/bin/bash
# ============================================================
# Feast setup script — chạy SAU KHI docker compose up -d
# ============================================================
# Tự động:
#   1. Tạo database + user Feast trong PostgreSQL (nếu chưa có)
#   2. feast apply (đăng ký feature view)
#   3. Index data từ parquet vào Milvus
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FEATURE_REPO="$PROJECT_DIR/feature_repo"
VENV="$PROJECT_DIR/.venv"

echo "=== 1. PostgreSQL: tạo database + user ==="
docker exec infra-observable-layer-postgres-1 psql -U postgres -c "CREATE DATABASE feast;" 2>/dev/null || echo "   DB 'feast' đã tồn tại, bỏ qua"
docker exec infra-observable-layer-postgres-1 psql -U postgres -c "CREATE USER airflow WITH PASSWORD 'airflow';" 2>/dev/null || echo "   User 'airflow' đã tồn tại, bỏ qua"
docker exec infra-observable-layer-postgres-1 psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE feast TO airflow;" 2>/dev/null
docker exec infra-observable-layer-postgres-1 psql -U postgres -d feast -c "GRANT ALL ON SCHEMA public TO airflow;" 2>/dev/null
echo "   ✅ PostgreSQL ready"

echo ""
echo "=== 2. feast apply ==="
cd "$FEATURE_REPO"
"$VENV/bin/feast" apply
echo "   ✅ feast apply done"

echo ""
echo "=== 3. Index data ==="
cd "$PROJECT_DIR"
"$VENV/bin/python" scripts/index_to_feast.py
echo "   ✅ Index done"

echo ""
echo "🎉 Feast setup hoàn tất! Có thể chạy RAG service."