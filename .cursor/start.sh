#!/usr/bin/env bash
# Per-boot service reconciliation for PickMeTalk Ops.
# Brings up PostgreSQL + Redis, waits for readiness, ensures the app role/db
# exist, then returns. Safe to run repeatedly.
set -euo pipefail

PG_VERSION="$(ls /etc/postgresql 2>/dev/null | sort -V | tail -1 || true)"
PG_VERSION="${PG_VERSION:-16}"

echo "[start] Ensuring PostgreSQL ${PG_VERSION} is running..."
if ! sudo pg_lsclusters -h 2>/dev/null | awk '{print $4}' | grep -q '^online$'; then
  sudo pg_ctlcluster "${PG_VERSION}" main start || true
fi

echo "[start] Ensuring Redis is running..."
if ! redis-cli ping >/dev/null 2>&1; then
  sudo redis-server /etc/redis/redis.conf --daemonize yes || true
fi

echo "[start] Waiting for PostgreSQL to accept connections..."
for _ in $(seq 1 30); do
  if sudo -u postgres pg_isready -q 2>/dev/null; then break; fi
  sleep 1
done

echo "[start] Ensuring app role and database exist..."
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='pickmetalk'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE ROLE pickmetalk LOGIN PASSWORD 'pickmetalk';"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='pickmetalk'" | grep -q 1 \
  || sudo -u postgres createdb -O pickmetalk pickmetalk

echo "[start] Services ready (PostgreSQL :5432, Redis :6379)."
