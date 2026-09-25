#!/usr/bin/env bash
# Idempotent repository bootstrap for PickMeTalk Ops.
# Installs Node deps, prepares a local .env, brings up services, and syncs +
# seeds the database. Re-runnable against cached/partial state.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# System services are normally baked into the environment snapshot. Install
# them here only if a fresh base image is missing them.
if ! command -v pg_ctlcluster >/dev/null 2>&1 || ! command -v redis-server >/dev/null 2>&1; then
  echo "[install] Installing PostgreSQL + Redis..."
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    postgresql postgresql-contrib redis-server
fi

echo "[install] Installing Node dependencies (root)..."
npm ci

echo "[install] Installing Node dependencies (web)..."
npm ci --prefix web

# Create a local .env if one is not present. Real Firebase/Supabase secrets can
# be layered on later via environment secrets; these local defaults are enough
# to run the API, web UI, worker, and database end-to-end.
if [ ! -f .env ]; then
  echo "[install] Generating local .env with fresh VAPID keys..."
  VAPID_OUT="$(npx tsx scripts/generate-vapid-keys.ts 2>/dev/null || true)"
  VAPID_PUBLIC_KEY="$(printf '%s\n' "$VAPID_OUT" | grep '^VAPID_PUBLIC_KEY=' | cut -d= -f2- || true)"
  VAPID_PRIVATE_KEY="$(printf '%s\n' "$VAPID_OUT" | grep '^VAPID_PRIVATE_KEY=' | cut -d= -f2- || true)"
  cat > .env <<EOF
DATABASE_URL="postgresql://pickmetalk:pickmetalk@localhost:5432/pickmetalk"
REDIS_URL="redis://localhost:6379"
PORT=3000
PUBLIC_BASE_URL="http://localhost:3000"

VAPID_PUBLIC_KEY="${VAPID_PUBLIC_KEY}"
VAPID_PRIVATE_KEY="${VAPID_PRIVATE_KEY}"
VAPID_SUBJECT="mailto:support@pickmetalk.com"

FIREBASE_PROJECT_ID=""
FIREBASE_CLIENT_EMAIL=""
FIREBASE_PRIVATE_KEY=""

NODE_ENV=development

PHOTO_UNIVERSE_ENABLED=true
PHOTO_UNIVERSE_DATA_ROOT="./data/photo-universe"
EOF
fi

# Bring up PostgreSQL + Redis so schema sync and seeding can run.
bash "$REPO_ROOT/.cursor/start.sh"

echo "[install] Syncing Prisma schema to the database..."
npx prisma db push

echo "[install] Seeding database (idempotent)..."
npm run db:seed

echo "[install] Done."
