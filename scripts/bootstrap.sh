#!/bin/bash
#
# bootstrap.sh — stand up ufc_ml_agents from scratch on a fresh box.
#
# What `git clone` already gives you (do NOT need to fetch separately):
#   - all code + run.sh
#   - model artifacts (models/saved/)
#   - a committed snapshot of the fight DB (data/ufc_database.db)
#
# What is NOT in git and must be provisioned by this script:
#   - Python venv + dependencies
#   - the ELO sidecar (data/enrichment/sergey_sidecar.sqlite, ~253 MB) —
#     REGENERATED from Sergey's Postgres, never carried around as a binary.
#   - the freshest completed fights (the committed DB may lag by days/weeks)
#
# Usage:
#   ./scripts/bootstrap.sh                 # full setup (venv, sidecar, sync)
#   SKIP_SIDECAR=1 ./scripts/bootstrap.sh  # skip ELO export (no PG access here)
#   SKIP_SYNC=1   ./scripts/bootstrap.sh   # skip UFCStats catch-up sync
#
# Requires (for the sidecar step) access to Sergey's Postgres `ufc-test-1`.
# On a box WITHOUT that access, run with SKIP_SIDECAR=1 and instead scp the
# file from an existing box:
#   scp <existing-box>:/path/ufc_ml_agents/data/enrichment/sergey_sidecar.sqlite \
#       data/enrichment/sergey_sidecar.sqlite
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
SIDECAR="data/enrichment/sergey_sidecar.sqlite"
# Where Sergey's Java backend keeps the Postgres password (sourced at runtime,
# never printed). Override with PG_APP_PROPS if his layout differs.
PG_APP_PROPS="${PG_APP_PROPS:-/usr/project/ufc-dataset/backend/src/main/resources/application.properties}"

echo "==> Repo: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Python venv + dependencies
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> Creating venv ($VENV)"
  python3 -m venv "$VENV"
fi
echo "==> Installing dependencies"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 2. ELO sidecar — regenerate from Sergey's Postgres
# ---------------------------------------------------------------------------
if [[ "${SKIP_SIDECAR:-0}" == "1" ]]; then
  echo "==> SKIP_SIDECAR=1 — leaving $SIDECAR as-is"
  if [[ ! -f "$SIDECAR" ]]; then
    echo "    WARNING: $SIDECAR is missing. ELO features will be unavailable."
    echo "    scp it from an existing box (see header) or re-run without SKIP_SIDECAR."
  fi
else
  mkdir -p "$(dirname "$SIDECAR")"
  echo "==> Exporting ELO sidecar from Sergey's Postgres"
  if [[ -f "$PG_APP_PROPS" ]]; then
    # Source the password at runtime; it is never echoed or stored.
    PGPASSWORD="$(grep -E '^spring.datasource.password=' "$PG_APP_PROPS" | cut -d= -f2-)"
  fi
  if [[ -z "${PGPASSWORD:-}" ]]; then
    echo "    ERROR: no Postgres password (set PGPASSWORD or PG_APP_PROPS)." >&2
    echo "    Re-run with SKIP_SIDECAR=1 to skip and scp the file instead." >&2
    exit 1
  fi
  # Export to a temp file, validate, then atomically swap into place.
  "$VENV/bin/python" scripts/export_sergey_sidecar.py \
    --pg-user "${PG_USER:-postgres}" \
    --pg-password "$PGPASSWORD" \
    --output "$SIDECAR.new"
  # Finalize WAL -> single self-contained file so a live swap is safe.
  "$VENV/bin/python" - "$SIDECAR.new" <<'PY'
import sqlite3, sys
p = sys.argv[1]
c = sqlite3.connect(p)
c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
c.execute("PRAGMA journal_mode=DELETE")
assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "integrity check failed"
c.close()
print("  sidecar integrity: ok")
PY
  mv "$SIDECAR.new" "$SIDECAR"
  rm -f "$SIDECAR.new-wal" "$SIDECAR.new-shm" "$SIDECAR-wal" "$SIDECAR-shm"
  echo "==> Sidecar ready: $SIDECAR"
fi

# ---------------------------------------------------------------------------
# 3. Bring the fight DB current (the committed snapshot may be stale)
# ---------------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" == "1" ]]; then
  echo "==> SKIP_SYNC=1 — using committed DB snapshot as-is"
else
  echo "==> Syncing latest completed UFCStats fights into data/ufc_database.db"
  "$VENV/bin/python" scripts/sync_ufcstats_completed.py
fi

echo
echo "==> Bootstrap complete. Launch the app with:"
echo "      ./run.sh"
