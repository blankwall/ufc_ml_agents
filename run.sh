#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/fastapi_app/runtime.env"

# Load operator-provided overrides / secrets first (e.g. THE_ODDS_API_KEY).
# Anything set in runtime.env takes precedence over the defaults below.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# ---------------------------------------------------------------------------
# Background-sync configuration.
#
# These defaults keep the app's data current on their own. Each var only fills
# in if it was not already provided by runtime.env or the surrounding shell
# (":=" = assign-if-unset), so operator overrides always win.
#
# NOTE: THE_ODDS_API_KEY is a secret and is intentionally NOT defaulted here.
# Set it in fastapi_app/runtime.env (or export it before running).
# ---------------------------------------------------------------------------

# The Odds API — market odds ingestion
: "${THE_ODDS_API_AUTO_SYNC:=1}"
: "${THE_ODDS_API_SYNC_INTERVAL_HOURS:=1}"
: "${THE_ODDS_API_SYNC_CHECK_SECONDS:=3600}"
: "${THE_ODDS_API_WINDOW_DAYS:=31}"

# UFCStats — completed-fight ingestion (keeps fighter records / results current)
: "${UFCSTATS_AUTO_SYNC:=1}"
: "${UFCSTATS_SYNC_INTERVAL_HOURS:=1}"
: "${UFCSTATS_SYNC_CHECK_SECONDS:=3600}"
# Lookback + per-run caps are set generously enough to catch up after a gap
# while staying polite. Lower them in runtime.env for pure steady-state.
: "${UFCSTATS_COMPLETED_LOOKBACK_DAYS:=21}"
: "${UFCSTATS_COMPLETED_MAX_PAGES:=2}"
: "${UFCSTATS_COMPLETED_MIN_FIGHTS:=5}"
: "${UFCSTATS_COMPLETED_MAX_EVENTS_PER_RUN:=3}"

# Sherdog — fighter-history recovery for records missing from UFCStats
: "${SHERDOG_RECOVERY_ENABLED:=1}"
: "${SHERDOG_RECOVERY_MAX_FIGHTERS_PER_RUN:=10}"

export \
  THE_ODDS_API_AUTO_SYNC THE_ODDS_API_SYNC_INTERVAL_HOURS \
  THE_ODDS_API_SYNC_CHECK_SECONDS THE_ODDS_API_WINDOW_DAYS \
  UFCSTATS_AUTO_SYNC UFCSTATS_SYNC_INTERVAL_HOURS UFCSTATS_SYNC_CHECK_SECONDS \
  UFCSTATS_COMPLETED_LOOKBACK_DAYS UFCSTATS_COMPLETED_MAX_PAGES \
  UFCSTATS_COMPLETED_MIN_FIGHTS UFCSTATS_COMPLETED_MAX_EVENTS_PER_RUN \
  SHERDOG_RECOVERY_ENABLED SHERDOG_RECOVERY_MAX_FIGHTERS_PER_RUN

cd "$SCRIPT_DIR/fastapi_app"

exec ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload
