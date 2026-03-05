#!/usr/bin/env bash
# =============================================================================
# scrape_all_odds.sh
#
# Scrapes moneyline odds from BestFightOdds.com for every UFC event in the
# database (2007+). Results are appended to data/odds/historical_odds.csv.
#
# Usage:
#   bash scrapers/scrape_all_odds.sh
#   bash scrapers/scrape_all_odds.sh --from-year 2020   # only 2020+ events
#   bash scrapers/scrape_all_odds.sh --dry-run           # print queries, don't fetch
#
# Requirements:
#   - Run from the project root: cd /path/to/ufc_ml_agents
#   - Python venv active: source .venv/bin/activate
#   - bestfightodds_scraper.py in scrapers/
#   - data/odds/all_event_names.txt exists (generated below if missing)
#
# Notes:
#   - Each request is cached in .cache/bfo/ — re-running is cheap (no re-fetch)
#   - BFO rate limit is 2s between requests (built into scraper)
#   - Events with no BFO data are skipped automatically (logged as warnings)
#   - Estimated runtime: ~681 events × 4s avg = ~45 minutes first run,
#     ~2 minutes on subsequent runs (cached)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT="$PROJECT_ROOT/data/odds/historical_odds.csv"
EVENT_LIST="$PROJECT_ROOT/data/odds/all_event_names.txt"
LOG_FILE="$PROJECT_ROOT/data/odds/scrape_log.txt"
PYTHON="${PYTHON:-python}"
FROM_YEAR=2007
DRY_RUN=false
BATCH_SIZE=10    # scrape N events per python call (reduces subprocess overhead)

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-year)
            FROM_YEAR="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --output)
            OUTPUT="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Ensure event list exists ──────────────────────────────────────────────────
if [[ ! -f "$EVENT_LIST" ]]; then
    echo "[setup] Generating event list from database..."
    $PYTHON - <<'PYEOF'
import sqlite3, pandas as pd
from datetime import datetime
from pathlib import Path

conn = sqlite3.connect('data/ufc_database.db')
df = pd.read_sql("SELECT name, date FROM events", conn)

def parse_date(d):
    for fmt in ['%B %d, %Y', '%B %Y']:
        try:
            return datetime.strptime(d.strip(), fmt)
        except:
            pass
    return None

df['parsed_date'] = df['date'].apply(parse_date)
df = df.dropna(subset=['parsed_date']).sort_values('parsed_date').reset_index(drop=True)
df_bfo = df[df['parsed_date'].dt.year >= 2007].copy()

Path('data/odds').mkdir(parents=True, exist_ok=True)
with open('data/odds/all_event_names.txt', 'w') as f:
    for _, row in df_bfo.iterrows():
        f.write(row['name'] + '\n')
print(f"Generated {len(df_bfo)} event names → data/odds/all_event_names.txt")
PYEOF
fi

# ── Load events, filter by year ───────────────────────────────────────────────
# For year filtering we need dates — pull from DB, filter, then match names
EVENTS_JSON=$($PYTHON - <<PYEOF
import sqlite3, pandas as pd, json, sys
from datetime import datetime

conn = sqlite3.connect('data/ufc_database.db')
df = pd.read_sql("SELECT name, date FROM events", conn)

def parse_date(d):
    for fmt in ['%B %d, %Y', '%B %Y']:
        try:
            return datetime.strptime(d.strip(), fmt)
        except:
            pass
    return None

df['parsed_date'] = df['date'].apply(parse_date)
df = df.dropna(subset=['parsed_date']).sort_values('parsed_date').reset_index(drop=True)
df_bfo = df[(df['parsed_date'].dt.year >= $FROM_YEAR) & (df['parsed_date'].dt.year <= 2025)].copy()
print(json.dumps(df_bfo['name'].tolist()))
PYEOF
)

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL=$(echo "$EVENTS_JSON" | $PYTHON -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "============================================================"
echo " BestFightOdds Scraper — Batch Run"
echo "============================================================"
echo " Events to scrape : $TOTAL (from year $FROM_YEAR)"
echo " Output file       : $OUTPUT"
echo " Cache dir         : $PROJECT_ROOT/.cache/bfo/"
echo " Log file          : $LOG_FILE"
echo " Dry run           : $DRY_RUN"
echo "============================================================"
echo ""

if $DRY_RUN; then
    echo "[dry-run] Would scrape:"
    echo "$EVENTS_JSON" | $PYTHON -c "
import json, sys
events = json.load(sys.stdin)
for i, e in enumerate(events):
    print(f'  {i+1:4d}. {e}')
"
    exit 0
fi

mkdir -p "$(dirname "$LOG_FILE")"
echo "Scrape started: $(date)" > "$LOG_FILE"

# ── Run scraper in batches ────────────────────────────────────────────────────
# We chunk the event list into batches of BATCH_SIZE to avoid one Python
# process running for the entire 45-minute duration. Each batch appends to
# the same output CSV (deduplication is handled by the scraper).

$PYTHON - <<PYEOF
import json, subprocess, sys, time
from pathlib import Path

events = json.loads("""$EVENTS_JSON""")
batch_size = $BATCH_SIZE
output = "$OUTPUT"
log_file = "$LOG_FILE"
python_bin = "$PYTHON"

total = len(events)
batches = [events[i:i+batch_size] for i in range(0, total, batch_size)]
n_batches = len(batches)

scraped = 0
failed_events = []

print(f"Processing {total} events in {n_batches} batches of {batch_size}...")
print()

for batch_num, batch in enumerate(batches, 1):
    pct = (batch_num - 1) / n_batches * 100
    print(f"[{batch_num:3d}/{n_batches}] {pct:5.1f}%  events {scraped+1}–{scraped+len(batch)}")

    # Build the --events argument list
    cmd = [
        python_bin,
        "scrapers/bestfightodds_scraper.py",
        "--events", *batch,
        "--output", output,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,   # 5 min max per batch of 10
        )
        # Print warnings/errors from the batch
        for line in result.stderr.splitlines():
            if "WARNING" in line or "ERROR" in line:
                print(f"  ⚠  {line.strip()}")

        if result.returncode not in (0, 1):   # 1 = no data found (ok)
            print(f"  ✗ Batch failed (exit {result.returncode})")
            with open(log_file, 'a') as lf:
                lf.write(f"BATCH FAILED {batch_num}: {batch}\n")
                lf.write(result.stderr[-500:] + "\n")

    except subprocess.TimeoutExpired:
        print(f"  ✗ Batch timed out after 5 minutes — skipping")
        with open(log_file, 'a') as lf:
            lf.write(f"TIMEOUT batch {batch_num}: {batch}\n")

    scraped += len(batch)

print()
print(f"Done. Scraped {scraped} events.")

# Summary of final output
try:
    import pandas as pd
    df = pd.read_csv(output)
    print(f"Total rows in {output}: {len(df)}")
    print(f"Events covered:         {df['event_name'].nunique()}")
    print(f"Date range:             {df['event_date'].iloc[-1]} → {df['event_date'].iloc[0]}")
except Exception as e:
    print(f"Could not read output: {e}")

with open(log_file, 'a') as lf:
    lf.write(f"Scrape finished: {__import__('datetime').datetime.now()}\n")
    lf.write(f"Total events attempted: {scraped}\n")
print(f"\nLog written to: {log_file}")
PYEOF

echo ""
echo "============================================================"
echo " All done. Output: $OUTPUT"
echo "============================================================"
