#!/usr/bin/env bash
# =============================================================================
# scrape_recent_odds.sh
#
# Targeted recovery run for 2023-2025 events using the improved fighter-name
# search strategy. Skips events already well-covered (>= MIN_FIGHTS fights).
#
# Usage:
#   bash scrapers/scrape_recent_odds.sh
#   bash scrapers/scrape_recent_odds.sh --min-fights 5   # re-scrape if < 5 fights
#   bash scrapers/scrape_recent_odds.sh --dry-run
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT="data/odds/historical_odds.csv"
MIN_FIGHTS=3   # re-scrape events with fewer than this many fights
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-fights) MIN_FIGHTS="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

echo "============================================================"
echo " Recovery scrape: 2023-2025 UFC events"
echo " Min fights threshold : $MIN_FIGHTS"
echo " Output               : $OUTPUT"
echo " Dry run              : $DRY_RUN"
echo "============================================================"

$PYTHON - <<PYEOF
import sqlite3, json, subprocess, sys, re
import pandas as pd
from datetime import datetime
from pathlib import Path

MIN_FIGHTS  = $MIN_FIGHTS
OUTPUT      = "$OUTPUT"
DRY_RUN     = "$DRY_RUN" == "true"
PYTHON_BIN  = "$PYTHON"

# ── Load DB events for 2023-2025 ────────────────────────────────────────────
conn = sqlite3.connect("data/ufc_database.db")

def parse_date(d):
    for fmt in ['%B %d, %Y', '%B %Y']:
        try: return datetime.strptime(d.strip(), fmt)
        except: pass
    return None

events_df = pd.read_sql("SELECT event_id, name, date FROM events", conn)
events_df['parsed_date'] = events_df['date'].apply(parse_date)
events_df = events_df.dropna(subset=['parsed_date'])
target = events_df[
    (events_df['parsed_date'].dt.year >= 2023) &
    (events_df['parsed_date'].dt.year <= 2025) &
    (~events_df['name'].str.startswith('Road to UFC'))  # BFO doesn't cover these well
].sort_values('parsed_date').reset_index(drop=True)

# ── Check current coverage ───────────────────────────────────────────────────
existing_counts = {}
if Path(OUTPUT).exists():
    scraped = pd.read_csv(OUTPUT)
    # Normalize event name for matching
    def norm(s): return re.sub(r'\s+', ' ', str(s).lower().strip())
    for name, grp in scraped.groupby('event_name'):
        existing_counts[norm(name)] = len(grp)

# ── Identify events needing re-scrape ────────────────────────────────────────
to_scrape = []
for _, row in target.iterrows():
    name = row['name']
    year = row['parsed_date'].year
    n_existing = existing_counts.get(re.sub(r'\s+', ' ', name.lower().strip()), 0)
    if n_existing < MIN_FIGHTS:
        to_scrape.append((name, year, n_existing))

print(f"\nEvents needing scrape (< {MIN_FIGHTS} fights currently): {len(to_scrape)}")
print()

if DRY_RUN:
    for name, year, n in to_scrape:
        print(f"  {year}  [{n:2d} fights]  {name}")
    sys.exit(0)

# ── Run in batches of 10 ─────────────────────────────────────────────────────
BATCH = 10
batches = [to_scrape[i:i+BATCH] for i in range(0, len(to_scrape), BATCH)]
print(f"Running {len(batches)} batches of up to {BATCH} events each...\n")

total_scraped = 0
for batch_num, batch in enumerate(batches, 1):
    names = [b[0] for b in batch]
    years = [str(b[1]) for b in batch]
    pct = (batch_num-1)/len(batches)*100
    print(f"[{batch_num:2d}/{len(batches)}] {pct:5.1f}%  {names[0][:50]}...")

    cmd = [PYTHON_BIN, "scrapers/bestfightodds_scraper.py",
           "--events", *names,
           "--output", OUTPUT]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in result.stderr.splitlines():
            if "WARNING" in line and "Missing odds" not in line:
                print(f"  ⚠  {line.split('|')[-1].strip()}")
            elif "Fighter search match" in line or "Exact match" in line:
                print(f"  ✓  {line.split('|')[-1].strip()}")
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout on batch {batch_num}")

    total_scraped += len(batch)

# ── Final summary ─────────────────────────────────────────────────────────────
print()
if Path(OUTPUT).exists():
    df = pd.read_csv(OUTPUT)
    recent = df[df['event_date'].str.contains('2023|2024|2025', na=False)]
    print("=== Coverage after recovery ===")
    by_year = recent.groupby(recent['event_date'].str[-4:]).agg(
        fights=('fighter1', 'count'),
        events=('event_name', 'nunique')
    )
    print(by_year.to_string())
    print(f"\nTotal rows in CSV: {len(df)}")
PYEOF

echo ""
echo "Done. Check data/odds/historical_odds.csv"
