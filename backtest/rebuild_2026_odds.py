#!/usr/bin/env python3
"""
Rebuild backtest/odds/ufc_2026_odds.csv from all available 2026 data sources:
  - data/future_fight_odds/ufc*.csv       (odds)
  - data/future_fight_odds/the_odds_api*.csv (new-event odds)
  - data/future_fight_odds/outcomes.csv   (results)
  - data/user_events/*.json               (odds + outcomes)
  - data/user_events/user_events/*.json   (odds + outcomes)
  - SQLite DB                             (fight results / winner_id)

Usage:
    python backtest/rebuild_2026_odds.py
    python backtest/rebuild_2026_odds.py --out backtest/odds/ufc_2026_odds.csv
    python backtest/rebuild_2026_odds.py --year 2025 --out ufc_2025_custom.csv
"""

import sys
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ODDS_DIR   = PROJECT_ROOT / "data" / "future_fight_odds"
USER_DIRS  = [
    PROJECT_ROOT / "data" / "user_events",
]
DEFAULT_OUT = PROJECT_ROOT / "backtest" / "odds" / "ufc_2026_odds.csv"
DB_PATH     = PROJECT_ROOT / "data" / "ufc_database.db"


# ── name aliases ─────────────────────────────────────────────────────────────
# Maps odds-source name (lowercased) → canonical DB name.
# Used when building fight keys so that odds-source names match DB names.
NAME_ALIASES: dict[str, str] = {
    "sean omalley":         "Sean O'Malley",
    "waldo cortes-acosta":  "Waldo Cortes Acosta",
    "charles johson":       "Charles Johnson",
    "kim sang wook":        "Sangwook Kim",
    "michal oleksiejczluk": "Michal Oleksiejczuk",
    "carlos leal miranda":  "Carlos Leal",
    "loneer kavanagh":      "Lone'er Kavanagh",
    "jose medina":          "Jose Daniel Medina",
    "bobby green":          "King Green",
    "long xiao":            "Xiao Long",
    "montserrat rendon":    "Montse Rendon",
    "soo young yoo":        "SuYoung You",
    "casey oneill":         "Casey O'Neill",
    "azamt bekoev":         "Azamat Bekoev",
    "lupita godinez":       "Loopy Godinez",
    "michael aswell":       "Michael Aswell Jr.",
    "lance gibson jr":      "Lance Gibson Jr.",  # normalize strips periods; this re-adds Jr. canonical
    "raul rosas jr":        "Raul Rosas Jr.",
    "luana carolina":       "Luana Santos",
    "melissa mullins":      "Melissa Croden",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"['\.]", "", s)   # strip apostrophes and periods (handles Jr. / O'Brien etc.)
    s = re.sub(r"-", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def canonical(name: str) -> str:
    """Apply alias table, then normalize for key construction."""
    resolved = NAME_ALIASES.get(normalize(name), name)
    return normalize(resolved)


def fight_key(f1: str, f2: str) -> str:
    return "_vs_".join(sorted([canonical(f1), canonical(f2)]))


def parse_date(date_raw: str, year: int = 2026) -> str:
    """Normalize supported date strings to 'Month D, YYYY'."""
    clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", str(date_raw).strip())
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(clean, format=fmt).strftime("%B %-d, %Y")
        except Exception:
            continue
    try:
        return pd.to_datetime(f"{clean} {year}", format="%B %d %Y").strftime("%B %-d, %Y")
    except Exception:
        return f"{clean} {year}"


# ── collectors ────────────────────────────────────────────────────────────────

def collect_odds(year: int) -> dict[str, dict]:
    """Return {fight_key: {date, fighter1, fighter2, fighter1_odds, fighter2_odds}}."""
    rows: dict[str, dict] = {}

    # future_fight_odds CSVs
    if ODDS_DIR.exists():
        for pattern in ("ufc*.csv", "the_odds_api*.csv"):
            for csv in sorted(ODDS_DIR.glob(pattern)):
                try:
                    df = pd.read_csv(csv)
                    for _, r in df.iterrows():
                        k = fight_key(r["fighter1"], r["fighter2"])
                        if k not in rows:
                            rows[k] = {
                                "date":          parse_date(r.get("event_date", ""), year),
                                "fighter1":      r["fighter1"],
                                "fighter2":      r["fighter2"],
                                "fighter1_odds": r["fighter1_odds"],
                                "fighter2_odds": r["fighter2_odds"],
                            }
                except Exception as e:
                    print(f"  WARN {csv.name}: {e}")

    # user_events JSONs — sort by event ID descending so newer scrapes (higher IDs) win
    def _event_id(p: Path) -> int:
        m = re.search(r'_(\d+)\.json$', p.name)
        return int(m.group(1)) if m else 0

    for udir in USER_DIRS:
        if not udir.exists():
            continue
        for jf in sorted(udir.glob("*.json"), key=_event_id, reverse=True):
            try:
                d = json.loads(jf.read_text())
                event_date = parse_date(d.get("event_date", ""), year)
                for f in d.get("fights", []):
                    k = fight_key(f["fighter1"], f["fighter2"])
                    if k not in rows:
                        rows[k] = {
                            "date":          event_date,
                            "fighter1":      f["fighter1"],
                            "fighter2":      f["fighter2"],
                            "fighter1_odds": f["fighter1_odds"],
                            "fighter2_odds": f["fighter2_odds"],
                        }
            except Exception as e:
                print(f"  WARN {jf.name}: {e}")

    return rows


def collect_outcomes() -> dict[str, str]:
    """Return {fight_key: winner_name} from all outcome sources."""
    outcomes: dict[str, str] = {}

    # outcomes.csv
    out_csv = ODDS_DIR / "outcomes.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        for _, r in df.iterrows():
            k = fight_key(r["fighter1"], r["fighter2"])
            outcomes[k] = r["winner"]

    # user_events JSON outcomes
    for udir in USER_DIRS:
        if not udir.exists():
            continue
        for jf in sorted(udir.glob("*.json")):
            try:
                d = json.loads(jf.read_text())
                for o in d.get("outcomes", []):
                    k = fight_key(o["fighter1"], o["fighter2"])
                    outcomes[k] = o["winner"]
            except Exception:
                pass

    # DB winner_id
    if DB_PATH.exists():
        engine = create_engine(f"sqlite:///{DB_PATH}")
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT f1.name, f2.name, w.name
                    FROM fights f
                    JOIN fighters f1 ON f.fighter_1_id = f1.id
                    JOIN fighters f2 ON f.fighter_2_id = f2.id
                    LEFT JOIN fighters w ON f.winner_id = w.id
                """)).fetchall()
                for f1n, f2n, wn in rows:
                    if wn:
                        k = fight_key(f1n, f2n)
                        if k not in outcomes:
                            outcomes[k] = wn
        except Exception as e:
            print(f"  WARN DB lookup: {e}")

    return outcomes


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rebuild 2026 odds CSV from all sources")
    parser.add_argument("--out",  default=str(DEFAULT_OUT), help="Output CSV path")
    parser.add_argument("--year", type=int, default=2026,   help="Year to tag undated events with")
    args = parser.parse_args()

    print(f"Collecting odds...")
    odds = collect_odds(args.year)
    print(f"  {len(odds)} unique fights found")

    print("Collecting outcomes...")
    outcomes = collect_outcomes()
    print(f"  {len(outcomes)} outcomes found")

    # Join
    final = []
    for k, r in odds.items():
        final.append({**r, "winner": outcomes.get(k)})

    df = pd.DataFrame(final)
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("_dt").drop(columns="_dt").reset_index(drop=True)

    # Summary
    print(f"\n{'Date':<22} {'Fights':>6}  {'Results':>8}")
    print("-" * 42)
    for date, grp in df.groupby("date", sort=False):
        with_res = grp["winner"].notna().sum()
        flag = "✓" if with_res == len(grp) else ("~" if with_res > 0 else "✗")
        print(f"  {flag} {date:<20} {len(grp):>5}   {with_res}/{len(grp)}")
    print("-" * 42)
    print(f"  {'TOTAL':<22} {len(df):>5}   {df['winner'].notna().sum()}/{len(df)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
