#!/usr/bin/env python3
"""
CLV (Closing Line Value) analysis — validate the model against open vs close.

Uses odds from fetch_odds_graphs.py output (opening + closing per fighter).
Measures:
  - |Model - Close| vs |Open - Close|  → model beats open if it's closer to close
  - CLV score = (Close - Open) for the predicted side
  - Model predicted movement = model prob is between open and close

Usage:
  # After fetching graphs for an event:
  python fetch_odds_graphs.py ufc-3971
  python analysis/clv_analysis.py /tmp/ufc-3971_odds.json

  # Multiple events (e.g. season):
  python analysis/clv_analysis.py /tmp/ufc-3971_odds.json /tmp/ufc-3970_odds.json
  python analysis/clv_analysis.py --dir /tmp --pattern "ufc-*_odds.json"

  # Optional: save detailed CSV
  python analysis/clv_analysis.py /tmp/ufc-3971_odds.json --out data/clv_results.csv
"""

import sys
import re
import io
import json
import argparse
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd

# Repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def american_odds_to_prob(american_odds) -> float:
    """Convert American odds (int or string '+144' / '-164') to implied probability."""
    if isinstance(american_odds, str):
        american_odds = american_odds.strip().replace("+", "")
        american_odds = int(american_odds)
    o = float(american_odds)
    if o > 0:
        return 100 / (o + 100)
    return abs(o) / (abs(o) + 100)


def parse_american(s) -> int | None:
    """Parse '+144' or '-164' to int."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().replace("+", "")
    try:
        return int(s)
    except ValueError:
        return None


def run_model_prediction(f1: str, f2: str, model_name: str = "mar_4_v2") -> float | None:
    """Return model P(f1 wins) or None on failure."""
    from xgboost_predict import xgboost_predict

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            xgboost_predict(
                fighter_1_name=f1,
                fighter_2_name=f2,
                model_name=model_name,
                quiet=True,
                allow_ambiguous=True,
                symmetric=True,
            )
    except (SystemExit, Exception):
        return None

    out = buf.getvalue()
    probs = re.findall(r":\s+([\d.]+)%\s+chance to win", out)
    if len(probs) >= 2:
        return float(probs[0]) / 100
    return None


def load_event_odds(path: Path) -> dict:
    """Load one event JSON from fetch_odds_graphs.py. Returns { matchup_id: fight_data }."""
    with open(path) as f:
        data = json.load(f)
    return data


def analyze_fight(mu_id: str, fight_data: dict, model_name: str) -> dict | None:
    """
    For one fight: get open/close probs, model prob, and CLV metrics.
    fight_data has: matchup_id, fighters [f1, f2], graphs { f1: { opening: { american_odds }, current: { american_odds } }, f2: { ... } }
    """
    fighters = fight_data.get("fighters") or []
    graphs = fight_data.get("graphs") or {}
    if len(fighters) < 2:
        return None

    f1, f2 = fighters[0], fighters[1]
    g1 = graphs.get(f1) or {}
    g2 = graphs.get(f2) or {}

    open_1 = parse_american(g1.get("opening", {}).get("american_odds"))
    close_1 = parse_american(g1.get("current", {}).get("american_odds"))
    open_2 = parse_american(g2.get("opening", {}).get("american_odds"))
    close_2 = parse_american(g2.get("current", {}).get("american_odds"))

    if None in (open_1, close_1, open_2, close_2):
        return None

    open_p1 = american_odds_to_prob(open_1)
    close_p1 = american_odds_to_prob(close_1)
    # Optional: vig removal so open_p1 + open_p2 = 1 (use decimal from JSON if available)
    # For now use raw implied (open_p1 + open_p2 may > 1)
    open_p2 = american_odds_to_prob(open_2)
    close_p2 = american_odds_to_prob(close_2)

    model_p1 = run_model_prediction(f1, f2, model_name)
    if model_p1 is None:
        return None

    model_error_close = abs(model_p1 - close_p1)
    open_error_close = abs(open_p1 - close_p1)
    model_beats_open = model_error_close < open_error_close

    # Model predicted movement: model prob lies between open and close for f1
    low, high = min(open_p1, close_p1), max(open_p1, close_p1)
    model_predicted_movement = low <= model_p1 <= high

    # CLV for f1: positive = line moved toward f1
    clv_f1 = close_p1 - open_p1

    return {
        "matchup_id": mu_id,
        "fighter1": f1,
        "fighter2": f2,
        "model_prob_f1": round(model_p1, 4),
        "open_prob_f1": round(open_p1, 4),
        "close_prob_f1": round(close_p1, 4),
        "open_odds_f1": open_1,
        "close_odds_f1": close_1,
        "model_error_vs_close": round(model_error_close, 4),
        "open_error_vs_close": round(open_error_close, 4),
        "model_beats_open": model_beats_open,
        "model_predicted_movement": model_predicted_movement,
        "clv_f1": round(clv_f1, 4),
    }


def run_clv_analysis(
    json_paths: list[Path],
    model_name: str = "mar_4_v2",
    event_name_from_path: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Load one or more event JSONs, run model on each fight, return (CLV dataframe, skipped_count)."""
    rows = []
    skipped = 0
    for path in json_paths:
        if not path.exists():
            continue
        event_name = path.stem.replace("_odds", "") if event_name_from_path else path.name
        data = load_event_odds(path)
        for mu_id, fight_data in data.items():
            row = analyze_fight(mu_id, fight_data, model_name)
            if row:
                row["event"] = event_name
                row["odds_file"] = str(path)
                rows.append(row)
            else:
                skipped += 1
    return pd.DataFrame(rows), skipped


def main():
    ap = argparse.ArgumentParser(description="CLV analysis from fetch_odds_graphs JSON")
    ap.add_argument("json_files", nargs="*", help="Paths to *_odds.json files")
    ap.add_argument("--dir", default=None, help="Directory to scan for JSONs")
    ap.add_argument("--pattern", default="*_odds.json", help="Glob pattern when using --dir")
    ap.add_argument("--model", default="mar_4_v2", help="Model name")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--quiet", action="store_true", help="Less stdout")
    args = ap.parse_args()

    paths = []
    for p in args.json_files:
        paths.append(Path(p))
    if args.dir:
        d = Path(args.dir)
        paths.extend(d.glob(args.pattern))

    if not paths:
        print("No JSON files given. Example: python analysis/clv_analysis.py /tmp/ufc-3971_odds.json")
        sys.exit(1)

    df, skipped = run_clv_analysis(paths, model_name=args.model)

    if df.empty:
        print("No fights with open/close odds and successful model predictions.")
        if skipped:
            print(f"  (Skipped {skipped} fights: missing odds or model lookup failed)")
        sys.exit(0)

    # Summary
    n = len(df)
    beats_open = df["model_beats_open"].sum()
    predicted_movement = df["model_predicted_movement"].sum()
    model_mae = df["model_error_vs_close"].mean()
    open_mae = df["open_error_vs_close"].mean()

    print("\n" + "=" * 70)
    print("CLV (Closing Line Value) Summary")
    print("=" * 70)
    print(f"  Fights analyzed      : {n}" + (f"  (skipped {skipped})" if skipped else ""))
    print(f"  Model closer to close: {beats_open}/{n}  ({100*beats_open/n:.1f}%)")
    print(f"  Model predicted move : {predicted_movement}/{n}  (model between open & close)")
    print()
    print("  Mean Absolute Error (vs closing line):")
    print(f"    Model MAE  = mean(|model_prob - close_prob|)  = {model_mae:.4f}")
    print(f"    Open  MAE  = mean(|open_prob  - close_prob|)  = {open_mae:.4f}")
    if model_mae < open_mae:
        print("    → Model MAE < Open MAE: model predicts the closing line better than the open.")
        print("      Strong evidence of edge.")
    else:
        print("    → Open MAE ≤ Model MAE: open line is closer to close on average.")
    print("=" * 70)

    if not args.quiet:
        print("\nPer-fight (first 20):")
        cols = ["event", "fighter1", "fighter2", "model_prob_f1", "open_prob_f1", "close_prob_f1",
                "model_beats_open", "model_predicted_movement", "clv_f1"]
        print(df[cols].head(20).to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return df


if __name__ == "__main__":
    main()
