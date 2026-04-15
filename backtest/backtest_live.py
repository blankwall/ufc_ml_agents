#!/usr/bin/env python3
"""
Live Backtest — 2026 events (truly out-of-sample)
==================================================
Loads scraped odds  →  data/future_fight_odds/all_events.csv
Loads outcomes      →  data/future_fight_odds/outcomes.csv
Runs in-process predictions (no subprocess, single model load per fight)
Outputs fight-by-fight results + ROI summary

Usage:
    python backtest_live.py                         # all events
    python backtest_live.py --event "UFC 324"       # one event
    python backtest_live.py --model mar_4_v2        # override model
    python backtest_live.py --edge 0.10             # only bet edge>=10%
    python backtest_live.py --no-blend              # skip underdog blend
    python backtest_live.py --quiet                # no per-fight output; summary only at end

Odds come from the CSV only (e.g. data/future_fight_odds/all_events.csv).
"""

import sys
import re
import io
import unicodedata
import argparse
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

ODDS_FILE    = "data/future_fight_odds/all_events.csv"
OUTCOMES_CSV = "data/future_fight_odds/outcomes.csv"
OUT_PRED     = "data/future_fight_odds/predictions.csv"
FLAT_BET     = 100

# BFO scraper name → DB canonical name
# Add entries here whenever a fighter appears with a different spelling on BFO
NAME_ALIASES: dict[str, str] = {
    # Keys are the result of norm() applied to the BFO name (no apostrophes, hyphens, dots)
    "sean omalley":            "Sean O'Malley",
    "waldo cortes acosta":     "Waldo Cortes Acosta",
    "charles johson":          "Charles Johnson",
    "kim sang wook":           "Sangwook Kim",
    "bobby green":             "King Green",
    "michal oleksiejczluk":    "Michal Oleksiejczuk",   # BFO typo (extra l)
}


# ── Name helpers ──────────────────────────────────────────────────────────────

def norm(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    # Hyphens become spaces (e.g. "Cortes-Acosta" → "Cortes Acosta")
    ascii_ = ascii_.replace("-", " ")
    # Strip other punctuation that varies between sources
    ascii_ = re.sub(r"['\.]", "", ascii_)
    return re.sub(r"\s+", " ", ascii_).strip().lower()

def fkey(f1: str, f2: str) -> str:
    a, b = sorted([norm(f1), norm(f2)])
    return f"{a}_vs_{b}"

def american_to_prob(odds: float) -> float:
    o = float(odds)
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)

def prob_to_american(p: float) -> int:
    p = min(max(p, 0.001), 0.999)
    return int(-p / (1 - p) * 100) if p >= 0.5 else int((1 - p) / p * 100)


# ── Fuzzy fight-key matching ──────────────────────────────────────────────────

def _tokens(name: str) -> set:
    """Split normalized name into tokens (words). Used for fuzzy matching."""
    return set(norm(name).split())

def find_outcome(odds_f1: str, odds_f2: str, outcomes: pd.DataFrame) -> str | None:
    # Resolve aliases so typos/nicknames match the canonical DB/UFCStats name
    odds_f1 = _resolve_alias(odds_f1)
    odds_f2 = _resolve_alias(odds_f2)
    """
    Match a fight from the odds file to an outcome row.

    Strategy (in order):
    1. Exact fight_key match
    2. Both last-name tokens found in at least one outcome fight_key
    3. One full name from odds found in the outcome fighter1/fighter2 name
    """
    key = fkey(odds_f1, odds_f2)

    # Normalize stored keys through the same norm() so hyphen/apostrophe differences vanish
    outcomes = outcomes.copy()
    outcomes["_nkey"] = outcomes["fight_key"].apply(lambda k: norm(k.replace("_vs_", "|||")).replace("|||", "_vs_"))

    # 1. Exact
    exact = outcomes[outcomes["_nkey"] == key]
    if not exact.empty:
        w = exact.iloc[0]["winner"]
        return str(w) if pd.notna(w) else None

    # 2. Token overlap — both fighters' last-name tokens appear in one outcome key
    t1 = {norm(odds_f1).split()[-1]}  # last name token for f1
    t2 = {norm(odds_f2).split()[-1]}  # last name token for f2
    for _, row in outcomes.iterrows():
        k = str(row["_nkey"])
        k_tokens = set(k.split("_"))
        if t1 & k_tokens and t2 & k_tokens:
            w = row["winner"]
            return str(w) if pd.notna(w) else None

    # 3. Substring: both names partially found in outcome names
    for _, row in outcomes.iterrows():
        n1 = norm(str(row.get("fighter1", "")))
        n2 = norm(str(row.get("fighter2", "")))
        # Check if odds names are substrings of db names or vice-versa
        f1n = norm(odds_f1)
        f2n = norm(odds_f2)
        match1 = (f1n in n1 or n1 in f1n or
                  f1n.split()[-1] in n1 or n1.split()[-1] in f1n)
        match2 = (f2n in n2 or n2 in f2n or
                  f2n.split()[-1] in n2 or n2.split()[-1] in f2n)
        if match1 and match2:
            w = row["winner"]
            return str(w) if pd.notna(w) else None

    return None


# ── In-process prediction ─────────────────────────────────────────────────────

def _resolve_alias(name: str) -> str:
    """Return the canonical DB name if a known alias exists, else original."""
    return NAME_ALIASES.get(norm(name), name)


def predict_fight(f1: str, f2: str, o1: int | None, o2: int | None,
                  model_name: str, use_blend: bool) -> dict | None:
    """
    Call xgboost_predict() in-process with captured stdout.
    Tries name aliases automatically if initial lookup fails.
    Returns {f1_prob, f2_prob, source} or None on failure.
    """
    from xgboost_predict import xgboost_predict

    f1_db = _resolve_alias(f1)
    f2_db = _resolve_alias(f2)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            xgboost_predict(
                fighter_1_name=f1_db,
                fighter_2_name=f2_db,
                model_name=model_name,
                quiet=True,
                allow_ambiguous=True,
                symmetric=True,
                odds_f1=o1 if use_blend else None,
                odds_f2=o2 if use_blend else None,
            )
    except SystemExit:
        pass
    except Exception as e:
        logger.debug(f"Prediction error {f1_db} vs {f2_db}: {e}")
        return None

    output = buf.getvalue()

    # Parse "Name: XX.X% chance to win"
    probs = re.findall(r":\s+([\d.]+)%\s+chance to win", output)
    if len(probs) >= 2:
        p1, p2 = float(probs[0]) / 100, float(probs[1]) / 100
        source = "blended" if "UNDERDOG SPECIALIST" in output else "general"
        return {"f1_prob": p1, "f2_prob": p2, "source": source}

    return None


# ── P&L calc ──────────────────────────────────────────────────────────────────

def calc_pnl(model_p_f1: float, o1: int, o2: int,
             winner: str, f1: str, f2: str,
             f1_db: str | None = None, f2_db: str | None = None) -> float | None:
    """
    f1/f2      = original BFO names (used for display)
    f1_db/f2_db = resolved DB names (used for winner matching — handles aliases)
    """
    if not winner or winner.lower() in ("", "pending", "none", "nan"):
        return None
    bet_on_f1 = model_p_f1 > 0.5
    w_norm = norm(winner)
    # Check both BFO name and DB name so aliases don't break winner detection
    n1_bfo = norm(f1)
    n2_bfo = norm(f2)
    n1_db  = norm(f1_db) if f1_db else n1_bfo
    n2_db  = norm(f2_db) if f2_db else n2_bfo

    def matches(candidate: str) -> bool:
        return w_norm in candidate or candidate in w_norm

    if bet_on_f1:
        won = matches(n1_bfo) or matches(n1_db)
    else:
        won = matches(n2_bfo) or matches(n2_db)

    odds = o1 if bet_on_f1 else o2
    if odds is None:
        return None
    if won:
        return FLAT_BET * (odds / 100 if odds > 0 else 100 / abs(odds))
    return -FLAT_BET


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",         default=None)
    parser.add_argument("--model",         default="mar_4_v2")
    parser.add_argument("--edge",          type=float, default=0.0)
    parser.add_argument("--no-blend",      action="store_true")
    parser.add_argument("--underdog-only", action="store_true",
                        help="Only count bets where the model picks the market underdog (odds > 0)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="No per-fight output; only aggregated summary at the end")
    args = parser.parse_args()

    if not Path(ODDS_FILE).exists():
        print(f"ERROR: {ODDS_FILE} not found — run scrapers/scrape_event_odds.py first")
        sys.exit(1)

    df_odds = pd.read_csv(ODDS_FILE)

    # Deduplicate: same fight can appear twice if BFO lists it under different fighter orderings
    # Apply alias resolution so "Bobby Green" and "King Green" collapse to the same key
    df_odds["_fkey"] = df_odds.apply(
        lambda r: fkey(_resolve_alias(r["fighter1"]), _resolve_alias(r["fighter2"])), axis=1
    )
    df_odds = df_odds.drop_duplicates(subset=["event_url", "_fkey"], keep="first")
    df_odds = df_odds.drop(columns=["_fkey"])

    if args.event:
        available_events = df_odds["event_name"].dropna().unique().tolist()
        df_odds = df_odds[df_odds["event_name"].str.contains(args.event, case=False, na=False)]
        if df_odds.empty:
            print(f"No fights found for event filter: {args.event!r}")
            print("Available event_name values:", available_events[:15])
            sys.exit(0)

    outcomes = pd.read_csv(OUTCOMES_CSV) if Path(OUTCOMES_CSV).exists() else pd.DataFrame()

    # Fix: event names from BFO may just say "UFC" — use event_url slug for identification
    # Map BFO event URLs to proper names from our scraped outcome event names
    url_to_event = {}
    if not outcomes.empty and "event_url" in outcomes.columns:
        for _, row in outcomes.drop_duplicates("event_url").iterrows():
            url_to_event[row.get("event_url", "")] = row.get("event_name", "")

    if not args.quiet:
        print(f"\nLoaded {len(df_odds)} fights  |  model: {args.model}  |  "
              f"blend: {not args.no_blend}  |  min_edge: {args.edge*100:.0f}%")

    # In quiet mode, only WARNING+ so model load/match messages don't flood output
    if args.quiet:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "WARNING"}])

    use_blend = not args.no_blend
    rows = []

    for event_url, grp in df_odds.groupby("event_url", sort=False):
        # Use the proper event name from outcomes if available
        ev_name = url_to_event.get(event_url, grp["event_name"].iloc[0])
        ev_date = grp["event_date"].iloc[0]

        if not args.quiet:
            print(f"\n{'─'*76}")
            print(f"  {ev_name}  ({ev_date})")
            print(f"{'─'*76}")
            print(f"  {'Fighter 1':22} {'Odds':>7} {'Fighter 2':22} {'Odds':>7} "
                  f"{'Mdl%':>6} {'Mkt%':>6} {'Edge':>6} {'Bet':>4} {'Winner':>22} {'P&L':>6}")
            print(f"  {'─'*22} {'─'*7} {'─'*22} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*22} {'─'*6}")

        for _, row in grp.iterrows():
            f1 = str(row["fighter1"])
            f2 = str(row["fighter2"])
            o1 = int(row["fighter1_odds"]) if pd.notna(row.get("fighter1_odds")) else None
            o2 = int(row["fighter2_odds"]) if pd.notna(row.get("fighter2_odds")) else None

            mkt_p1 = american_to_prob(o1) if o1 else 0.5

            # Prediction
            pred = predict_fight(f1, f2, o1, o2, args.model, use_blend)
            if pred is None:
                if not args.quiet:
                    f1_db = _resolve_alias(f1)
                    f2_db = _resolve_alias(f2)
                    print(f"  {f1[:22]:22} {'?':>7} {f2[:22]:22} {'?':>7}  SKIP  [{f1_db} vs {f2_db}]")
                continue

            p1   = pred["f1_prob"]
            edge = p1 - mkt_p1

            # Outcome lookup
            winner = find_outcome(f1, f2, outcomes) if not outcomes.empty else None
            winner_str = winner or "pending"

            profit = calc_pnl(p1, o1, o2, winner_str, f1, f2,
                              f1_db=_resolve_alias(f1), f2_db=_resolve_alias(f2)) if winner else None
            bet_flag = "BET" if edge >= args.edge and edge > 0 else "    "

            rows.append({
                "event_name":        ev_name,
                "event_date":        ev_date,
                "fighter1":          f1,
                "fighter2":          f2,
                "fighter1_odds":     o1,
                "fighter2_odds":     o2,
                "market_prob_f1":    round(mkt_p1, 4),
                "model_prob_f1":     round(p1, 4),
                "edge_f1":           round(edge, 4),
                "model_source":      pred["source"],
                "model_pick":        f1 if p1 > 0.5 else f2,
                "winner":            winner_str,
                "pnl":               profit,
                "is_underdog":       mkt_p1 < 0.40 or (1 - mkt_p1) < 0.40,
                "bet_flag":          bet_flag.strip(),
            })

            if not args.quiet:
                o1_str = f"{'+' if o1 and o1>0 else ''}{o1}" if o1 else "?"
                o2_str = f"{'+' if o2 and o2>0 else ''}{o2}" if o2 else "?"
                pnl_str = f"{profit:+.0f}" if profit is not None else "    "
                print(
                    f"  {f1[:22]:22} {o1_str:>7} {f2[:22]:22} {o2_str:>7} "
                    f"{p1*100:>6.1f} {mkt_p1*100:>6.1f} {edge*100:>+6.1f} "
                    f"{bet_flag:>4} {winner_str[:22]:>22} {pnl_str:>6}"
                )

    if args.quiet:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])

    # ── Results CSV ───────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PRED, index=False)

    # ── Summary table ─────────────────────────────────────────────────────
    if out_df.empty or "winner" not in out_df.columns:
        settled = pd.DataFrame()
        pending = pd.DataFrame()
    else:
        settled = out_df[out_df["winner"] != "pending"]
        pending = out_df[out_df["winner"] == "pending"]

    # ── Aggregated summary (always printed at the end) ─────────────────────
    print(f"\n{'='*76}")
    print(f"  BACKTEST SUMMARY  —  model: {args.model}")
    print(f"{'='*76}")

    if not settled.empty:
        settled = settled.copy()

        def is_correct(r):
            return (norm(r["model_pick"]) in norm(r["winner"]) or
                    norm(r["winner"]) in norm(r["model_pick"]))

        settled["correct"] = settled.apply(is_correct, axis=1)

        def pick_is_underdog(r):
            bet_f1 = r["model_prob_f1"] > 0.5
            odds   = r["fighter1_odds"] if bet_f1 else r["fighter2_odds"]
            return bool(odds) and float(odds) > 0

        settled["pick_is_dog"] = settled.apply(pick_is_underdog, axis=1)

        acc     = settled["correct"].mean()
        tot_pnl = settled["pnl"].sum()
        staked  = len(settled) * FLAT_BET
        roi_pct = tot_pnl / staked * 100 if staked else 0

        # Aggregate table (key numbers first)
        print(f"\n  AGGREGATE")
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  {'Fights':>8}  {'Settled':>8}  {'Pending':>8}  {'Correct':>10}  {'Accuracy':>10}  {'P&L':>10}  {'ROI':>8}")
        print(f"  {len(out_df):>8}  {len(settled):>8}  {len(pending):>8}  {settled['correct'].sum():>10}  {acc*100:>9.1f}%  ${tot_pnl:>+9.0f}  {roi_pct:>+7.1f}%")
        print(f"  ─────────────────────────────────────────────────────────────")

        # All settled
        print(f"\n  ALL SETTLED ({len(settled)}):  accuracy {acc*100:.1f}%  |  P&L ${tot_pnl:+,.0f}  |  ROI {roi_pct:+.1f}%")

        # Underdog picks
        dog_bets = settled[settled["pick_is_dog"]].copy()
        if not dog_bets.empty:
            dog_pnl = dog_bets["pnl"].sum()
            dog_staked = len(dog_bets) * FLAT_BET
            dog_roi = dog_pnl / dog_staked * 100
            print(f"  UNDERDOG PICKS ({len(dog_bets)}):  accuracy {dog_bets['correct'].mean()*100:.1f}%  |  P&L ${dog_pnl:+,.0f}  |  ROI {dog_roi:+.1f}%")

        # Favourite picks
        fav_bets = settled[~settled["pick_is_dog"]].copy()
        if not fav_bets.empty:
            fav_roi = fav_bets["pnl"].sum() / (len(fav_bets) * FLAT_BET) * 100
            print(f"  FAVOURITE PICKS ({len(fav_bets)}):  accuracy {fav_bets['correct'].mean()*100:.1f}%  |  ROI {fav_roi:+.1f}%")

        # By event
        print(f"\n  BY EVENT")
        print(f"  {'Event':<44} {'W-L':>6}  {'ROI':>8}")
        print(f"  {'-'*44} {'-'*6}  {'-'*8}")
        for ev, ev_grp in settled.groupby("event_name"):
            ev_roi = ev_grp["pnl"].sum() / (len(ev_grp) * FLAT_BET) * 100
            wl = f"{ev_grp['correct'].sum()}-{len(ev_grp) - ev_grp['correct'].sum()}"
            print(f"  {ev[:44]:<44} {wl:>6}  {ev_roi:>+7.1f}%")

        if args.edge > 0:
            bet_s = settled[settled["edge_f1"].abs() >= args.edge]
            if not bet_s.empty:
                edge_roi = bet_s["pnl"].sum() / (len(bet_s) * FLAT_BET) * 100
                print(f"\n  EDGE >= {args.edge*100:.0f}% ({len(bet_s)} bets):  ROI {edge_roi:+.1f}%")
    else:
        print(f"  Fights predicted : {len(out_df)}  |  Settled : {len(settled)}  |  Pending : {len(pending)}")

    print(f"\n  Results saved: {OUT_PRED}")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
