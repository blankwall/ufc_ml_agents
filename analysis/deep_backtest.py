#!/usr/bin/env python3
"""
Deep Backtest Analysis
======================
Loads the evaluation CSV and slices it every useful way:
  - Confidence buckets → accuracy + ROI
  - Edge thresholds → ROI degradation curves
  - Underdog vs favourite tiers
  - Weight-class breakdown
  - Kelly fraction sizing simulation
  - Biggest wins / biggest misses
  - Months-over-time trend
"""

import sys
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports_mar_4_v2")

# ── locate the most-recent eval_data CSV ──────────────────────────────────────
csvs = sorted(REPORTS_DIR.glob("eval_data_*.csv"))
if not csvs:
    sys.exit("No eval_data CSV found in reports_mar_4_v2/")
csv_path = csvs[-1]
print(f"Using: {csv_path}\n")
df_raw = pd.read_csv(csv_path)

# ── build one-row-per-fight view ──────────────────────────────────────────────
# eval_data has 2 rows per fight (both perspectives), pick the winner perspective
df = df_raw.copy()
df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

# Each fight has 2 rows (both perspectives). Use model_prob_f1_symmetric (or model_prob_f1)
# Keep the row where model says fighter_1 wins (prob >= 0.5) → this is the "model's pick" row
prob_col = "model_prob_f1_symmetric" if "model_prob_f1_symmetric" in df.columns else "model_prob_f1"
df_fights = (
    df.sort_values(prob_col, ascending=False)
      .drop_duplicates("fight_key", keep="first")
      .copy()
)
print(f"Unique fights: {len(df_fights)}")

# Derived fields
# confidence = model's probability for its picked fighter (always >= 0.5 after dedup)
df_fights["confidence"]   = df_fights[prob_col].clip(0.5, 1.0)
df_fights["market_prob"]  = df_fights["market_prob_f1"].clip(0.01, 0.99)
df_fights["edge"]         = df_fights[prob_col] - df_fights["market_prob"]
# correct = model picked fighter_1 (prob>=0.5) AND fighter_1 actually won (target==1)
df_fights["correct"]      = df_fights["target"] == 1
df_fights["decimal_odds"] = 1.0 / df_fights["market_prob"]

# Profit per 1-unit flat bet on model's pick
df_fights["profit_flat"] = np.where(
    df_fights["correct"],
    df_fights["decimal_odds"] - 1,
    -1.0
)

sep = "=" * 72

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERALL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("1. OVERALL SUMMARY  (model picks fighter with prob >= 0.5)")
print(sep)
n = len(df_fights)
n_correct = df_fights["correct"].sum()
roi_flat = df_fights["profit_flat"].sum() / n * 100
print(f"  Fights evaluated : {n}")
print(f"  Correct picks    : {n_correct} / {n}  ({n_correct/n*100:.1f}%)")
print(f"  Flat-bet ROI     : {roi_flat:+.1f}%  (bet 1 unit on every model pick)")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIDENCE BUCKETS  (model prob in [0.5, 1.0])
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("2. CONFIDENCE BUCKETS  — accuracy & ROI by model confidence band")
print(sep)
bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85%+"]
df_fights["conf_bucket"] = pd.cut(df_fights["confidence"], bins=bins, labels=labels, right=False)

bucket_stats = (
    df_fights.groupby("conf_bucket", observed=True)
    .agg(
        n=("correct", "count"),
        wins=("correct", "sum"),
        profit=("profit_flat", "sum"),
    )
    .assign(
        accuracy=lambda x: x["wins"] / x["n"] * 100,
        roi=lambda x: x["profit"] / x["n"] * 100,
    )
)

print(f"  {'Conf Band':<12} {'N':>5} {'Acc':>7} {'ROI':>8}  {'Profit':>8}")
print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*8}  {'-'*8}")
for label, row in bucket_stats.iterrows():
    bar = "█" * int(max(0, row["roi"]) / 5)
    print(f"  {label:<12} {int(row['n']):>5} {row['accuracy']:>6.1f}% {row['roi']:>+7.1f}%  {row['profit']:>+7.2f}u  {bar}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 3. EDGE THRESHOLDS  — only bet when model edge > X%
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("3. EDGE THRESHOLDS  — filter to bets where (model prob - market prob) > threshold")
print(sep)
print(f"  {'Min Edge':>9} {'N':>5} {'Acc':>7} {'ROI':>8}  {'Profit':>8}")
print(f"  {'-'*9} {'-'*5} {'-'*7} {'-'*8}  {'-'*8}")
for thresh in [0, 5, 10, 15, 20, 25, 30]:
    subset = df_fights[df_fights["edge"] * 100 >= thresh]
    if len(subset) == 0:
        continue
    acc = subset["correct"].mean() * 100
    profit = subset["profit_flat"].sum()
    roi = profit / len(subset) * 100
    print(f"  {thresh:>8}%  {len(subset):>5} {acc:>6.1f}% {roi:>+7.1f}%  {profit:>+7.2f}u")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 4. UNDERDOG ANALYSIS  — market price tiers
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("4. UNDERDOG ANALYSIS  — model bets grouped by market price of our pick")
print(sep)
price_bins  = [0.00, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.01]
price_labs  = ["dog>+300", "dog+200-300", "dog+120-200", "even/slight fav",
               "fav -120 to -200", "fav -200 to -300", "fav>-300"]
df_fights["price_tier"] = pd.cut(df_fights["market_prob"], bins=price_bins,
                                  labels=price_labs, right=False)

tier_stats = (
    df_fights.groupby("price_tier", observed=True)
    .agg(n=("correct","count"), wins=("correct","sum"), profit=("profit_flat","sum"))
    .assign(
        acc=lambda x: x["wins"]/x["n"]*100,
        roi=lambda x: x["profit"]/x["n"]*100,
        avg_odds=lambda x: (1/df_fights.groupby("price_tier", observed=True)["market_prob"].mean()).reindex(x.index)
    )
)

print(f"  {'Market Tier':<20} {'N':>4} {'W':>4} {'Acc':>7} {'ROI':>8}  {'Profit':>8}")
print(f"  {'-'*20} {'-'*4} {'-'*4} {'-'*7} {'-'*8}  {'-'*8}")
for tier, row in tier_stats.iterrows():
    print(f"  {str(tier):<20} {int(row['n']):>4} {int(row['wins']):>4} {row['acc']:>6.1f}% {row['roi']:>+7.1f}%  {row['profit']:>+7.2f}u")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 5. UNDERDOG vs FAVOURITE SPLIT (model's pick)
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("5. UNDERDOG vs FAVOURITE SPLIT  — is the model's pick market fav or dog?")
print(sep)
for label, mask in [("Model picks UNDERDOG  (mkt <50%)", df_fights["market_prob"] < 0.50),
                    ("Model picks FAVOURITE (mkt >50%)", df_fights["market_prob"] >= 0.50)]:
    s = df_fights[mask]
    if len(s) == 0:
        continue
    acc  = s["correct"].mean() * 100
    roi  = s["profit_flat"].sum() / len(s) * 100
    prof = s["profit_flat"].sum()
    print(f"  {label}")
    print(f"    N={len(s)}  Acc={acc:.1f}%  ROI={roi:+.1f}%  Profit={prof:+.2f}u")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 6. TOP-N% CONFIDENCE FILTERS
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("6. TOP-N% CONFIDENCE FILTERS  — take only the N most confident bets")
print(sep)
print(f"  {'Top %':>6} {'N':>5} {'Conf thresh':>12} {'Acc':>7} {'ROI':>8}  {'Profit':>8}")
print(f"  {'-'*6} {'-'*5} {'-'*12} {'-'*7} {'-'*8}  {'-'*8}")
for pct in [10, 15, 20, 25, 33, 50]:
    thresh = df_fights["confidence"].quantile(1 - pct/100)
    subset = df_fights[df_fights["confidence"] >= thresh]
    if len(subset) == 0:
        continue
    acc  = subset["correct"].mean() * 100
    roi  = subset["profit_flat"].sum() / len(subset) * 100
    prof = subset["profit_flat"].sum()
    print(f"  {pct:>5}%  {len(subset):>5}  {thresh:>11.1%}  {acc:>6.1f}%  {roi:>+7.1f}%  {prof:>+7.2f}u")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 7. KELLY CRITERION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("7. KELLY CRITERION SIMULATION  — fractional Kelly bet sizing")
print(sep)
# Kelly fraction = (p*(b+1) - 1) / b  where b = decimal_odds - 1, p = model prob
df_fights["kelly_f"] = (
    (df_fights["confidence"] * df_fights["decimal_odds"] - 1) /
    (df_fights["decimal_odds"] - 1).replace(0, np.nan)
).clip(0, 0.25)  # cap at 25% of bankroll

for frac, label in [(1.0, "Full Kelly"), (0.5, "Half Kelly"), (0.25, "Quarter Kelly")]:
    # Only bet when kelly > 0 (i.e. positive edge)
    bankroll = 1000.0
    history  = []
    for _, row in df_fights.sort_values("event_date").iterrows():
        k = row["kelly_f"] * frac
        if k <= 0:
            history.append(bankroll)
            continue
        stake  = bankroll * k
        payout = stake * (row["decimal_odds"] - 1) if row["correct"] else -stake
        bankroll += payout
        history.append(bankroll)

    roi_k = (bankroll - 1000) / 1000 * 100
    print(f"  {label:<15}  Final bankroll: ${bankroll:>8.2f}  ROI: {roi_k:>+7.1f}%")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 8. WEIGHT CLASS BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
if "weight_class" in df_fights.columns:
    print(sep)
    print("8. WEIGHT CLASS BREAKDOWN")
    print(sep)
    wc = (
        df_fights.groupby("weight_class")
        .agg(n=("correct","count"), wins=("correct","sum"), profit=("profit_flat","sum"))
        .assign(acc=lambda x: x["wins"]/x["n"]*100,
                roi=lambda x: x["profit"]/x["n"]*100)
        .sort_values("roi", ascending=False)
    )
    print(f"  {'Weight Class':<22} {'N':>4} {'Acc':>7} {'ROI':>8}  {'Profit':>8}")
    print(f"  {'-'*22} {'-'*4} {'-'*7} {'-'*8}  {'-'*8}")
    for wc_name, row in wc.iterrows():
        if row["n"] < 3:
            continue
        print(f"  {str(wc_name):<22} {int(row['n']):>4} {row['acc']:>6.1f}% {row['roi']:>+7.1f}%  {row['profit']:>+7.2f}u")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 9. MONTHLY TREND
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("9. MONTHLY TREND  — accuracy & ROI per month")
print(sep)
df_fights["month"] = df_fights["event_date"].dt.to_period("M")
monthly = (
    df_fights.dropna(subset=["month"])
    .groupby("month")
    .agg(n=("correct","count"), wins=("correct","sum"), profit=("profit_flat","sum"))
    .assign(acc=lambda x: x["wins"]/x["n"]*100,
            roi=lambda x: x["profit"]/x["n"]*100)
)
print(f"  {'Month':<9} {'N':>4} {'Acc':>7} {'ROI':>8}")
print(f"  {'-'*9} {'-'*4} {'-'*7} {'-'*8}")
for period, row in monthly.iterrows():
    bar = "▓" * int(max(0, row["roi"]) / 10) + ("░" * int(max(0, -row["roi"]) / 10))
    print(f"  {str(period):<9} {int(row['n']):>4} {row['acc']:>6.1f}% {row['roi']:>+7.1f}%  {bar}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 10. BIGGEST WINS & MISSES
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("10. BIGGEST UNDERDOG WINS  — model correctly backed big underdogs")
print(sep)
dog_wins = df_fights[(df_fights["correct"]) & (df_fights["market_prob"] < 0.45)].copy()
dog_wins = dog_wins.sort_values("profit_flat", ascending=False).head(10)
print(f"  {'Fighter':<24} {'Date':<12} {'Model%':>7} {'Market%':>8} {'Profit':>8}")
print(f"  {'-'*24} {'-'*12} {'-'*7} {'-'*8} {'-'*8}")
f1_col = "f1_name" if "f1_name" in df_fights.columns else ("fighter_1_name" if "fighter_1_name" in df_fights.columns else df_fights.columns[0])
for _, row in dog_wins.iterrows():
    name = str(row.get(f1_col, "?"))[:23]
    date = str(row["event_date"])[:10] if pd.notna(row["event_date"]) else "?"
    print(f"  {name:<24} {date:<12} {row['confidence']:>6.1%}  {row['market_prob']:>7.1%}  {row['profit_flat']:>+7.2f}u")
print()

print(sep)
print("11. WORST MISSES  — model was most confident but got it wrong")
print(sep)
misses = df_fights[~df_fights["correct"]].copy()
misses = misses.sort_values("confidence", ascending=False).head(10)
print(f"  {'Fighter':<24} {'Date':<12} {'Model%':>7} {'Market%':>8} {'Loss':>8}")
print(f"  {'-'*24} {'-'*12} {'-'*7} {'-'*8} {'-'*8}")
for _, row in misses.iterrows():
    name = str(row.get(f1_col, "?"))[:23]
    date = str(row["event_date"])[:10] if pd.notna(row["event_date"]) else "?"
    print(f"  {name:<24} {date:<12} {row['confidence']:>6.1%}  {row['market_prob']:>7.1%}  {row['profit_flat']:>+7.2f}u")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 12. CALIBRATION CHECK  — does 60% confidence really win 60%?
# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("12. CALIBRATION CHECK  — predicted probability vs actual win rate")
print(sep)
cal_bins = np.arange(0.50, 1.01, 0.05)
cal_labels = [f"{int(b*100)}-{int((b+0.05)*100)}%" for b in cal_bins[:-1]]
df_fights["cal_bucket"] = pd.cut(df_fights["confidence"], bins=cal_bins, labels=cal_labels, right=False)
cal = (
    df_fights.groupby("cal_bucket", observed=True)
    .agg(n=("correct","count"), wins=("correct","sum"), mean_pred=("confidence","mean"))
    .assign(actual_rate=lambda x: x["wins"]/x["n"]*100,
            pred_rate=lambda x: x["mean_pred"]*100)
)
print(f"  {'Band':<12} {'N':>4} {'Predicted':>10} {'Actual':>8}  {'Gap':>6}")
print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*8}  {'-'*6}")
for band, row in cal.iterrows():
    gap = row["actual_rate"] - row["pred_rate"]
    flag = "✓" if abs(gap) < 5 else ("↑" if gap > 0 else "↓")
    print(f"  {str(band):<12} {int(row['n']):>4} {row['pred_rate']:>9.1f}%  {row['actual_rate']:>7.1f}%  {gap:>+5.1f}%  {flag}")
print()

print(sep)
print("SUMMARY TABLE — Best strategies by ROI")
print(sep)
strategies = [
    ("All picks (flat)",         df_fights,                                                         "flat"),
    ("Edge > 10%",               df_fights[df_fights["edge"] >= 0.10],                             "flat"),
    ("Edge > 20%",               df_fights[df_fights["edge"] >= 0.20],                             "flat"),
    ("Top 25% confidence",       df_fights[df_fights["confidence"] >= df_fights["confidence"].quantile(0.75)], "flat"),
    ("Underdog picks only",      df_fights[df_fights["market_prob"] < 0.50],                       "flat"),
    ("Dog >+200 (mkt <33%)",     df_fights[df_fights["market_prob"] < 0.33],                       "flat"),
    ("Fav picks only",           df_fights[df_fights["market_prob"] >= 0.50],                      "flat"),
]
print(f"  {'Strategy':<30} {'N':>5} {'Acc':>7} {'ROI':>8}  {'Profit':>8}")
print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*8}  {'-'*8}")
for name, sub, _ in strategies:
    if len(sub) == 0:
        continue
    acc  = sub["correct"].mean() * 100
    prof = sub["profit_flat"].sum()
    roi  = prof / len(sub) * 100
    print(f"  {name:<30} {len(sub):>5} {acc:>6.1f}% {roi:>+7.1f}%  {prof:>+7.2f}u")
print()
