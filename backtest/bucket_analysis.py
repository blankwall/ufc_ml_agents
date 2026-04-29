import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.confidence_profile import build_confidence_bands


@dataclass
class FightData:
    correct: bool
    female: bool
    edge: float
    pnl: float
    pick_odds: float
    pick_prob: float
    bet: bool
    skip_reason: str


BUCKET_RANGES = {
    -400: (-10000, -350),
    -300: (-350, -250),
    -200: (-250, -150),
    +200: (-150, 250),
    +300: (250, 350),
    +400: (350, 10000),
}

EDGE_TIERS = {
    "0–5%":   (0.00, 0.05),
    "5–10%":  (0.05, 0.10),
    "10–15%": (0.10, 0.15),
    "15%+":   (0.15, 1.00),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convert_prob_to_american_odds(prob):
    if prob >= 1.0:
        return -10000
    if prob <= 0.0:
        return 10000
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def market_implied_prob(american_odds):
    if american_odds >= 100:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def assign_bucket(model_odds):
    for bucket_key, (lo, hi) in BUCKET_RANGES.items():
        if lo <= model_odds < hi:
            return bucket_key
    return None


def assign_edge_tier(edge):
    for label, (lo, hi) in EDGE_TIERS.items():
        if lo <= edge < hi:
            return label
    return None


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation that varies between data sources."""
    return re.sub(r"['\.\-]", "", name).lower().strip()


def _stats(entries):
    """Return (count, wins, losses, win_rate, profit, roi, avg_edge, avg_conf)."""
    if not entries:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    count = len(entries)
    wins = sum(1 for e in entries if e.correct)
    losses = count - wins
    win_rate = wins / count
    profit = sum(e.pnl for e in entries)
    roi = (profit / count) * 100
    avg_edge = sum(e.edge for e in entries) / count
    avg_conf = sum(e.pick_prob for e in entries) / count
    return count, wins, losses, win_rate, profit, roi, avg_edge, avg_conf


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_bets_txt(filepath: str) -> set[tuple[str, str]]:
    """Return a set of (date_str, normalized_fighter_name) from a bets.txt file.

    Expected line format:
        [2026-01-25] Nikita Krylov  @  +112  prob=60.3%  ev=+0.28  WON  (+1.12)  vs Opponent
    """
    bets: set[tuple[str, str]] = set()
    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]\s+(.+?)\s+@\s")
    with open(filepath) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                date = m.group(1)
                name = _normalize(m.group(2))
                bets.add((date, name))
    return bets


def parse_csv(filepath: str, bets: set | None = None) -> list[FightData]:
    """Read backtest results CSV.

    If *bets* is provided (a set of (date, normalized_pick) from parse_bets_txt),
    only rows whose (date, pick) appear in that set are included.
    """
    rows = []

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("pick_prob"):
                continue

            # If a bets filter was supplied, skip rows not in it
            if bets is not None:
                key = (row.get("date", ""), _normalize(row.get("pick", "")))
                if key not in bets:
                    continue

            pick_prob = float(row["pick_prob"])
            pick_odds = float(row["pick_odds"]) if row.get("pick_odds") else 0.0
            mkt_prob = market_implied_prob(pick_odds)
            edge = pick_prob - mkt_prob

            pnl = float(row["actual_pnl"]) if row.get("actual_pnl") else 0.0
            correct = row.get("pick_correct") == "True"
            female = row.get("female") == "True"
            bet = row.get("bet") == "True"
            skip_reason = row.get("skip_reason", "").strip()

            rows.append(
                FightData(
                    correct=correct,
                    female=female,
                    edge=edge,
                    pnl=pnl,
                    pick_odds=pick_odds,
                    pick_prob=pick_prob,
                    bet=bet,
                    skip_reason=skip_reason,
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bucket_prob_range(bucket_key: int) -> str:
    """Return a human-readable probability range for a confidence bucket key."""
    lo_odds, hi_odds = BUCKET_RANGES[bucket_key]
    # Clamp the sentinels so the displayed % stays readable
    lo_prob = market_implied_prob(max(lo_odds, -2000))
    hi_prob = market_implied_prob(min(hi_odds,  2000))
    # Odds range is ordered negative→positive, so lo_odds → higher prob
    p_hi = round(lo_prob * 100)
    p_lo = round(hi_prob * 100)
    return f"{p_lo}%–{p_hi}%"


_BUCKET_COL_WIDTH = 20  # "  -400 (78%–99%)" fits comfortably


def _bucket_header():
    col = _BUCKET_COL_WIDTH
    return (
        f"{'Bucket':<{col}} {'Gender':>6} {'N':>5} {'W':>4} {'L':>4} "
        f"{'WinRate':>8} {'Profit':>8} {'ROI':>7} {'AvgEdge':>8} {'AvgConf':>8}"
    )


def _fmt_bucket_label(key: int | None, prob_range: str) -> str:
    """Format a bucket key with its market odds prob range, e.g. '-200 (60%–71%)'."""
    if key is None:
        return ""
    return f"{key:+d} ({prob_range})"


def _bucket_row(label, gender_label, entries):
    col = _BUCKET_COL_WIDTH
    n, w, l, wr, profit, roi, avg_edge, avg_conf = _stats(entries)
    if n == 0:
        return (
            f"{label:<{col}} {gender_label:>6} {'--':>5} {'--':>4} {'--':>4} "
            f"{'--':>8} {'--':>8} {'--':>7} {'--':>8} {'--':>8}"
        )
    return (
        f"{label:<{col}} {gender_label:>6} {n:>5} {w:>4} {l:>4} "
        f"{wr:>7.1%} {profit:>8.2f} {roi:>6.1f}% {avg_edge:>7.1%} {avg_conf:>7.1%}"
    )


# ---------------------------------------------------------------------------
# Section 1: Confidence bucket breakdown (overall + gender split)
# ---------------------------------------------------------------------------

def analyze_confidence_buckets(rows):
    print("=" * 75)
    print("ODDS BUCKET BREAKDOWN  (bucketed by market pick_odds; AvgConf = model %)")
    print("=" * 75)
    print(_bucket_header())
    print("-" * 75)

    buckets = {k: [] for k in BUCKET_RANGES}
    for r in rows:
        key = assign_bucket(r.pick_odds)
        if key is not None:
            buckets[key].append(r)

    all_entries = []
    best_roi, best_bucket = None, None

    for key in sorted(buckets.keys()):
        entries = buckets[key]
        all_entries.extend(entries)
        prob_range = _bucket_prob_range(key)
        label = _fmt_bucket_label(key, prob_range)

        male = [e for e in entries if not e.female]
        female = [e for e in entries if e.female]

        print(_bucket_row(label, "ALL", entries))
        if male:
            print(_bucket_row("", "M", male))
        if female:
            print(_bucket_row("", "F", female))

        _, _, _, _, _, roi, _, _ = _stats(entries)
        if entries and (best_roi is None or roi > best_roi):
            best_roi = roi
            best_bucket = key

    print("-" * 75)
    print(_bucket_row("TOTAL", "ALL", all_entries))
    print(_bucket_row("", "M", [e for e in all_entries if not e.female]))
    print(_bucket_row("", "F", [e for e in all_entries if e.female]))
    print()
    if best_bucket is not None:
        prob_range = _bucket_prob_range(best_bucket)
        print(f"  Best bucket: {_fmt_bucket_label(best_bucket, prob_range)}  ROI {best_roi:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Section 2: Edge-tier breakdown
# ---------------------------------------------------------------------------

def analyze_edge_tiers(rows):
    print("=" * 75)
    print("EDGE-TIER BREAKDOWN  (model prob − market implied prob)")
    print("=" * 75)
    print(_bucket_header())
    print("-" * 75)

    tiers = {label: [] for label in EDGE_TIERS}
    for r in rows:
        tier = assign_edge_tier(r.edge)
        if tier is not None:
            tiers[tier].append(r)

    all_entries = []
    for label in EDGE_TIERS:
        entries = tiers[label]
        all_entries.extend(entries)
        male = [e for e in entries if not e.female]
        female = [e for e in entries if e.female]
        print(_bucket_row(label, "ALL", entries))
        if male:
            print(_bucket_row("", "M", male))
        if female:
            print(_bucket_row("", "F", female))

    print("-" * 75)
    print(_bucket_row("TOTAL", "ALL", all_entries))
    print()


# ---------------------------------------------------------------------------
# Section 3: Confidence score bands (pick_prob deciles)
# ---------------------------------------------------------------------------

def analyze_confidence_scores(rows):
    print("=" * 80)
    print("CONFIDENCE SCORE BANDS  (score = pick_prob decile for the current dataset)")
    print("=" * 80)
    print(
        f"{'Score':<7} {'ProbRange':<18} {'N':>5} {'WinRate':>8} "
        f"{'AvgPred':>8} {'Gap':>8}"
    )
    print("-" * 80)

    bands = build_confidence_bands([(row.pick_prob, row.correct) for row in rows])
    for band in bands:
        gap = (band.win_rate - band.avg_prob) * 100
        prob_range = f"{band.min_prob*100:>4.1f}%–{band.max_prob*100:>4.1f}%"
        print(
            f"{band.score:<7} {prob_range:<18} {band.sample_size:>5} "
            f"{band.win_rate:>7.1%} {band.avg_prob:>7.1%} {gap:>+7.1f}pp"
        )
    print()


# ---------------------------------------------------------------------------
# Section 4: Skip reason breakdown  (only relevant when NOT filtering by bets)
# ---------------------------------------------------------------------------

def analyze_skip_reasons(rows):
    print("=" * 75)
    print("SKIP REASON BREAKDOWN")
    print("=" * 75)
    header = f"{'Reason':<38} {'N':>5} {'Correct':>8} {'WinRate':>8}"
    print(header)
    print("-" * 63)

    by_reason = defaultdict(list)
    for r in rows:
        if not r.bet and r.skip_reason:
            by_reason[r.skip_reason].append(r)

    # Collapse per-threshold messages into canonical categories
    collapsed: dict[str, list] = defaultdict(list)
    for reason, entries in by_reason.items():
        if reason.startswith("favorite confidence"):
            collapsed["favorite confidence"].extend(entries)
        elif reason.startswith("favorite cap"):
            collapsed["favorite cap"].extend(entries)
        elif reason.startswith("underdog confidence"):
            collapsed["underdog confidence"].extend(entries)
        elif reason.startswith("underdog cap"):
            collapsed["underdog cap"].extend(entries)
        elif reason.startswith("underdog edge"):
            collapsed["underdog edge"].extend(entries)
        else:
            collapsed[reason].extend(entries)

    for reason, entries in sorted(collapsed.items(), key=lambda x: -len(x[1])):
        n = len(entries)
        correct = sum(1 for e in entries if e.correct)
        wr = correct / n if n else 0.0
        print(f"{reason:<38} {n:>5} {correct:>8} {wr:>7.1%}")
    print()


# ---------------------------------------------------------------------------
# Section 5: Weighted ROI using config edge buckets (--config)
# ---------------------------------------------------------------------------

def _get_multiplier(edge: float, female: bool, config: dict) -> float | None:
    """Return the bet multiplier for a given edge, or None if skip."""
    buckets = config.get("edge_buckets", [])
    multiplier = None
    for b in buckets:
        if b["min_edge"] <= edge < b["max_edge"]:
            if b.get("action") == "skip":
                return None
            multiplier = b.get("multiplier")
            break

    if multiplier is None:
        return None

    wmma = config.get("wmma_rules", {})
    if wmma.get("enabled") and female:
        if edge < wmma.get("min_edge", 0):
            return None
        multiplier = min(multiplier, wmma.get("max_multiplier", multiplier))

    return multiplier


def analyze_weighted_roi(rows, config: dict):
    base_unit = config.get("betting", {}).get("base_unit", 100)
    buckets = config.get("edge_buckets", [])

    # Build tier labels from config
    tier_labels = []
    for b in buckets:
        lo = b["min_edge"]
        hi = b["max_edge"]
        if b.get("action") == "skip":
            label = f"{lo*100:.0f}–{hi*100:.0f}% (skip)"
        else:
            label = f"{lo*100:.0f}–{hi*100:.0f}% ({b['multiplier']}x)"
        tier_labels.append((label, lo, hi, b.get("multiplier"), b.get("action")))

    print("=" * 85)
    print(f"WEIGHTED ROI ANALYSIS  (base_unit=${base_unit})")
    print("=" * 85)

    col = 22
    hdr = (
        f"{'Tier':<{col}} {'Mult':>5} {'N':>4} {'W':>3} {'L':>3} "
        f"{'WinRate':>8} {'Staked':>9} {'Profit':>9} {'ROI':>8} {'AvgEdge':>8}"
    )
    print(hdr)
    print("-" * 85)

    total_staked = 0.0
    total_profit = 0.0
    total_n = 0
    total_w = 0

    flat_staked = 0.0
    flat_profit = 0.0

    for label, lo, hi, mult, action in tier_labels:
        tier_rows = [r for r in rows if lo <= r.edge < hi]
        if not tier_rows:
            print(f"{label:<{col}} {'skip' if action == 'skip' else f'{mult}x':>5} {'--':>4} {'--':>3} {'--':>3} {'--':>8} {'--':>9} {'--':>9} {'--':>8} {'--':>8}")
            continue

        n = len(tier_rows)
        w = sum(1 for r in tier_rows if r.correct)
        l = n - w
        wr = w / n
        avg_edge = sum(r.edge for r in tier_rows) / n

        if action == "skip" or mult is None:
            print(f"{label:<{col}} {'skip':>5} {n:>4} {w:>3} {l:>3} {wr:>7.1%} {'—':>9} {'—':>9} {'—':>8} {avg_edge:>7.1%}")
            continue

        staked = n * base_unit * mult
        # pnl is per-$1 unit, so dollar profit = pnl * base_unit * multiplier
        profit = sum(r.pnl * base_unit * mult for r in tier_rows)

        total_staked += staked
        total_profit += profit
        total_n += n
        total_w += w

        flat_staked += n * base_unit
        flat_profit += sum(r.pnl * base_unit for r in tier_rows)

        roi = (profit / staked) * 100 if staked else 0

        print(
            f"{label:<{col}} {mult:>5.1f} {n:>4} {w:>3} {l:>3} "
            f"{wr:>7.1%} ${staked:>8.0f} ${profit:>8.2f} {roi:>7.1f}% {avg_edge:>7.1%}"
        )

    print("-" * 85)

    # Totals
    total_roi = (total_profit / total_staked) * 100 if total_staked else 0
    flat_roi = (flat_profit / flat_staked) * 100 if flat_staked else 0

    print(
        f"{'WEIGHTED TOTAL':<{col}} {'':>5} {total_n:>4} {total_w:>3} "
        f"{total_n - total_w:>3} {total_w/total_n if total_n else 0:>7.1%} "
        f"${total_staked:>8.0f} ${total_profit:>8.2f} {total_roi:>7.1f}%"
    )
    print(
        f"{'FLAT $100 TOTAL':<{col}} {'':>5} {total_n:>4} {total_w:>3} "
        f"{total_n - total_w:>3} {total_w/total_n if total_n else 0:>7.1%} "
        f"${flat_staked:>8.0f} ${flat_profit:>8.2f} {flat_roi:>7.1f}%"
    )

    print()
    if flat_staked > 0:
        lift = total_roi - flat_roi
        print(f"  Weighted ROI: {total_roi:+.1f}%  vs  Flat ROI: {flat_roi:+.1f}%  →  Lift: {lift:+.1f}pp")
        print(f"  Weighted P&L: ${total_profit:+.2f}  vs  Flat P&L: ${flat_profit:+.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze backtest results by model confidence bucket"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to backtest results CSV (e.g. backtest_2026_results.csv)",
    )
    parser.add_argument(
        "--bets",
        type=str,
        default=None,
        help="Path to bets.txt — if supplied, only fights actually bet on are analyzed",
    )
    parser.add_argument(
        "--section",
        type=str,
        choices=["buckets", "edge", "confidence", "skip_reasons", "weighted", "all"],
        default="all",
        help="Which analysis section to show (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to betting_config.json — enables weighted ROI analysis",
    )
    args = parser.parse_args()

    bets_filter = None
    if args.bets:
        bets_filter = parse_bets_txt(args.bets)
        print(f"Loaded {len(bets_filter)} bets from {args.bets}\n")

    rows = parse_csv(args.results, bets=bets_filter)

    if not rows:
        print("No rows matched — check your CSV / bets file paths.")
        return

    if args.section in ("buckets", "all"):
        analyze_confidence_buckets(rows)
    if args.section in ("edge", "all"):
        analyze_edge_tiers(rows)
    if args.section in ("confidence", "all"):
        analyze_confidence_scores(rows)
    if args.section in ("skip_reasons", "all") and bets_filter is None:
        analyze_skip_reasons(rows)

    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text())
    else:
        # Auto-detect config in default location
        default_cfg = Path(__file__).parent.parent / "config" / "betting_config.json"
        if default_cfg.exists():
            config = json.loads(default_cfg.read_text())

    if config and args.section in ("weighted", "all"):
        analyze_weighted_roi(rows, config)


if __name__ == "__main__":
    main()
