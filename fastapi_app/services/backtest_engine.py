"""
UFC Backtest Engine
===================
Primary data source: SQLite DB (data/ufc_database.db) + training_data.csv features,
scored live with mar_4_v2.  Covers ~6,600 fights with closing odds across all years.
mar_4_v2 was trained with --holdout-from-year 2025, so 2025+ is true out-of-sample.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from pydantic import BaseModel

ROOT_DIR   = Path(__file__).parent.parent.parent
DB_PATH    = ROOT_DIR / "data" / "ufc_database.db"
TRAIN_CSV  = ROOT_DIR / "data" / "processed" / "training_data.csv"
MODEL_DIR  = ROOT_DIR / "models" / "saved"

# mar_4_v2 was trained on fights through end of 2024.
# Any fight from this year onward is true out-of-sample.
HOLDOUT_YEAR = 2025

# ── in-process cache ──────────────────────────────────────────────────────────
_db_dataset:  Optional[pd.DataFrame] = None
_gen_model:   Optional[xgb.XGBClassifier] = None
_gen_scaler   = None
_gen_features: Optional[list] = None
_ud_model:    Optional[xgb.XGBClassifier] = None
_ud_scaler    = None
_ud_features: Optional[list] = None


# ── model loaders ─────────────────────────────────────────────────────────────

def get_general_model():
    global _gen_model, _gen_scaler, _gen_features
    if _gen_model is None:
        _gen_model = xgb.XGBClassifier()
        _gen_model.load_model(MODEL_DIR / "mar_4_v2.json")
        _gen_scaler   = joblib.load(MODEL_DIR / "mar_4_v2_feature_scaler.pkl")
        _gen_features = joblib.load(MODEL_DIR / "mar_4_v2_feature_names.pkl")
    return _gen_model, _gen_scaler, _gen_features


def get_underdog_model():
    global _ud_model, _ud_scaler, _ud_features
    if _ud_model is None:
        _ud_model = xgb.XGBClassifier()
        _ud_model.load_model(MODEL_DIR / "underdog_v1.json")
        _ud_scaler   = joblib.load(MODEL_DIR / "underdog_v1_feature_scaler.pkl")
        _ud_features = joblib.load(MODEL_DIR / "underdog_v1_feature_names.pkl")
    return _ud_model, _ud_scaler, _ud_features


# ── DB dataset builder ────────────────────────────────────────────────────────

def build_db_dataset() -> pd.DataFrame:
    """
    Build a fully scored dataset from the DB + training_data.csv.

    Returns one row per fighter-perspective per fight (two rows per fight),
    with columns matching what run_backtest() expects:
      event_date, f1_name, f2_name, weight_class, is_title_fight,
      target, market_prob_f1, model_prob_f1_symmetric, price_f1,
      + all 251 feature columns (for underdog blend scoring).
    """
    con = sqlite3.connect(DB_PATH)

    # ── training features (16562 rows, 251 features + metadata) ──────────────
    td = pd.read_csv(TRAIN_CSV)
    # td.fight_id  = fights.id  (integer PK)
    # td.event_id  = events.id  (integer PK)
    # td.fighter_1_id / fighter_2_id = fighters.id  (integer PK)

    # ── score with general model ──────────────────────────────────────────────
    model, scaler, features = get_general_model()
    X    = td.reindex(columns=features, fill_value=0).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(X), columns=features, index=td.index)
    td["model_prob_f1_symmetric"] = model.predict_proba(X_sc)[:, 1]

    # ── DB: event dates + names ───────────────────────────────────────────────
    events = pd.read_sql("SELECT id AS event_id, name AS event_name, date AS raw_date FROM events", con)
    events["event_date"] = pd.to_datetime(events["raw_date"], format="%B %d, %Y",
                                          errors="coerce")

    # ── DB: fights (canonical fighter order + winner) ─────────────────────────
    fights_db = pd.read_sql("""
        SELECT f.id        AS fight_pk,
               f.event_id,
               f.fighter_1_id AS db_f1_id,
               f.fighter_2_id AS db_f2_id,
               f.winner_id    AS db_winner_id,
               f.weight_class,
               f.is_title_fight,
               fi1.name       AS db_f1_name,
               fi2.name       AS db_f2_name
        FROM fights f
        JOIN fighters fi1 ON f.fighter_1_id = fi1.id
        JOIN fighters fi2 ON f.fighter_2_id = fi2.id
    """, con)

    # ── DB: closing odds per fight ────────────────────────────────────────────
    odds_db = pd.read_sql("""
        SELECT fight_id                         AS fight_pk,
               MAX(fighter_1_implied_prob)      AS f1_raw_impl,
               MAX(fighter_2_implied_prob)      AS f2_raw_impl,
               MAX(fighter_1_odds)              AS f1_amer,
               MAX(fighter_2_odds)              AS f2_amer
        FROM   betting_odds
        WHERE  is_closing_line = 1
        GROUP  BY fight_id
    """, con)
    con.close()

    # Fill missing side via complement and vig-normalise
    odds_db["f1_impl"] = odds_db["f1_raw_impl"].fillna(1 - odds_db["f2_raw_impl"])
    odds_db["f2_impl"] = odds_db["f2_raw_impl"].fillna(1 - odds_db["f1_raw_impl"])
    total = odds_db["f1_impl"] + odds_db["f2_impl"]
    odds_db["market_prob_db_f1"] = odds_db["f1_impl"] / total

    # ── join fights -> events + odds ──────────────────────────────────────────
    fights_db = fights_db.merge(events[["event_id", "event_date", "event_name"]], on="event_id", how="left")
    fights_db = fights_db.merge(odds_db, on="fight_pk", how="inner")  # keep only fights with odds

    # ── join training_data -> fights ─────────────────────────────────────────
    # td.fight_id == fights_db.fight_pk
    td = td.merge(
        fights_db[["fight_pk", "event_date", "event_name", "db_f1_id", "db_f2_id",
                   "db_winner_id", "weight_class", "is_title_fight",
                   "db_f1_name", "db_f2_name",
                   "market_prob_db_f1", "f1_amer", "f2_amer"]],
        left_on="fight_id", right_on="fight_pk", how="inner"
    )

    # ── align market probs with training_data's fighter ordering ──────────────
    # td.fighter_1_id may be the DB canonical f1 or f2
    is_canonical_f1 = td["fighter_1_id"] == td["db_f1_id"]

    td["market_prob_f1"] = np.where(
        is_canonical_f1,
        td["market_prob_db_f1"],
        1.0 - td["market_prob_db_f1"],
    )

    # American odds for td's f1
    td["price_f1"] = np.where(
        is_canonical_f1,
        td["f1_amer"].combine_first(td["market_prob_f1"].apply(prob_to_american)),
        td["f2_amer"].combine_first((1 - td["market_prob_f1"]).apply(
            lambda p: prob_to_american(1.0 - p)
        )),
    )
    # Where actual odds unavailable, derive from market prob
    td["price_f1"] = td["price_f1"].fillna(td["market_prob_f1"].apply(prob_to_american))

    # Fighter display names aligned with td ordering
    td["f1_name"] = np.where(is_canonical_f1, td["db_f1_name"], td["db_f2_name"])
    td["f2_name"] = np.where(is_canonical_f1, td["db_f2_name"], td["db_f1_name"])

    # Resolve duplicate columns from merge (use DB versions as canonical)
    td["weight_class"]  = td["weight_class_y"].fillna(td.get("weight_class_x"))
    td["is_title_fight"] = td["is_title_fight_y"].fillna(td.get("is_title_fight_x"))
    td.drop(columns=["weight_class_x", "weight_class_y",
                     "is_title_fight_x", "is_title_fight_y"], inplace=True, errors="ignore")

    # Sanity-check target against DB winner
    # target=1 means td.fighter_1_id won → fighter_1_id == db_winner_id
    td["target_check"] = (td["fighter_1_id"] == td["db_winner_id"]).astype(int)
    mismatch = (td["target"] != td["target_check"]).sum()
    if mismatch > 0:
        # Use DB winner as ground truth
        td["target"] = td["target_check"]

    # Drop fights with no result (winner unknown)
    td = td[td["db_winner_id"].notna()].copy()

    # Flag in-sample rows (trained on pre-HOLDOUT_YEAR data)
    td["is_in_sample"] = td["event_date"].dt.year < HOLDOUT_YEAR

    return td.reset_index(drop=True)


def get_db_data() -> pd.DataFrame:
    global _db_dataset
    if _db_dataset is None:
        _db_dataset = build_db_dataset()
    # If the cached dataset is missing columns added in later builds, rebuild it
    required_cols = {"event_name", "is_in_sample"}
    if not required_cols.issubset(_db_dataset.columns):
        _db_dataset = build_db_dataset()
    return _db_dataset.copy()


def count_fights_no_odds(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count completed fights in the date range that have no closing-line odds."""
    con = sqlite3.connect(DB_PATH)
    all_events = pd.read_sql(
        "SELECT id AS event_id, date AS raw_date FROM events", con
    )
    all_events["event_date"] = pd.to_datetime(
        all_events["raw_date"], format="%B %d, %Y", errors="coerce"
    )
    ev_ids = all_events.loc[
        (all_events["event_date"] >= start) & (all_events["event_date"] <= end),
        "event_id",
    ].tolist()

    if not ev_ids:
        con.close()
        return 0

    placeholders = ",".join("?" * len(ev_ids))
    fights_in_range = pd.read_sql(
        f"SELECT id FROM fights WHERE event_id IN ({placeholders})"
        f"  AND result IN ('fighter_1','fighter_2')",
        con, params=ev_ids,
    )
    odds_ids = pd.read_sql(
        "SELECT DISTINCT fight_id FROM betting_odds WHERE is_closing_line=1",
        con,
    )["fight_id"].tolist()
    con.close()

    no_odds = fights_in_range[~fights_in_range["id"].isin(odds_ids)]
    return len(no_odds)


# ── pydantic params ───────────────────────────────────────────────────────────

class BacktestParams(BaseModel):
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    focus: str = "all"          # "all" | "favorites" | "underdogs"
    ud_threshold: float = 0.40  # market prob cutoff for underdog label
    # Note: mar_4_v2 trained on pre-2025 data; 2025+ is true out-of-sample.
    min_confidence: float = 0.50
    max_confidence: float = 1.00
    min_edge: float = 0.00      # model prob − market prob
    # American-odds range for the bet being placed (None = no limit)
    min_american_odds: Optional[int] = None   # e.g. -300 → skip anything shorter
    max_american_odds: Optional[int] = None   # e.g. +500 → skip anything longer
    weight_classes: list[str] = []            # empty = all
    use_underdog_blend: bool = True
    blend_weight: float = 0.65
    flat_bet: float = 100.0


# ── helpers ───────────────────────────────────────────────────────────────────

def prob_to_american(p: float) -> int:
    p = min(max(p, 0.001), 0.999)
    return int(-p / (1 - p) * 100) if p >= 0.5 else int((1 - p) / p * 100)


def flat_pnl(odds: float, won: bool, flat_bet: float) -> float:
    if won:
        return flat_bet * (odds / 100 if odds > 0 else 100 / abs(odds))
    return -flat_bet


def build_summary(df: pd.DataFrame, flat_bet: float) -> dict:
    total_staked = len(df) * flat_bet
    total_profit = df["pnl"].sum()
    accuracy = df["correct"].mean()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0.0

    # max drawdown
    cum = df["pnl"].cumsum()
    running_max = cum.cummax()
    max_drawdown = float((running_max - cum).max())

    # per-bet Sharpe (unitless, higher = more consistent)
    mu = df["pnl"].mean()
    sigma = df["pnl"].std()
    sharpe = float(mu / sigma) if sigma > 0 else 0.0

    # Upset detection: meaningful only when f1 is the market underdog in all rows
    actual_upsets = int((df["target"] == 1).sum())   # f1 won = upset (only valid for underdog rows)
    detected_upsets = int(((df["model_prob"] > 0.5) & (df["target"] == 1)).sum())
    # Return None for non-underdog focus so the UI can hide it
    upset_detection = (detected_upsets / actual_upsets) if actual_upsets > 0 else None

    # In-sample detection
    n_in_sample = int(df["is_in_sample"].sum()) if "is_in_sample" in df.columns else 0
    n_out_sample = len(df) - n_in_sample
    in_sample_warning = None
    if n_in_sample > 0:
        pct = round(n_in_sample / len(df) * 100)
        in_sample_warning = (
            f"{n_in_sample} of {len(df)} bets ({pct}%) are IN-SAMPLE "
            f"(mar_4_v2 trained on pre-{HOLDOUT_YEAR} data). "
            f"In-sample ROI is inflated by model memorisation — "
            f"only {HOLDOUT_YEAR}+ results reflect true out-of-sample performance."
        )

    return {
        "n_bets": len(df),
        "n_in_sample": n_in_sample,
        "n_out_sample": n_out_sample,
        "accuracy": round(float(accuracy) * 100, 1),
        "roi": round(float(roi), 1),
        "total_profit": round(float(total_profit), 0),
        "total_staked": round(float(total_staked), 0),
        "max_drawdown": round(max_drawdown, 0),
        "sharpe": round(sharpe, 3),
        "avg_edge": round(float(df["edge"].mean()) * 100, 1),
        "avg_confidence": round(float(df["confidence"].mean()) * 100, 1),
        "upset_detection": round(upset_detection * 100, 1) if upset_detection is not None else None,
        "in_sample_warning": in_sample_warning,
    }


# ── chart builders ────────────────────────────────────────────────────────────

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,14,22,0.6)",
    font=dict(family="Inter, sans-serif", color="#c9d1e0"),
    margin=dict(l=50, r=20, t=45, b=40),
)


def chart_cumulative_pnl(df: pd.DataFrame) -> str:
    df = df.sort_values("event_date")
    cum = df["pnl"].cumsum()
    colors = ["#ff4b6e" if v < 0 else "#00d4aa" for v in cum]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["event_date"],
        y=cum,
        mode="lines",
        name="Cumulative P&L",
        line=dict(color="#4facfe", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(79,172,254,0.08)",
        hovertemplate="%{x|%b %d}<br>$%{y:+,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#444", line_dash="dot")
    fig.update_layout(
        title=dict(text="Cumulative P&L", font=dict(size=14)),
        xaxis_title="",
        yaxis_title="P&L ($)",
        showlegend=False,
        **DARK_LAYOUT,
    )
    return fig.to_json()


def chart_accuracy_by_confidence(df: pd.DataFrame) -> str:
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
    labels = ["50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85+"]
    df = df.copy()
    df["bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    g = df.groupby("bucket", observed=True).agg(
        n=("correct", "count"),
        acc=("correct", "mean"),
        roi=("pnl", lambda x: x.sum() / (len(x) * df["pnl"].abs().mean()) * 100 if len(x) > 0 else 0),
    ).reset_index()

    fig = go.Figure()
    bar_colors = ["#00d4aa" if a >= 0.50 else "#ff4b6e" for a in g["acc"]]
    fig.add_trace(go.Bar(
        x=g["bucket"].astype(str),
        y=(g["acc"] * 100).round(1),
        marker_color=bar_colors,
        text=[f"n={n}" for n in g["n"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x}%<br>Accuracy: %{y:.1f}%<extra></extra>",
        name="Accuracy",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#666",
                  annotation_text="50% baseline", annotation_font_color="#888")
    fig.update_layout(
        title=dict(text="Accuracy by Confidence", font=dict(size=14)),
        xaxis_title="Model Confidence (%)",
        yaxis_title="Accuracy (%)",
        showlegend=False,
        **DARK_LAYOUT,
    )
    return fig.to_json()


def chart_roi_by_weight_class(df: pd.DataFrame, flat_bet: float) -> str:
    g = df.groupby("weight_class").agg(
        n=("pnl", "count"),
        profit=("pnl", "sum"),
    ).reset_index()
    g["roi"] = g["profit"] / (g["n"] * flat_bet) * 100
    g = g.sort_values("roi")

    colors = ["#ff4b6e" if r < 0 else "#00d4aa" for r in g["roi"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=g["weight_class"],
        x=g["roi"].round(1),
        orientation="h",
        marker_color=colors,
        text=[f"n={n}  {r:+.1f}%" for n, r in zip(g["n"], g["roi"].round(1))],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{y}<br>ROI: %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#444", line_dash="dot")
    fig.update_layout(
        title=dict(text="ROI by Weight Class", font=dict(size=14)),
        xaxis_title="ROI (%)",
        yaxis_title="",
        showlegend=False,
        height=350,
        **DARK_LAYOUT,
    )
    return fig.to_json()


def chart_yearly_roi(df: pd.DataFrame, flat_bet: float) -> str:
    df = df.copy()
    df["year"] = df["event_date"].dt.year.astype(str)
    g = df.groupby("year").agg(
        n=("pnl", "count"),
        profit=("pnl", "sum"),
        in_sample=("is_in_sample", "first"),
    ).reset_index()
    g["roi"] = g["profit"] / (g["n"] * flat_bet) * 100

    # Colour: red/orange for in-sample, green/red for out-of-sample
    def bar_color(row):
        if row["in_sample"]:
            return "#f59e0b"  # amber = in-sample (distorted)
        return "#00d4aa" if row["roi"] >= 0 else "#ff4b6e"

    colors = [bar_color(r) for _, r in g.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g["year"],
        y=g["roi"].round(1),
        marker_color=colors,
        text=[f"n={n}<br>{'⚠ in-sample' if ins else 'out-of-sample'}"
              for n, ins in zip(g["n"], g["in_sample"])],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="%{x}<br>ROI: %{y:+.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#444", line_dash="dot")
    fig.update_layout(
        title=dict(
            text="ROI by Year  (amber = in-sample / model trained on these fights)",
            font=dict(size=13),
        ),
        xaxis_title="",
        yaxis_title="ROI (%)",
        showlegend=False,
        **DARK_LAYOUT,
    )
    return fig.to_json()


def chart_monthly_roi(df: pd.DataFrame, flat_bet: float) -> str:
    df = df.copy()
    df["month"] = df["event_date"].dt.to_period("M").astype(str)
    g = df.groupby("month").agg(
        n=("pnl", "count"),
        profit=("pnl", "sum"),
    ).reset_index()
    g["roi"] = g["profit"] / (g["n"] * flat_bet) * 100

    colors = ["#ff4b6e" if r < 0 else "#00d4aa" for r in g["roi"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g["month"],
        y=g["roi"].round(1),
        marker_color=colors,
        text=[f"n={n}" for n in g["n"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x}<br>ROI: %{y:+.1f}%<br>Bets: %{text}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#444", line_dash="dot")
    fig.update_layout(
        title=dict(text="Monthly ROI", font=dict(size=14)),
        xaxis_title="",
        yaxis_title="ROI (%)",
        showlegend=False,
        **DARK_LAYOUT,
    )
    return fig.to_json()


# ── main backtest runner ──────────────────────────────────────────────────────

def run_backtest(params: BacktestParams) -> dict:
    df_full = get_db_data()

    # ── date filter
    start = pd.to_datetime(params.start_date)
    end = pd.to_datetime(params.end_date)
    df_full = df_full[(df_full["event_date"] >= start) & (df_full["event_date"] <= end)]

    if df_full.empty:
        return {"error": "No fights found in the specified date range."}

    # ── weight class filter
    if params.weight_classes:
        df_full = df_full[df_full["weight_class"].isin(params.weight_classes)]
    if df_full.empty:
        return {"error": "No fights found for the selected weight classes."}

    # ── select working subset based on focus
    if params.focus == "all":
        # One canonical row per fight: take the perspective where the model picks f1
        # (sort descending by model_prob, keep first occurrence per fight).
        # This is outcome-independent and avoids the winner-perspective bias
        # of filtering to target==1.
        df_work = (
            df_full
            .sort_values("model_prob_f1_symmetric", ascending=False)
            .drop_duplicates("fight_id", keep="first")
            .copy()
        )
    elif params.focus == "underdogs":
        # All rows where f1 is the market underdog (both outcomes represented)
        df_work = df_full[df_full["market_prob_f1"] < params.ud_threshold].copy()
    elif params.focus == "favorites":
        # Rows where f1 is the market favorite
        df_work = df_full[df_full["market_prob_f1"] >= (1.0 - params.ud_threshold)].copy()
    else:
        return {"error": f"Unknown focus: {params.focus}"}

    if df_work.empty:
        return {"error": f"No fights match focus='{params.focus}' with the current filters."}

    # ── compute model probability (with optional underdog blending)
    df_work["model_prob"] = df_work["model_prob_f1_symmetric"]

    if params.focus == "underdogs" and params.use_underdog_blend:
        ud_model, scaler, ud_features = get_underdog_model()
        X = df_work.reindex(columns=ud_features, fill_value=0).fillna(0)
        X_sc = pd.DataFrame(
            scaler.transform(X), columns=ud_features, index=df_work.index
        )
        ud_probs = ud_model.predict_proba(X_sc)[:, 1]
        df_work["model_prob"] = (
            params.blend_weight * ud_probs
            + (1 - params.blend_weight) * df_work["model_prob_f1_symmetric"]
        )

    # ── derived columns
    df_work["bet_on_f1"] = df_work["model_prob"] > 0.5
    df_work["confidence"] = df_work["model_prob"].apply(lambda p: max(p, 1.0 - p))

    # Edge relative to the fighter being bet on (positive = model has value vs market)
    direction = (2 * df_work["bet_on_f1"].astype(int) - 1)
    df_work["edge"] = direction * (df_work["model_prob"] - df_work["market_prob_f1"])

    # Compute American odds for the bet (using price_f1 if betting f1, derived if f2)
    def _bet_odds(row) -> float:
        if row["bet_on_f1"]:
            return float(row["price_f1"])
        return float(prob_to_american(1.0 - row["market_prob_f1"]))

    df_work["bet_odds"] = df_work.apply(_bet_odds, axis=1)

    # ── apply filters
    mask = (
        (df_work["confidence"] >= params.min_confidence)
        & (df_work["confidence"] <= params.max_confidence)
        & (df_work["edge"] >= params.min_edge)
    )
    if params.min_american_odds is not None:
        # Skip bets shorter than min_american_odds (e.g., -300 → skip -400, -500 …)
        mask &= df_work["bet_odds"] >= params.min_american_odds
    if params.max_american_odds is not None:
        # Skip bets longer than max_american_odds (e.g., +400 → skip +500, +600 …)
        mask &= df_work["bet_odds"] <= params.max_american_odds

    # Capture skipped fights (pass all filters EXCEPT edge) for transparency
    df_skipped = df_work[~mask].copy()

    df_bets = df_work[mask].copy()

    if df_bets.empty:
        return {"error": "No bets pass the specified filters. Try relaxing confidence, edge, or odds constraints."}

    # ── simulate
    df_bets["correct"] = (
        (df_bets["bet_on_f1"] & (df_bets["target"] == 1))
        | (~df_bets["bet_on_f1"] & (df_bets["target"] == 0))
    )
    df_bets["pnl"] = df_bets.apply(
        lambda r: flat_pnl(r["bet_odds"], bool(r["correct"]), params.flat_bet), axis=1
    )
    df_bets = df_bets.sort_values("event_date").reset_index(drop=True)
    df_bets["cumulative_pnl"] = df_bets["pnl"].cumsum()

    # ── build results
    summary = build_summary(df_bets, params.flat_bet)

    charts = {
        "cumulative_pnl": chart_cumulative_pnl(df_bets),
        "accuracy_by_confidence": chart_accuracy_by_confidence(df_bets),
        "roi_by_weight_class": chart_roi_by_weight_class(df_bets, params.flat_bet),
        "monthly_roi": chart_monthly_roi(df_bets, params.flat_bet),
        "yearly_roi": chart_yearly_roi(df_bets, params.flat_bet),
    }

    # ── fight-level table helpers
    def _bet_fighter(r):
        return r["f1_name"] if r["bet_on_f1"] else r["f2_name"]

    def _against_fighter(r):
        return r["f2_name"] if r["bet_on_f1"] else r["f1_name"]

    def _build_fight_rows(df: pd.DataFrame, include_result: bool = True) -> list:
        base = df.assign(
            date=df["event_date"].dt.strftime("%Y-%m-%d"),
            bet_on=df.apply(_bet_fighter, axis=1),
            against=df.apply(_against_fighter, axis=1),
            market_pct=(df["market_prob_f1"] * 100).round(1),
            model_pct=(df["model_prob"] * 100).round(1),
            edge_pct=(df["edge"] * 100).round(1),
            sample=df["is_in_sample"].map({True: "IN", False: "OUT"}),
            event=df["event_name"].fillna(""),
        )
        cols = ["date", "event", "bet_on", "against", "weight_class",
                "market_pct", "model_pct", "edge_pct", "bet_odds", "sample"]
        rename = {"weight_class": "class", "market_pct": "mkt%",
                  "model_pct": "mdl%", "edge_pct": "edge%", "bet_odds": "odds"}
        if include_result:
            base = base.assign(
                result=df["correct"].map({True: "WIN", False: "LOSS"}),
                profit=df["pnl"].round(0).astype(int),
                cumulative=df["cumulative_pnl"].round(0).astype(int),
            )
            cols += ["result", "profit", "cumulative"]
        return base[cols].rename(columns=rename).to_dict("records")

    fight_rows   = _build_fight_rows(df_bets, include_result=True)
    skipped_rows = _build_fight_rows(
        df_skipped.assign(
            bet_on_f1=df_skipped["model_prob"] > 0.5,
            correct=False,
            pnl=0.0,
            cumulative_pnl=0.0,
        ),
        include_result=False,
    )

    # ── per-event summary
    def _event_summary(df: pd.DataFrame) -> list:
        rows = []
        for event, grp in df.groupby("event_name", sort=False):
            n    = len(grp)
            wins = int(grp["correct"].sum())
            pnl  = float(grp["pnl"].sum())
            roi  = pnl / (n * params.flat_bet) * 100 if n > 0 else 0
            date = grp["event_date"].iloc[0].strftime("%Y-%m-%d")
            rows.append({
                "event": event or "",
                "date": date,
                "bets": n,
                "wins": wins,
                "accuracy": round(wins / n * 100, 1) if n > 0 else 0,
                "pnl": round(pnl, 0),
                "roi": round(roi, 1),
            })
        return sorted(rows, key=lambda r: r["date"])

    event_rows = _event_summary(df_bets)

    # ── coverage counts
    n_no_odds = count_fights_no_odds(start, end)

    return {
        "summary": summary,
        "charts": charts,
        "fights": fight_rows,
        "skipped": skipped_rows,
        "events": event_rows,
        "coverage": {
            "total_with_odds": int(df_work["fight_id"].nunique()),
            "selected": len(fight_rows),
            "skipped_edge": len(skipped_rows),
            "no_odds": n_no_odds,
        },
    }


def get_meta() -> dict:
    df = get_db_data()
    # Unique fights = unique fight_id values
    unique = df.drop_duplicates("fight_id")
    return {
        "date_min": df["event_date"].min().strftime("%Y-%m-%d"),
        "date_max": df["event_date"].max().strftime("%Y-%m-%d"),
        "weight_classes": sorted(df["weight_class"].dropna().unique().tolist()),
        "total_fights": int(len(unique)),
        "holdout_note": "mar_4_v2 trained on pre-2025 data — 2025+ is true out-of-sample",
    }
