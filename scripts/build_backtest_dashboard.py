#!/usr/bin/env python3
"""
Build an HTML dashboard from a year-specific backtest results CSV.
Focus: edge (model prob vs market implied prob) and EV — the signals that matter for betting.

Usage:
  python scripts/build_backtest_dashboard.py
  python scripts/build_backtest_dashboard.py --csv backtest/backtest_2026_results.csv --out reports/backtest_dashboard.html
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Name normalization and outcome matching (aligned with backtest_live)
NAME_ALIASES = {
    "sean omalley": "Sean O'Malley",
    "waldo cortes acosta": "Waldo Cortes Acosta",
    "charles johson": "Charles Johnson",
    "bobby green": "King Green",
    "michal oleksiejczluk": "Michal Oleksiejczuk",
}


def _norm(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_ = ascii_.replace("-", " ")
    ascii_ = re.sub(r"['\.]", "", ascii_)
    return re.sub(r"\s+", " ", ascii_).strip().lower()


def _fkey(f1: str, f2: str) -> str:
    a, b = sorted([_norm(f1), _norm(f2)])
    return f"{a}_vs_{b}"


def _resolve_alias(name: str) -> str:
    return NAME_ALIASES.get(_norm(name), name)


def _find_outcome(f1: str, f2: str, outcomes: pd.DataFrame) -> str | None:
    """Match a fight to an outcome row; returns winner name or None."""
    f1 = _resolve_alias(f1)
    f2 = _resolve_alias(f2)
    key = _fkey(f1, f2)
    oc = outcomes.copy()
    oc["_nkey"] = oc["fight_key"].apply(lambda k: _norm(str(k).replace("_vs_", "|||")).replace("|||", "_vs_"))
    exact = oc[oc["_nkey"] == key]
    if not exact.empty:
        w = exact.iloc[0]["winner"]
        return str(w) if pd.notna(w) else None
    t1 = {_norm(f1).split()[-1]}
    t2 = {_norm(f2).split()[-1]}
    for _, row in oc.iterrows():
        k_tokens = set(str(row["_nkey"]).split("_"))
        if t1 & k_tokens and t2 & k_tokens:
            w = row["winner"]
            return str(w) if pd.notna(w) else None
    for _, row in oc.iterrows():
        n1, n2 = _norm(str(row.get("fighter1", ""))), _norm(str(row.get("fighter2", "")))
        f1n, f2n = _norm(f1), _norm(f2)
        m1 = f1n in n1 or n1 in f1n or (f1n.split()[-1] in n1 if f1n.split() else False) or (n1.split()[-1] in f1n if n1.split() else False)
        m2 = f2n in n2 or n2 in f2n or (f2n.split()[-1] in n2 if f2n.split() else False) or (n2.split()[-1] in f2n if n2.split() else False)
        if m1 and m2:
            w = row["winner"]
            return str(w) if pd.notna(w) else None
    return None


def american_to_implied(odds: float) -> float:
    if pd.isna(odds):
        return float("nan")
    o = float(odds)
    if o > 0:
        return 100 / (o + 100)
    return abs(o) / (abs(o) + 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(REPO_ROOT / "backtest" / "backtest_2026_results.csv"))
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "reports" / "backtest_dashboard.html"))
    ap.add_argument("--outcomes", type=str, default="", help="Optional: outcomes CSV (event_name,fighter1,fighter2,winner,fight_key) to show W/L")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Drop rows where model failed (error=True or missing probs)
    df = df[df.get("error", False) == False].copy()
    df = df.dropna(subset=["pick_prob", "pick_odds"])

    # Market implied prob for the side we picked
    df["market_implied"] = df["pick_odds"].apply(american_to_implied)
    df["edge"] = (df["pick_prob"] - df["market_implied"]).round(4)
    # Pick EV: if pick is fighter1 use ev1 else ev2
    df["pick_ev"] = df.apply(
        lambda r: r["ev1"] if r["pick"] == r["fighter1"] else r["ev2"],
        axis=1,
    )

    # Optional: join outcomes to show pick W/L
    outcomes_path = args.outcomes or str(REPO_ROOT / "data" / "future_fight_odds" / "outcomes.csv")
    outcomes_df = pd.DataFrame()
    if Path(outcomes_path).exists():
        outcomes_df = pd.read_csv(outcomes_path)
    df["winner"] = None
    df["pick_won"] = None
    if not outcomes_df.empty and "winner" in outcomes_df.columns and "fight_key" in outcomes_df.columns:
        for i, row in df.iterrows():
            w = _find_outcome(row["fighter1"], row["fighter2"], outcomes_df)
            if w and str(w).lower() not in ("", "pending", "none", "nan"):
                df.at[i, "winner"] = w
                pick_norm = _norm(_resolve_alias(row["pick"]))
                winner_norm = _norm(w)
                df.at[i, "pick_won"] = pick_norm == winner_norm

    n = len(df)
    edge_pos = (df["edge"] > 0).sum()
    edge_5 = (df["edge"] >= 0.05).sum()
    edge_10 = (df["edge"] >= 0.10).sum()
    avg_edge = df["edge"].mean()
    avg_ev = df["pick_ev"].mean()
    ev_pos = (df["pick_ev"] > 0).sum()

    # High-edge hit rate (when outcomes available)
    has_outcomes = df["pick_won"].notna().any()
    high_edge_settled = df[(df["edge"] >= 0.10) & (df["pick_won"].notna())] if has_outcomes else pd.DataFrame()
    high_edge_wins = int(high_edge_settled["pick_won"].sum()) if not high_edge_settled.empty else 0
    high_edge_n = len(high_edge_settled)
    high_edge_hit_pct = (100.0 * high_edge_wins / high_edge_n) if high_edge_n else None

    # Edge buckets for chart
    def edge_bucket(e):
        if pd.isna(e):
            return "N/A"
        if e < 0:
            return "< 0%"
        if e < 0.05:
            return "0–5%"
        if e < 0.10:
            return "5–10%"
        if e < 0.15:
            return "10–15%"
        return "≥ 15%"

    df["edge_bucket"] = df["edge"].apply(edge_bucket)
    bucket_order = ["< 0%", "0–5%", "5–10%", "10–15%", "≥ 15%"]
    by_bucket = df.groupby("edge_bucket", sort=False).agg(
        count=("edge", "count"),
        avg_edge=("edge", "mean"),
        avg_ev=("pick_ev", "mean"),
    )
    bucket_counts = [int(by_bucket.loc[b, "count"]) if b in by_bucket.index else 0 for b in bucket_order]
    bucket_evs = [float(by_bucket.loc[b, "avg_ev"]) if b in by_bucket.index else 0.0 for b in bucket_order]

    # Scatter data: market_implied vs pick_prob (above line = edge)
    scatter = df[["market_implied", "pick_prob", "edge", "pick_ev"]].dropna().head(500)
    scatter_list = scatter.to_dict("records")

    # Top edge picks (for table) — include winner/pick_won when available
    top_cols = ["date", "fighter1", "fighter2", "pick", "pick_prob", "market_implied", "edge", "pick_ev"]
    if "winner" in df.columns and "pick_won" in df.columns:
        top_cols.extend(["winner", "pick_won"])
    top_edge = (
        df.nlargest(30, "edge")[top_cols]
        .round({"pick_prob": 3, "market_implied": 3, "edge": 3, "pick_ev": 3})
    )
    top_edge["pick_prob_pct"] = (top_edge["pick_prob"] * 100).round(1)
    top_edge["market_pct"] = (top_edge["market_implied"] * 100).round(1)
    top_edge["edge_pct"] = (top_edge["edge"] * 100).round(1)
    top_edge["result"] = top_edge.apply(
        lambda r: "W" if r.get("pick_won") is True else ("L" if r.get("pick_won") is False else "—"),
        axis=1,
    )
    top_edge_list = top_edge.to_dict("records")

    # Histogram: edge distribution (fixed 5% bins from -20% to 25%)
    edges = df["edge"].dropna() * 100  # in %
    hist_edges = list(range(-20, 26, 5))
    hist_counts = []
    for i in range(len(hist_edges) - 1):
        lo, hi = hist_edges[i], hist_edges[i + 1]
        hist_counts.append(int(((edges >= lo) & (edges < hi)).sum()))
    hist_labels = [f"{hist_edges[i]}–{hist_edges[i+1]}%" for i in range(len(hist_edges) - 1)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Dashboard — Edge &amp; EV</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{ --bg: #0f1419; --card: #1a2332; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem; line-height: 1.5; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .card {{ background: var(--card); border-radius: 8px; padding: 1rem; border: 1px solid #30363d; }}
    .card .value {{ font-size: 1.75rem; font-weight: 700; }}
    .card .label {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.25rem; }}
    .card.positive .value {{ color: var(--green); }}
    .card.negative .value {{ color: var(--red); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
    .panel {{ background: var(--card); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; border: 1px solid #30363d; }}
    .panel h2 {{ font-size: 1rem; margin: 0 0 1rem 0; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #30363d; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .edge-pos {{ color: var(--green); }}
    .edge-neg {{ color: var(--red); }}
    .chart-wrap {{ position: relative; height: 280px; }}
  </style>
</head>
<body>
  <h1>Backtest dashboard</h1>
  <p class="subtitle">Edge = model probability − market implied probability. Positive edge = value.</p>

  <div class="cards">
    <div class="card">
      <div class="value">{n}</div>
      <div class="label">Fights</div>
    </div>
    <div class="card {'positive' if avg_edge > 0 else 'negative' if avg_edge < 0 else ''}">
      <div class="value">{(avg_edge * 100):+.1f}%</div>
      <div class="label">Avg edge (pick)</div>
    </div>
    <div class="card">
      <div class="value">{edge_pos} <span style="font-size:0.9rem;color:var(--muted)">/ {n}</span></div>
      <div class="label">With edge &gt; 0</div>
    </div>
    <div class="card">
      <div class="value">{edge_5}</div>
      <div class="label">With edge ≥ 5%</div>
    </div>
    <div class="card">
      <div class="value">{edge_10}</div>
      <div class="label">With edge ≥ 10%</div>
    </div>
    <div class="card {'positive' if avg_ev > 0 else 'negative' if avg_ev < 0 else ''}">
      <div class="value">{avg_ev:+.3f}</div>
      <div class="label">Avg EV (pick, units)</div>
    </div>
    <div class="card">
      <div class="value">{ev_pos}</div>
      <div class="label">Positive EV picks</div>
    </div>
    {"<div class=\"card\"><div class=\"value\">" + f"{high_edge_wins}/{high_edge_n}" + "</div><div class=\"label\">High-edge (≥10%) hit rate (settled)</div></div>" if has_outcomes and high_edge_n else ""}
    {"<div class=\"card " + ("positive" if high_edge_hit_pct and high_edge_hit_pct >= 55 else "negative" if high_edge_hit_pct else "") + "\"><div class=\"value\">" + (f"{high_edge_hit_pct:.0f}%" if high_edge_hit_pct is not None else "—") + "</div><div class=\"label\">High-edge accuracy %</div></div>" if has_outcomes and high_edge_n else ""}
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Edge distribution (model prob − market implied)</h2>
      <div class="chart-wrap">
        <canvas id="edgeHist"></canvas>
      </div>
    </div>
    <div class="panel">
      <h2>Edge by bucket (count &amp; avg EV)</h2>
      <div class="chart-wrap">
        <canvas id="edgeBucket"></canvas>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>Model prob vs market implied (each pick)</h2>
    <p class="subtitle" style="margin-bottom:0.75rem">Points above the diagonal = positive edge.</p>
    <div class="chart-wrap" style="height:320px">
      <canvas id="scatter"></canvas>
    </div>
  </div>

  <div class="panel">
    <h2>Top 30 by edge (best value spots)</h2>
    <p class="subtitle" style="margin-top:0;margin-bottom:0.75rem">Result = did the model pick win? (W/L/—). High edge does not guarantee a win; check results to see calibration.</p>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Fight</th>
            <th>Pick</th>
            <th class="num">Model %</th>
            <th class="num">Market %</th>
            <th class="num">Edge %</th>
            <th class="num">EV</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {"".join(
            f'<tr><td>{r["date"]}</td><td>{r["fighter1"]} vs {r["fighter2"]}</td><td>{r["pick"]}</td>'
            f'<td class="num">{r["pick_prob_pct"]}</td><td class="num">{r["market_pct"]}</td>'
            f'<td class="num edge-pos">{r["edge_pct"]}</td><td class="num">{r["pick_ev"]:+.3f}</td>'
            f'<td class="{"positive" if r.get("result") == "W" else "negative" if r.get("result") == "L" else ""}">{r.get("result", "—")}</td></tr>'
            for r in top_edge_list
          )}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const histData = {json.dumps(hist_counts)};
    const histLabels = {json.dumps(hist_labels)};
    new Chart(document.getElementById("edgeHist"), {{
      type: "bar",
      data: {{
        labels: histLabels,
        datasets: [{{ label: "Fights", data: histData, backgroundColor: "rgba(88, 166, 255, 0.6)" }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true }} }}
      }}
    }});

    const bucketLabels = {json.dumps(bucket_order)};
    const bucketCounts = {json.dumps(bucket_counts)};
    const bucketEvs = {json.dumps(bucket_evs)};
    new Chart(document.getElementById("edgeBucket"), {{
      type: "bar",
      data: {{
        labels: bucketLabels,
        datasets: [
          {{ label: "Count", data: bucketCounts, yAxisID: "y", backgroundColor: "rgba(88, 166, 255, 0.6)" }},
          {{ label: "Avg EV", data: bucketEvs, yAxisID: "y1", type: "line", borderColor: "#3fb950", tension: 0.2 }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          y: {{ beginAtZero: true, position: "left" }},
          y1: {{ position: "right", grid: {{ drawOnChartArea: false }} }}
        }}
      }}
    }});

    const scatterData = {json.dumps(scatter_list)};
    new Chart(document.getElementById("scatter"), {{
      type: "scatter",
      data: {{
        datasets: [
          {{ label: "Pick", data: scatterData.map(r => ({{ x: r.market_implied, y: r.pick_prob }})), backgroundColor: scatterData.map(r => r.edge >= 0 ? "rgba(63, 185, 80, 0.7)" : "rgba(248, 81, 73, 0.7)"), pointRadius: 5 }},
          {{ label: "Fair line", data: [{{ x: 0, y: 0 }}, {{ x: 1, y: 1 }}], type: "line", borderColor: "rgba(139, 148, 158, 0.6)", borderDash: [5, 5], pointRadius: 0, fill: false }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ min: 0, max: 1, title: {{ display: true, text: "Market implied prob" }} }},
          y: {{ min: 0, max: 1, title: {{ display: true, text: "Model prob" }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out_path}")
    print("Open in a browser to view.")


if __name__ == "__main__":
    main()
