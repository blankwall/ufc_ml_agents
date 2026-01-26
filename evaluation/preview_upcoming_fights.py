#!/usr/bin/env python3
"""
Preview Upcoming Fights - Model vs Market
----------------------------------------

Lightweight helper that:
  - Reads an upcoming fights CSV/Excel (same schema as export_predictions_to_excel)
  - Uses the saved model + pipeline to add probabilities and edges
  - Prints a simple fight-level summary (top edges) to the terminal
  - Optionally writes a CSV with all model/market fields
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_predictions_to_excel import (  # type: ignore
    load_input,
    add_model_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview upcoming fights with model vs market edges"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/predictions/upcoming_fights.xlsx",
        help="Path to input Excel/CSV with fights and odds",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/predictions/upcoming_fights_with_model.csv",
        help="Path to output CSV with full model/market details",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top-edge fights to show in the terminal",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_model",
        help="Model name to use (default: xgboost_model, e.g., xgboost_model_with_2025)",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help=(
            "Compute probabilities in both fighter orders and use a symmetric probability for EV/edges. "
            "This reduces order sensitivity (adds model_p_*_raw_pct columns in the output CSV)."
        ),
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading upcoming fights from {in_path} ...")
    df_in = load_input(in_path)

    logger.info(f"Adding model probabilities and edges using '{args.model_name}'...")
    df_out = add_model_predictions(df_in, model_name=args.model_name, symmetric=bool(args.symmetric))

    # Save full details to CSV for Excel / further analysis
    df_out.to_csv(out_path, index=False)
    logger.success(f"Wrote full model vs market data to {out_path}")

    if df_out.empty:
        logger.warning("No rows in model output; nothing to preview.")
        return

    # ------------------------------------------------------------------
    # Build a simple fight-level summary using best edge per fight
    # ------------------------------------------------------------------
    summary_rows = []
    for _, row in df_out.iterrows():
        f1 = str(row.get("fighter_1_name", "")).strip()
        f2 = str(row.get("fighter_2_name", "")).strip()
        event = row.get("event", "")
        fight_date = row.get("fight_date", "")

        # Model probabilities (percent)
        p1 = float(row.get("model_p_f1_pct", 0.0) or 0.0)
        p2 = float(row.get("model_p_f2_pct", 0.0) or 0.0)

        # Implied probabilities (percent)
        imp1 = float(row.get("implied_p_f1_pct", 0.0) or 0.0)
        imp2 = float(row.get("implied_p_f2_pct", 0.0) or 0.0)

        # Edges (percent)
        e1 = float(row.get("edge_f1_pct", 0.0) or 0.0)
        e2 = float(row.get("edge_f2_pct", 0.0) or 0.0)

        # Choose best side by edge
        if e1 >= e2:
            best_side = "fighter_1"
            best_name = f1
            best_model_pct = p1
            best_implied_pct = imp1
            best_edge_pct = e1
        else:
            best_side = "fighter_2"
            best_name = f2
            best_model_pct = p2
            best_implied_pct = imp2
            best_edge_pct = e2

        risk_notes = str(row.get("risk_notes", "") or "")

        summary_rows.append(
            {
                "event": event,
                "fight_date": fight_date,
                "fighter_1": f1,
                "fighter_2": f2,
                "best_side": best_side,
                "best_fighter": best_name,
                "best_model_prob_pct": round(best_model_pct, 1),
                "best_market_prob_pct": round(best_implied_pct, 1),
                "best_edge_pct": round(best_edge_pct, 1),
                "risk_notes": risk_notes,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        logger.warning("Summary is empty; nothing to preview.")
        return

    # Sort by edge descending and show top N
    summary_df_sorted = summary_df.sort_values(
        "best_edge_pct", ascending=False
    )

    top_n = min(args.top_n, len(summary_df_sorted))
    print("\n" + "=" * 80)
    print(f"Top {top_n} fights by model edge vs market")
    print("=" * 80)
    print(
        summary_df_sorted.head(top_n)[
            [
                "event",
                "fight_date",
                "fighter_1",
                "fighter_2",
                "best_fighter",
                "best_model_prob_pct",
                "best_market_prob_pct",
                "best_edge_pct",
                "risk_notes",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()


