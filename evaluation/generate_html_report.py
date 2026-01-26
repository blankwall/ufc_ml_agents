#!/usr/bin/env python3
"""
Generate Interactive HTML Report for 2025 Model Evaluation

Creates a collapsible HTML report showing:
- Each UFC 2025 event as a collapsible section
- Predicted winner vs actual winner
- Color coding: Green (correct), Red (incorrect), Yellow (close call)
- Model probabilities, market odds, and edges
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from database.schema import Event


def get_event_name(event_id: int, session) -> str:
    """Get event name from database."""
    try:
        event = session.query(Event).filter_by(id=event_id).first()
        if event:
            return event.name or f"Event {event_id}"
        return f"Event {event_id}"
    except:
        return f"Event {event_id}"


def generate_html_report(
    eval_data_path: Path,
    output_path: Path,
    min_year: int = 2025
) -> None:
    """Generate interactive HTML report from evaluation data."""
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {eval_data_path}...")
    df = pd.read_csv(eval_data_path)
    
    # Filter to specified year
    df = df[df["event_year"] >= min_year].copy()
    logger.info(f"Loaded {len(df)} evaluation rows (includes 2 perspectives per fight) for {min_year}+")
    
    if df.empty:
        logger.error("No data to report")
        return
    
    # Deduplicate fights: Each fight appears twice (once per fighter perspective)
    # Create unique fight key using event_id + sorted fighter IDs
    if "fighter_1_id" in df.columns and "fighter_2_id" in df.columns:
        df["fight_key"] = df.apply(
            lambda r: f"{int(r['event_id'])}_{min(int(r['fighter_1_id']), int(r['fighter_2_id']))}_{max(int(r['fighter_1_id']), int(r['fighter_2_id']))}",
            axis=1
        )
        logger.info(f"Created fight keys, found {df['fight_key'].nunique()} unique fights")
        
        # Keep only the first occurrence of each fight (where winner is fighter_1, target=1)
        # This ensures we always show fights from the winner's perspective
        df = df.sort_values("target", ascending=False)  # Put target=1 rows first
        df = df.drop_duplicates(subset=["fight_key"], keep="first")
        logger.info(f"After deduplication: {len(df)} fights")
    else:
        logger.warning("fighter_1_id/fighter_2_id not found, showing all rows (may have duplicates)")
    
    # Get event names from database
    db = DatabaseManager()
    session = db.get_session()
    
    try:
        # Add event names
        df["event_name"] = df["event_id"].apply(lambda x: get_event_name(int(x), session))
        
        # Calculate fight-level results
        df["model_pick"] = (df["model_prob_f1"] >= 0.5).astype(int)
        df["correct"] = df["model_pick"] == df["target"]
        df["model_confidence"] = np.abs(df["model_prob_f1"] - 0.5) * 2  # 0 to 1

        # --------------------------------------------------------------
        # Underdog stats (market-based)
        # --------------------------------------------------------------
        # We define "underdog" using market implied probabilities from the odds file.
        # Because the eval CSV contains both the odds-file ordering (fighter1/fighter2)
        # and the evaluation row ordering (f1_name/f2_name), we must map correctly.
        underdog_pick_str = "N/A"
        underdog_win_str = "N/A"
        try:
            required_cols = {
                "fighter1_prob",
                "fighter2_prob",
                "fighter1_norm_odds",
                "fighter2_norm_odds",
                "f1_name_norm",
            }
            if required_cols.issubset(df.columns):
                def _market_probs_row(r):
                    f1n = str(r.get("f1_name_norm", "") or "")
                    a = str(r.get("fighter1_norm_odds", "") or "")
                    b = str(r.get("fighter2_norm_odds", "") or "")
                    p1 = float(r.get("fighter1_prob", np.nan))
                    p2 = float(r.get("fighter2_prob", np.nan))
                    if f1n and f1n == a:
                        return p1, p2
                    if f1n and f1n == b:
                        return p2, p1
                    return np.nan, np.nan

                df[["market_prob_f1_calc", "market_prob_f2_calc"]] = df.apply(
                    lambda r: pd.Series(_market_probs_row(r)), axis=1
                )

                df["f1_is_underdog"] = df["market_prob_f1_calc"] < df["market_prob_f2_calc"]
                df["f2_is_underdog"] = df["market_prob_f2_calc"] < df["market_prob_f1_calc"]

                # "Model bet the underdog" = model picked the side with lower market implied probability
                df["model_picked_underdog"] = (
                    ((df["model_pick"] == 1) & df["f1_is_underdog"])
                    | ((df["model_pick"] == 0) & df["f2_is_underdog"])
                )

                underdog_picks_total = int(df["model_picked_underdog"].sum())
                underdog_picks_correct = int((df["model_picked_underdog"] & df["correct"]).sum())
                underdog_pick_rate = (underdog_picks_correct / underdog_picks_total) if underdog_picks_total else 0.0
                underdog_pick_str = f"{underdog_picks_correct}/{underdog_picks_total} ({underdog_pick_rate:.0%})"

                # "Underdog won" in the winner-perspective row (we intentionally keep target=1 rows first)
                # If f1_is_underdog is true here, the winner was the market underdog (an upset).
                underdog_wins_total = int(df["f1_is_underdog"].sum())
                underdog_wins_correct = int((df["f1_is_underdog"] & df["correct"]).sum())
                underdog_win_acc = (underdog_wins_correct / underdog_wins_total) if underdog_wins_total else 0.0
                underdog_win_str = f"{underdog_wins_correct}/{underdog_wins_total} ({underdog_win_acc:.0%})"
        except Exception:
            # Keep as N/A if anything goes wrong (missing columns / data issues)
            pass
        
        # Determine predicted and actual winner names
        df["predicted_winner"] = df.apply(
            lambda r: r["f1_name"] if r["model_prob_f1"] >= 0.5 else r["f2_name"],
            axis=1
        )
        df["actual_winner"] = df.apply(
            lambda r: r["f1_name"] if r["target"] == 1 else r["f2_name"],
            axis=1
        )
        
        # Group by event
        events = df.groupby(["event_id", "event_name", "event_date"]).agg({
            "correct": ["sum", "count"],
            "model_prob_f1": "count"
        }).reset_index()
        
        events.columns = ["event_id", "event_name", "event_date", "correct", "total", "total2"]
        events = events.drop(columns=["total2"])
        events["accuracy"] = events["correct"] / events["total"]
        events = events.sort_values("event_date")
        
        # Overall stats
        total_correct = df["correct"].sum()
        total_fights = len(df)
        overall_accuracy = total_correct / total_fights if total_fights > 0 else 0
        
    finally:
        session.close()
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UFC Model Evaluation Report - {min_year}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f5f5f5;
            border-bottom: 3px solid #d32f2f;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #d32f2f;
            margin: 10px 0;
        }}
        
        .stat-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .events-container {{
            padding: 20px;
        }}
        
        .event-section {{
            margin-bottom: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }}
        
        .event-header {{
            background: linear-gradient(135deg, #424242 0%, #616161 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }}
        
        .event-header:hover {{
            background: linear-gradient(135deg, #616161 0%, #757575 100%);
        }}
        
        .event-header .event-title {{
            font-size: 1.3em;
            font-weight: 600;
        }}
        
        .event-header .event-stats {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}
        
        .event-badge {{
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .event-accuracy {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .event-accuracy.high {{ color: #4caf50; }}
        .event-accuracy.medium {{ color: #ff9800; }}
        .event-accuracy.low {{ color: #f44336; }}
        
        .event-content {{
            display: none;
            padding: 20px;
            background: #fafafa;
        }}
        
        .event-content.active {{
            display: block;
        }}
        
        .fights-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .fights-table thead {{
            background: #424242;
            color: white;
        }}
        
        .fights-table th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
        }}
        
        .fights-table td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .fights-table tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        .fighter-name {{
            font-weight: 600;
            font-size: 1.05em;
        }}
        
        .winner-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 8px;
        }}
        
        .winner-badge.actual {{
            background: #4caf50;
            color: white;
        }}
        
        .winner-badge.predicted {{
            background: #2196f3;
            color: white;
        }}
        
        .row-correct {{
            background: #e8f5e9 !important;
            border-left: 4px solid #4caf50;
        }}
        
        .row-incorrect {{
            background: #ffebee !important;
            border-left: 4px solid #f44336;
        }}
        
        .row-close {{
            background: #e8f5e9 !important;
            border-left: 4px solid #4caf50;
        }}
        
        .prob-bar {{
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}
        
        .prob-fill {{
            height: 100%;
            background: linear-gradient(90deg, #2196f3 0%, #1976d2 100%);
            transition: width 0.3s;
        }}
        
        .prob-text {{
            position: absolute;
            width: 100%;
            text-align: center;
            line-height: 20px;
            font-size: 0.85em;
            font-weight: 600;
            color: #333;
        }}
        
        .edge {{
            font-weight: bold;
        }}
        
        .edge.positive {{ color: #4caf50; }}
        .edge.negative {{ color: #f44336; }}
        .edge.neutral {{ color: #666; }}
        
        .toggle-icon {{
            font-size: 1.5em;
            transition: transform 0.3s;
        }}
        
        .toggle-icon.active {{
            transform: rotate(180deg);
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🥊 UFC Model Evaluation Report</h1>
            <div class="subtitle">2025 Season Performance Analysis</div>
        </div>
        
        <div class="stats-summary">
            <div class="stat-card">
                <div class="label">Overall Accuracy</div>
                <div class="value">{overall_accuracy:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Correct Predictions</div>
                <div class="value">{total_correct}/{total_fights}</div>
            </div>
            <div class="stat-card">
                <div class="label">Events Analyzed</div>
                <div class="value">{len(events)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Holdout Period</div>
                <div class="value">2025+</div>
            </div>
            <div class="stat-card">
                <div class="label">Underdog Picks (Model)</div>
                <div class="value">{underdog_pick_str}</div>
            </div>
            <div class="stat-card">
                <div class="label">Underdog Wins (Upsets)</div>
                <div class="value">{underdog_win_str}</div>
            </div>
        </div>
        
        <div class="events-container">
"""
    
    # Generate event sections
    for _, event_row in events.iterrows():
        event_id = int(event_row["event_id"])
        event_name = event_row["event_name"]
        event_date = pd.to_datetime(event_row["event_date"]).strftime("%B %d, %Y")
        event_accuracy = event_row["accuracy"]
        event_correct = int(event_row["correct"])
        event_total = int(event_row["total"])
        
        # Determine accuracy class for color coding
        if event_accuracy >= 0.8:
            acc_class = "high"
        elif event_accuracy >= 0.6:
            acc_class = "medium"
        else:
            acc_class = "low"
        
        # Get fights for this event
        event_fights = df[df["event_id"] == event_id].copy()
        
        html += f"""
            <div class="event-section">
                <div class="event-header" onclick="toggleEvent('event-{event_id}')">
                    <div class="event-title">{event_name}</div>
                    <div class="event-stats">
                        <span class="event-badge">{event_date}</span>
                        <span class="event-badge">{event_correct}/{event_total} Fights</span>
                        <span class="event-accuracy {acc_class}">{event_accuracy:.0%}</span>
                        <span class="toggle-icon" id="toggle-{event_id}">▼</span>
                    </div>
                </div>
                <div class="event-content" id="event-{event_id}">
                    <table class="fights-table">
                        <thead>
                            <tr>
                                <th>Fight</th>
                                <th>Predicted Winner</th>
                                <th>Actual Winner</th>
                                <th>Model Prob</th>
                                <th>Market Prob</th>
                                <th>Edge</th>
                                <th>Correct?</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        # Add fights
        for _, fight in event_fights.iterrows():
            f1_name = fight["f1_name"]
            f2_name = fight["f2_name"]
            
            model_prob_f1 = fight["model_prob_f1"]
            model_prob_f2 = 1.0 - model_prob_f1
            
            market_prob_f1 = fight.get("market_prob_f1", 0.5)
            market_prob_f2 = 1.0 - market_prob_f1
            
            predicted_winner = fight["predicted_winner"]
            actual_winner = fight["actual_winner"]
            
            is_correct = fight["correct"]
            confidence = fight["model_confidence"]
            
            # Determine row class
            if is_correct:
                if confidence > 0.6:
                    row_class = "row-correct"
                else:
                    row_class = "row-close"  # Correct but low confidence
            else:
                row_class = "row-incorrect"
            
            # Get the PREDICTED winner's probabilities (not the actual winner's)
            if predicted_winner == f1_name:
                predicted_model_prob = model_prob_f1
                predicted_market_prob = market_prob_f1
            else:
                predicted_model_prob = model_prob_f2
                predicted_market_prob = market_prob_f2
            
            # Calculate edge for the PREDICTED winner
            predicted_edge = predicted_model_prob - predicted_market_prob
            edge_pct = predicted_edge * 100
            
            # Edge formatting
            if abs(edge_pct) < 2:
                edge_class = "neutral"
                edge_symbol = ""
            elif edge_pct > 0:
                edge_class = "positive"
                edge_symbol = "+"
            else:
                edge_class = "negative"
                edge_symbol = ""
            
            # Result icon - simple: green check if correct, X if wrong
            if is_correct:
                result_icon = "✓"
                result_color = "#4caf50"  # Green
            else:
                result_icon = "✗"
                result_color = "#f44336"  # Red
            
            html += f"""
                            <tr class="{row_class}">
                                <td>
                                    <div class="fighter-name">{f1_name}</div>
                                    <div style="color: #999; font-size: 0.9em;">vs</div>
                                    <div class="fighter-name">{f2_name}</div>
                                </td>
                                <td>
                                    <strong>{predicted_winner}</strong>
                                    <span class="winner-badge predicted">Predicted</span>
                                </td>
                                <td>
                                    <strong>{actual_winner}</strong>
                                    <span class="winner-badge actual">Actual</span>
                                </td>
                                <td style="text-align: center; font-size: 1.1em; font-weight: 600;">
                                    {predicted_model_prob:.1%}
                                </td>
                                <td style="text-align: center; font-size: 1.1em; font-weight: 600; color: #757575;">
                                    {predicted_market_prob:.1%}
                                </td>
                                <td style="text-align: center;">
                                    <span class="edge {edge_class}">{edge_symbol}{edge_pct:.1f}%</span>
                                </td>
                                <td style="text-align: center;">
                                    <span style="font-size: 1.5em; color: {result_color};">{result_icon}</span>
                                </td>
                            </tr>
"""
        
        html += """
                        </tbody>
                    </table>
                </div>
            </div>
"""
    
    # Close HTML
    html += f"""
        </div>
        
        <div class="footer">
            Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}<br>
            Model: XGBoost • Holdout Period: {min_year}+ • Total Events: {len(events)}
        </div>
    </div>
    
    <script>
        function toggleEvent(eventId) {{
            const content = document.getElementById(eventId);
            const icon = document.getElementById('toggle-' + eventId.replace('event-', ''));
            
            content.classList.toggle('active');
            icon.classList.toggle('active');
        }}
        
        // Expand first event by default
        document.addEventListener('DOMContentLoaded', function() {{
            const firstEvent = document.querySelector('.event-content');
            const firstIcon = document.querySelector('.toggle-icon');
            if (firstEvent) {{
                firstEvent.classList.add('active');
                firstIcon.classList.add('active');
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    
    logger.success(f"Generated HTML report: {output_path}")
    logger.info(f"  Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_fights})")
    logger.info(f"  Events: {len(events)}")
    logger.info(f"  Open in browser: file://{output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report for model evaluation"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Path to evaluation data CSV (from evaluate_model.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/model_evaluation_2025.html",
        help="Output HTML file path"
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2025,
        help="Minimum year to include in report"
    )
    
    args = parser.parse_args()
    
    eval_data_path = Path(args.eval_data)
    output_path = Path(args.output)
    
    generate_html_report(eval_data_path, output_path, args.min_year)


if __name__ == "__main__":
    main()

