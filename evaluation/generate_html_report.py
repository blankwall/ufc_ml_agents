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

        # BEFORE deduplication: Calculate underdog stats on ALL rows
        # Count underdog stats per unique fight (one per fight, not both perspectives)
        if "market_prob_f1" in df.columns:
            # Create fight_key if needed
            df_temp = df.copy()
            if "fight_key" not in df_temp.columns:
                if "fighter_1_id" in df_temp.columns and "fighter_2_id" in df_temp.columns:
                    df_temp["fight_key"] = df_temp.apply(
                        lambda r: f"{int(r['fighter_1_id'])}_{min(int(r['fighter_1_id']), int(r['fighter_2_id']))}_{max(int(r['fighter_1_id']), int(r['fighter_2_id']))}",
                        axis=1
                    )

            # Calculate model pick and correct
            df_temp["model_pick_pre"] = (df_temp["model_prob_f1"] >= 0.5).astype(int)
            df_temp["correct_pre"] = df_temp["model_pick_pre"] == df_temp["target"]

            # Determine if the model picked the market underdog in each row
            df_temp["model_picked_underdog"] = np.where(
                df_temp["model_prob_f1"] >= 0.5,
                df_temp["market_prob_f1"] < df_temp.get("market_prob_f2", 0.5),
                df_temp.get("market_prob_f2", 0.5) < df_temp["market_prob_f1"]
            )

            # Determine if underdog won the fight
            df_temp["underdog_won"] = np.where(
                df_temp["market_prob_f1"] < df_temp.get("market_prob_f2", 0.5),
                df_temp["target"] == 1,
                df_temp["target"] == 0
            )

            # Count per unique fight: model bet on underdog AND was correct
            seen_fights_bets = set()
            underdog_picks_correct_pre = 0
            underdog_picks_total_pre = 0

            for idx, row in df_temp.iterrows():
                fight_key = row.get("fight_key", "")
                if fight_key and fight_key not in seen_fights_bets:
                    if row["model_picked_underdog"]:
                        underdog_picks_total_pre += 1
                        if row["correct_pre"]:
                            underdog_picks_correct_pre += 1
                        seen_fights_bets.add(fight_key)

            # Count per unique fight: underdog won AND model bet on underdog AND was correct
            seen_fights_won = set()
            underdog_wins_correct_pre = 0
            underdog_wins_total_pre = 0

            for idx, row in df_temp.iterrows():
                fight_key = row.get("fight_key", "")
                if fight_key and fight_key not in seen_fights_won:
                    if row["underdog_won"]:
                        underdog_wins_total_pre += 1
                        if row["model_picked_underdog"] and row["correct_pre"]:
                            underdog_wins_correct_pre += 1
                        seen_fights_won.add(fight_key)
        else:
            df["model_picked_underdog"] = False
            underdog_picks_total_pre = 0
            underdog_picks_correct_pre = 0
            underdog_wins_total_pre = 0
            underdog_wins_correct_pre = 0

        # Keep only the first occurrence of each fight (where winner is fighter_1, target=1)
        # This ensures we always show fights from the winner's perspective
        df = df.sort_values("target", ascending=False)  # Put target=1 rows first
        df = df.drop_duplicates(subset=["fight_key"], keep="first")
        logger.info(f"After deduplication: {len(df)} fights")
    else:
        logger.warning("fighter_1_id/fighter_2_id not found, showing all rows (may have duplicates)")
        underdog_picks_total_pre = 0
        underdog_picks_correct_pre = 0

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

        # For row highlighting: check if THIS ROW (in winner perspective) was an underdog pick
        # In winner-perspective rows (f1 is winner), was the model's pick an underdog?
        if "market_prob_f1" in df.columns:
            df["is_underdog_pick_display"] = np.where(
                df["model_prob_f1"] >= 0.5,  # Model picked f1 (winner)
                df["market_prob_f1"] < df.get("market_prob_f2", 0.5),  # f1 was underdog
                df.get("market_prob_f2", 0.5) < df["market_prob_f1"]  # f2 was underdog (but model picked loser)
            )
        else:
            df["is_underdog_pick_display"] = False

        # Also calculate a winner-perspective version for display purposes
        if "market_prob_f1" in df.columns:
            # In winner-perspective rows (f1 is winner), was f1 the underdog?
            df["winner_was_underdog"] = df["market_prob_f1"] < df.get("market_prob_f2", 0.5)
        else:
            df["winner_was_underdog"] = False

        # --------------------------------------------------------------
        # Underdog stats (market-based)
        # --------------------------------------------------------------
        # Show all underdog picks (not filtered by edge)
        underdog_pick_str = "N/A"
        underdog_win_str = "N/A"
        try:
            if underdog_picks_total_pre > 0:
                # Model picked underdog: model picked the fighter with lower market probability
                underdog_pick_rate = (underdog_picks_correct_pre / underdog_picks_total_pre) if underdog_picks_total_pre > 0 else 0.0
                underdog_pick_str = f"{underdog_picks_correct_pre}/{underdog_picks_total_pre} ({underdog_pick_rate:.0%})"

            if underdog_wins_total_pre > 0:
                # Underdog won (upset): underdog won the fight AND model correctly predicted
                underdog_win_rate = (underdog_wins_correct_pre / underdog_wins_total_pre) if underdog_wins_total_pre > 0 else 0.0
                underdog_win_str = f"{underdog_wins_correct_pre}/{underdog_wins_total_pre} ({underdog_win_rate:.0%})"
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
            background: #1b5e20 !important;
            color: white !important;
            border-left: 6px solid #2e7d32;
        }}

        .row-incorrect {{
            background: #b71c1c !important;
            color: white !important;
            border-left: 6px solid #c62828;
        }}

        .row-close {{
            background: #1b5e20 !important;
            color: white !important;
            border-left: 6px solid #2e7d32;
        }}

        /* Underdog picks - pastel colors */
        .row-correct.underdog {{
            background: #b2dfdb !important;  /* Blueish green (pastel) */
            color: #004d40 !important;
            border-left: 6px solid #009688;
        }}

        .row-incorrect.underdog {{
            background: #ffccbc !important;  /* Orangish red (pastel) */
            color: #bf360c !important;
            border-left: 6px solid #ff7043;
        }}

        /* Top 25% indicator - gold star badge */
        .top-25-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #ffd700 0%, #ffb300 100%);
            color: #5d4037;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}

        /* Override text colors for dark backgrounds */
        .row-correct a, .row-incorrect a, .row-close a {{
            color: inherit;
        }}

        /* Ensure badges work on both dark and light backgrounds */
        .row-correct .winner-badge.actual,
        .row-incorrect .winner-badge.actual,
        .row-close .winner-badge.actual {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }}

        .row-correct .winner-badge.predicted,
        .row-incorrect .winner-badge.predicted,
        .row-close .winner-badge.predicted {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }}

        .row-correct.underdog .winner-badge.actual,
        .row-incorrect.underdog .winner-badge.actual {{
            background: #4caf50;
            color: white;
        }}

        .row-correct.underdog .winner-badge.predicted,
        .row-incorrect.underdog .winner-badge.predicted {{
            background: #2196f3;
            color: white;
        }}

        /* Adjust edge text colors for different backgrounds */
        .row-correct .edge,
        .row-incorrect .edge,
        .row-close .edge {{
            color: inherit !important;
        }}

        .row-correct.underdog .edge.positive,
        .row-incorrect.underdog .edge.positive {{
            color: #2e7d32 !important;
        }}

        .row-correct.underdog .edge.negative,
        .row-incorrect.underdog .edge.negative {{
            color: #c62828 !important;
        }}

        /* Ensure fighter names are readable on dark backgrounds */
        .row-correct .fighter-name,
        .row-incorrect .fighter-name,
        .row-close .fighter-name {{
            color: white !important;
        }}

        .row-correct.underdog .fighter-name,
        .row-incorrect.underdog .fighter-name {{
            color: inherit !important;
        }}

        /* Market probability text color override */
        .market-prob-dark {{
            color: rgba(255,255,255,0.7) !important;
        }}

        .market-prob-light {{
            color: #757575 !important;
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

        <div style="padding: 0 30px 20px 30px;">
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 20px; border-radius: 8px; border-left: 5px solid #2196f3;">
                <h3 style="margin: 0 0 15px 0; color: #1565c0;">📊 Color Coding Guide</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 50px; height: 30px; background: #1b5e20; border-radius: 4px; margin-right: 12px; border-left: 4px solid #2e7d32;"></div>
                        <div>
                            <div style="font-weight: 600;">Dark Green</div>
                            <div style="font-size: 0.9em; color: #666;">Correct prediction (regular pick)</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 50px; height: 30px; background: #b71c1c; border-radius: 4px; margin-right: 12px; border-left: 4px solid #c62828;"></div>
                        <div>
                            <div style="font-weight: 600;">Dark Red</div>
                            <div style="font-size: 0.9em; color: #666;">Incorrect prediction (regular pick)</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 50px; height: 30px; background: #b2dfdb; border-radius: 4px; margin-right: 12px; border-left: 4px solid #009688;"></div>
                        <div>
                            <div style="font-weight: 600;">Blue-Green (Pastel)</div>
                            <div style="font-size: 0.9em; color: #666;">Correct underdog pick</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 50px; height: 30px; background: #ffccbc; border-radius: 4px; margin-right: 12px; border-left: 4px solid #ff7043;"></div>
                        <div>
                            <div style="font-weight: 600;">Orange-Red (Pastel)</div>
                            <div style="font-size: 0.9em; color: #666;">Incorrect underdog pick</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="background: linear-gradient(135deg, #ffd700 0%, #ffb300 100%); padding: 5px 12px; border-radius: 4px; margin-right: 12px; font-weight: bold; color: #5d4037; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">⭐ Top 25%</div>
                        <div>
                            <div style="font-weight: 600;">Top 25% Badge</div>
                            <div style="font-size: 0.9em; color: #666;">Highest model confidence on this card</div>
                        </div>
                    </div>
                </div>
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

        # Calculate top 25% by model confidence for this event
        event_fights = event_fights.copy()
        if len(event_fights) > 0:
            # confidence_threshold_75th = 75th percentile (top 25%)
            confidence_threshold_75th = event_fights["model_confidence"].quantile(0.75)
            event_fights["is_top_25_pct"] = event_fights["model_confidence"] >= confidence_threshold_75th

            # Also mark underdogs (model picked the market underdog)
            # Use the display column that checks if THIS ROW (winner perspective) was an underdog pick
            if "is_underdog_pick_display" in event_fights.columns:
                event_fights["is_underdog_pick"] = event_fights["is_underdog_pick_display"]
            else:
                event_fights["is_underdog_pick"] = False

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
                    <div style="background: #fff3e0; padding: 10px 15px; margin-bottom: 15px; border-radius: 6px; border-left: 4px solid #ff9800;">
                        <strong>📊 Legend:</strong>
                        <span style="margin-left: 15px;"><span style="display: inline-block; width: 12px; height: 12px; background: #1b5e20; border-radius: 2px; margin-right: 5px;"></span> Dark green/red = Regular picks</span>
                        <span style="margin-left: 15px;"><span style="display: inline-block; width: 12px; height: 12px; background: #b2dfdb; border-radius: 2px; margin-right: 5px;"></span> Blue-green/Orange = Underdog picks</span>
                        <span style="margin-left: 15px;">⭐ <strong>Top 25%</strong> = Highest model confidence on this card</span>
                    </div>
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
            is_top_25 = fight.get("is_top_25_pct", False)
            is_underdog = fight.get("is_underdog_pick", False)

            # Determine row class
            if is_correct:
                if confidence > 0.6:
                    row_class = "row-correct"
                else:
                    row_class = "row-close"  # Correct but low confidence
            else:
                row_class = "row-incorrect"

            # Add underdog class if applicable
            if is_underdog:
                row_class += " underdog"

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

            # Result icon - simple: check if correct, X if wrong
            # For dark backgrounds (row-correct, row-incorrect), use white
            # For underdog pastel backgrounds, use darker colors
            if is_correct:
                result_icon = "✓"
                if is_underdog:
                    result_color = "#00695c"  # Dark teal for pastel background
                else:
                    result_color = "#ffffff"  # White for dark background
            else:
                result_icon = "✗"
                if is_underdog:
                    result_color = "#bf360c"  # Dark orange for pastel background
                else:
                    result_color = "#ffffff"  # White for dark background

            # Build top 25% badge if applicable
            top_25_badge = '<span class="top-25-badge">⭐ Top 25%</span>' if is_top_25 else ''

            # Determine market probability class (dark vs light background)
            market_prob_class = "market-prob-dark" if not is_underdog else "market-prob-light"

            html += f"""
                            <tr class="{row_class}">
                                <td>
                                    <div class="fighter-name">{f1_name}</div>
                                    <div style="color: #999; font-size: 0.9em;">vs</div>
                                    <div class="fighter-name">{f2_name}</div>
                                    {top_25_badge}
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
                                <td style="text-align: center; font-size: 1.1em; font-weight: 600;" class="{market_prob_class}">
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

