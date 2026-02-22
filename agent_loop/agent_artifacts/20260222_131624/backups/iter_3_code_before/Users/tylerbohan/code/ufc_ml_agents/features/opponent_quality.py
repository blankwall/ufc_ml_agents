"""
Opponent Quality Features
Strength of schedule, opponent win rates, and quality-adjusted metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
from datetime import datetime

from .utils import safe_mean, safe_divide


def extract_opponent_quality_features(
    fight_history: pd.DataFrame,
    get_fighter_record: Callable[[int], Optional[Dict]]
) -> Dict[str, float]:
    """
    Estimate the quality of opposition this fighter has faced and beaten.
    
    This is a "strength of schedule" style feature set, based on opponents'
    overall records. To avoid overrating small-sample undefeated runs or very
    old wins, we restrict to fights in the last ~3 years when computing aggregates.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        get_fighter_record: Function that takes fighter_id and returns record dict
            with keys: wins, losses, draws, total_fights, win_rate
        
    Returns:
        Dictionary of opponent quality features
    """
    if len(fight_history) == 0:
        return {
            "avg_opponent_win_rate": 0.0,
            "avg_beaten_opponent_win_rate": 0.0,
            "avg_lost_to_opponent_win_rate": 0.0,
            "avg_opponent_total_fights": 0.0,
            "avg_beaten_opponent_total_fights": 0.0,
            "opponent_quality_score": 0.0,
        }
    
    df = fight_history
    
    # Restrict to recent window (last 3 years) so very old wins don't dominate
    if "event_date_parsed" in df.columns:
        now = datetime.now()
        three_years_ago = now - pd.DateOffset(years=3)
        df_recent = df[df["event_date_parsed"] >= three_years_ago]
        if len(df_recent) == 0:
            df_recent = df  # fall back to full history if no recent fights
    else:
        df_recent = df
    
    all_opp_win_rates = []
    beaten_opp_win_rates = []
    lost_to_opp_win_rates = []
    
    all_opp_total_fights = []
    beaten_opp_total_fights = []
    
    for _, row in df_recent.iterrows():
        opponent_id = row.get("opponent_id")
        if pd.isna(opponent_id):
            continue
        
        record = get_fighter_record(int(opponent_id))
        if record is None:
            continue
        
        wr = record.get("win_rate", 0.0)
        tf = record.get("total_fights", 0)
        
        all_opp_win_rates.append(wr)
        all_opp_total_fights.append(tf)
        
        if row.get("result") == "win":
            beaten_opp_win_rates.append(wr)
            beaten_opp_total_fights.append(tf)
        elif row.get("result") == "loss":
            lost_to_opp_win_rates.append(wr)
    
    avg_opp_wr = safe_mean(all_opp_win_rates)
    avg_lost_opp_wr = safe_mean(lost_to_opp_win_rates) if lost_to_opp_win_rates else 0.0
    avg_beaten_opp_wr = safe_mean(beaten_opp_win_rates)
    
    num_beaten_opponents = len(beaten_opp_win_rates)
    num_losses = len(lost_to_opp_win_rates)
    
    # FIXED: Properly account for elite losses
    # The key insight: losing to elite fighters (high win rate) shouldn't hurt as much
    # as losing to weak fighters (low win rate)
    # 
    # Old formula: raw_score = avg_beaten_opp_wr - avg_lost_opp_wr
    # Problem: Losing to 0.8 win rate opponent = -0.8 penalty (bad!)
    #
    # New formula: raw_score = avg_beaten_opp_wr - (1 - avg_lost_opp_wr)
    # Result: Losing to 0.8 win rate opponent = -0.2 penalty (small)
    #         Losing to 0.3 win rate opponent = -0.7 penalty (large)
    #
    # This properly rewards beating good opponents while penalizing losses to weak ones
    
    if num_losses > 0:
        # Penalty is inverse of opponent quality: losing to elite = small penalty
        # losing to weak = large penalty
        loss_penalty = 1.0 - avg_lost_opp_wr
        
        # Shrink penalty when sample is tiny (few losses)
        if num_beaten_opponents < 3:
            loss_penalty = loss_penalty * (num_beaten_opponents / 3.0)
        
        raw_score = avg_beaten_opp_wr - loss_penalty
    else:
        # No losses = no penalty, just reward for beating good opponents
        raw_score = avg_beaten_opp_wr
    
    # Keep avg_lost_opp_wr_adjusted for backward compatibility (used in other features)
    avg_lost_opp_wr_adjusted = avg_lost_opp_wr
    
    avg_opp_total_fights = safe_mean(all_opp_total_fights)
    avg_beaten_opp_total_fights = safe_mean(beaten_opp_total_fights)
    
    # Sample-aware shrinkage: small number of beaten opponents or very low
    # opponent mileage should pull the score toward 0.
    
    # Opponent mileage factor: 0 at 0 fights, 1 once opponents average ~10 fights.
    strength_factor = min(1.0, avg_opp_total_fights / 10.0) if avg_opp_total_fights > 0 else 0.0
    # Sample size factor: 0 at 0 wins, 1 once they've beaten ~5 distinct opponents.
    sample_factor = min(1.0, num_beaten_opponents / 5.0) if num_beaten_opponents > 0 else 0.0
    
    opponent_quality_score = raw_score * strength_factor * sample_factor
    
    # Clamp to reasonable range
    # Note: With the fixed formula, scores can legitimately be higher
    # (e.g., beating 0.8 win rate opponents with no losses = 0.8 score)
    # Old clamp of 0.35 was too restrictive. New range allows full expression.
    opponent_quality_score = max(min(opponent_quality_score, 1.0), -1.0)
    
    return {
        "avg_opponent_win_rate": float(avg_opp_wr),
        "avg_beaten_opponent_win_rate": float(avg_beaten_opp_wr),
        "avg_lost_to_opponent_win_rate": float(avg_lost_opp_wr_adjusted),
        "avg_opponent_total_fights": float(avg_opp_total_fights),
        "avg_beaten_opponent_total_fights": float(avg_beaten_opp_total_fights),
        "opponent_quality_score": float(opponent_quality_score),
    }

