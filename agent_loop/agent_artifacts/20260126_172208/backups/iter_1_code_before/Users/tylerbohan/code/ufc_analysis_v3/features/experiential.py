"""
Experiential Features
Career statistics, fight history, experience, and record-based features
"""

import pandas as pd
from typing import Dict, Optional
from database.schema import Fighter

from .utils import (
    safe_divide, ensure_numeric, is_finish, is_ko, 
    is_submission, is_decision, calculate_rolling_rate
)


def extract_career_stats(
    fighter: Fighter,
    fight_history: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract career statistics from fighter and fight history.
    
    Prefers deriving stats from actual fight history in database over
    stored Fighter record to keep features consistent with training data scope.
    
    Args:
        fighter: Fighter database object
        fight_history: DataFrame with fight history
        
    Returns:
        Dictionary of career statistics
    """
    # Derive record from fights in database
    if len(fight_history) == 0:
        wins = fighter.wins or 0
        losses = fighter.losses or 0
        draws = fighter.draws or 0
        total_fights = wins + losses + draws
    else:
        wins = len(fight_history[fight_history['result'] == 'win'])
        losses = len(fight_history[fight_history['result'] == 'loss'])
        draws = len(fight_history[fight_history['result'] == 'draw'])
        total_fights = len(fight_history)
    
    # Cap wins to prevent very long unbeaten runs from dominating
    wins_capped = min(wins, 15)
    
    features = {
        "total_fights": float(total_fights),
        "wins": float(wins_capped),
        "losses": float(losses),
        "draws": float(draws),
    }
    
    return features


def extract_fight_history_features(fight_history: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features from overall fight history.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        
    Returns:
        Dictionary of fight history features
    """
    if len(fight_history) == 0:
        return {
            "has_fight_history": 0.0,
            "finish_rate": 0.0,
            "ko_rate": 0.0,
            "submission_rate": 0.0,
            "decision_rate": 0.0,
            "title_fight_experience": 0.0,
            "avg_fight_duration_rounds": 0.0,
        }
    
    total_fights = len(fight_history)
    method_series = fight_history['method'].astype(str)
    
    # Finish rates
    finishes = method_series.str.contains('KO|TKO|SUB|Submission', na=False, case=False).sum()
    kos = method_series.str.contains('KO|TKO', na=False, case=False).sum()
    subs = method_series.str.contains('SUB|Submission', na=False, case=False).sum()
    decisions = method_series.str.contains('DEC|Decision', na=False, case=False).sum()
    
    features = {
        "has_fight_history": 1.0,
        "finish_rate": safe_divide(finishes, total_fights),
        "ko_rate": safe_divide(kos, total_fights),
        "submission_rate": safe_divide(subs, total_fights),
        "decision_rate": safe_divide(decisions, total_fights),
        "title_fight_experience": float(
            (fight_history['is_title_fight'] == True).sum() if 'is_title_fight' in fight_history.columns else 0
        ),
        "avg_fight_duration_rounds": float(
            fight_history['round'].mean() if 'round' in fight_history.columns and len(fight_history) > 0 else 0.0
        ),
    }
    
    return features


def extract_early_finishing_features(fight_history: pd.DataFrame) -> Dict[str, float]:
    """
    Extract early-finish / KO power profile features.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        
    Returns:
        Dictionary of early finishing features
    """
    if len(fight_history) == 0:
        return {
            "first_round_finish_rate": 0.0,
            "first_round_ko_rate": 0.0,
            "early_finish_rate_last_3": 0.0,
            "early_finish_rate_last_5": 0.0,
        }
    
    df = fight_history
    wins_mask = df["result"] == "win"
    wins_df = df[wins_mask]
    
    if len(wins_df) == 0:
        return {
            "first_round_finish_rate": 0.0,
            "first_round_ko_rate": 0.0,
            "early_finish_rate_last_3": 0.0,
            "early_finish_rate_last_5": 0.0,
        }
    
    method_series = wins_df["method"].astype(str)
    finish_mask = method_series.str.contains("KO|TKO|SUB|Submission", case=False, na=False)
    ko_mask = method_series.str.contains("KO|TKO", case=False, na=False)
    
    # First round finish rate (among wins)
    first_round_finishes = ((wins_df["round"] == 1) & finish_mask).sum()
    first_round_finish_rate = safe_divide(first_round_finishes, len(wins_df))
    
    # First round KO rate (among KO wins)
    ko_wins = wins_df[ko_mask]
    if len(ko_wins) > 0:
        first_round_kos = (ko_wins["round"] == 1).sum()
        first_round_ko_rate = safe_divide(first_round_kos, len(ko_wins))
    else:
        first_round_ko_rate = 0.0
    
    # Early finish rates in recent windows (rounds 1-2)
    def _early_finish_rate_over_window(window: int) -> float:
        recent = df.head(window)
        if len(recent) == 0:
            return 0.0
        recent_wins = recent[recent["result"] == "win"]
        if len(recent_wins) == 0:
            return 0.0
        
        recent_methods = recent_wins["method"].astype(str)
        recent_finish_mask = recent_methods.str.contains("KO|TKO|SUB|Submission", case=False, na=False)
        early_mask = (recent_wins["round"] <= 2) & recent_finish_mask
        
        return safe_divide(early_mask.sum(), len(recent_wins))
    
    early_finish_rate_last_3 = _early_finish_rate_over_window(3)
    early_finish_rate_last_5 = _early_finish_rate_over_window(5)
    
    return {
        "first_round_finish_rate": float(first_round_finish_rate),
        "first_round_ko_rate": float(first_round_ko_rate),
        "early_finish_rate_last_3": float(early_finish_rate_last_3),
        "early_finish_rate_last_5": float(early_finish_rate_last_5),
    }


def extract_round_3_features(fight_history: pd.DataFrame) -> Dict[str, float]:
    """
    Extract round 3 performance features as cardio proxy.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        
    Returns:
        Dictionary of round 3 performance features
    """
    if len(fight_history) == 0:
        return {
            "round_3_fight_rate": 0.0,
            "round_3_win_rate": 0.0,
            "round_3_finish_rate": 0.0,
            "round_3_performance_score": 0.0,
        }
    
    total_fights = len(fight_history)
    
    # Filter fights that went to round 3 or beyond
    round_3_fights = fight_history[
        (fight_history['round'] >= 3) & (fight_history['round'].notna())
    ]
    
    if len(round_3_fights) == 0:
        return {
            "round_3_fight_rate": 0.0,
            "round_3_win_rate": 0.0,
            "round_3_finish_rate": 0.0,
            "round_3_performance_score": 0.0,
        }
    
    # Round 3 fight rate (% of fights that reach round 3)
    round_3_fight_rate = safe_divide(len(round_3_fights), total_fights)
    
    # Win rate in fights that go to round 3
    round_3_wins = (round_3_fights['result'] == 'win').sum()
    round_3_win_rate = safe_divide(round_3_wins, len(round_3_fights))
    
    # Finish rate in round 3 (finishing opponent in round 3)
    round_3_finishes = (
        (round_3_fights['round'] == 3) &
        (round_3_fights['result'] == 'win') &
        (round_3_fights['method'].str.contains('KO|TKO|SUB|Submission', na=False, case=False))
    ).sum()
    round_3_finish_rate = safe_divide(round_3_finishes, len(round_3_fights))
    
    # Combined performance score (weighted combination)
    round_3_performance_score = (
        (round_3_win_rate * 0.5) +
        (round_3_finish_rate * 0.3) +
        (round_3_fight_rate * 0.2)
    )
    
    return {
        "round_3_fight_rate": float(round_3_fight_rate),
        "round_3_win_rate": float(round_3_win_rate),
        "round_3_finish_rate": float(round_3_finish_rate),
        "round_3_performance_score": float(round_3_performance_score),
    }

