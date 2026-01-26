"""
Time-Based Features
Rolling statistics, momentum, decline, activity, and time-decayed metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Callable, Union
from datetime import datetime

from .utils import (
    safe_divide, safe_mean, ensure_numeric, is_finish, is_ko,
    calculate_time_decayed_metric
)


def extract_rolling_stats(
    fight_history: pd.DataFrame,
    rolling_windows: List[int] = [3, 5]
) -> Dict[str, float]:
    """
    Calculate rolling statistics over different windows.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        rolling_windows: List of window sizes (e.g., [3, 5, 10])
        
    Returns:
        Dictionary of rolling statistics
    """
    features = {}
    
    if len(fight_history) == 0:
        for window in rolling_windows:
            features[f'win_rate_last_{window}'] = 0.0
            features[f'finish_rate_last_{window}'] = 0.0
            features[f'ko_rate_last_{window}'] = 0.0
        features['performance_trend'] = 0.0
        features['athleticism_decline'] = 0.0
        return features
    
    for window in rolling_windows:
        recent_fights = fight_history.head(window)
        
        if len(recent_fights) == 0:
            features[f'win_rate_last_{window}'] = 0.0
            features[f'finish_rate_last_{window}'] = 0.0
            features[f'ko_rate_last_{window}'] = 0.0
            continue
        
        wins = (recent_fights['result'] == 'win').sum()
        finishes = recent_fights['method'].str.contains(
            'KO|TKO|SUB|Submission', na=False, case=False
        ).sum()
        
        features[f'win_rate_last_{window}'] = safe_divide(wins, len(recent_fights))
        features[f'finish_rate_last_{window}'] = safe_divide(finishes, len(recent_fights))
        
        # KO rate among wins
        kos_recent = recent_fights[
            recent_fights['method'].str.contains('KO|TKO', na=False, case=False)
        ]
        ko_wins_recent = ((kos_recent['result'] == 'win')).sum()
        wins_recent = wins
        features[f'ko_rate_last_{window}'] = safe_divide(ko_wins_recent, wins_recent)
    
    # Performance trend: difference between last 3 and last 5
    if 'win_rate_last_3' in features and 'win_rate_last_5' in features:
        features['performance_trend'] = features['win_rate_last_3'] - features['win_rate_last_5']
    else:
        features['performance_trend'] = 0.0
    
    # Athleticism decline: career KO rate vs recent KO rate
    if 'ko_rate' in features and 'ko_rate_last_3' in features:
        features['athleticism_decline'] = features['ko_rate'] - features['ko_rate_last_3']
    else:
        features['athleticism_decline'] = 0.0
    
    return features


def extract_momentum_features(
    fight_history: pd.DataFrame,
    as_of_date: Optional[Union[datetime, str]] = None,
) -> Dict[str, float]:
    """
    Extract momentum, recent form, and activity timing features.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        
    Returns:
        Dictionary of momentum features
    """
    if len(fight_history) == 0:
        return {
            'current_win_streak': 0.0,
            'current_loss_streak': 0.0,
            'fights_in_last_year': 0.0,
            'activity_rate': 0.0,
            'days_since_last_fight': 0.0,
            'days_between_last_2_fights': 0.0,
            'long_layoff_over_1yr': 0.0,
            'long_layoff_over_2yr': 0.0,
        }
    
    # Calculate win/loss streaks
    win_streak = 0
    loss_streak = 0
    
    for _, fight in fight_history.iterrows():
        if fight['result'] == 'win':
            win_streak += 1
            loss_streak = 0
        elif fight['result'] == 'loss':
            loss_streak += 1
            win_streak = 0
        else:
            break
    
    # Activity rate (fights per year) - approximate (last up to 4 fights)
    fights_in_last_year = len(fight_history.head(min(len(fight_history), 4)))
    
    # Days since last fight and spacing
    days_since_last_fight = 0.0
    days_between_last_2 = 0.0
    
    try:
        if 'event_date_parsed' in fight_history.columns:
            last_date = fight_history['event_date_parsed'].iloc[0]
            # Point-in-time safe reference: use as_of_date when provided (historical eval),
            # otherwise use "now" (upcoming/live usage).
            if as_of_date is not None:
                ref = pd.to_datetime(as_of_date)
            else:
                ref = datetime.now()
            days_since_last_fight = max(0.0, (ref - last_date).days)
            
            if len(fight_history) > 1:
                prev_date = fight_history['event_date_parsed'].iloc[1]
                days_between_last_2 = max(0.0, (last_date - prev_date).days)
    except Exception:
        days_since_last_fight = 0.0
        days_between_last_2 = 0.0
    
    features = {
        'current_win_streak': float(win_streak),
        'current_loss_streak': float(loss_streak),
        'fights_in_last_year': float(fights_in_last_year),
        'activity_rate': float(fights_in_last_year),
        'days_since_last_fight': float(days_since_last_fight),
        'days_between_last_2_fights': float(days_between_last_2),
        'long_layoff_over_1yr': 1.0 if days_since_last_fight >= 365 else 0.0,
        'long_layoff_over_2yr': 1.0 if days_since_last_fight >= 730 else 0.0,
    }
    
    return features


def extract_decline_features(
    fight_history: pd.DataFrame,
    as_of_date: Optional[Union[datetime, str]] = None,
) -> Dict[str, float]:
    """
    Extract longer-horizon decline / slump patterns.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        
    Returns:
        Dictionary of decline features
    """
    if len(fight_history) == 0:
        return {
            "fights_since_last_win": 0.0,
            "years_since_last_win": 0.0,
            "has_ever_won": 0.0,
            "losses_since_last_win": 0.0,
            "decision_losses_since_last_win": 0.0,
            "finish_losses_since_last_win": 0.0,
            "recent_vs_career_win_rate": 0.0,
            "win_rate_last_3_years": 0.0,
            "wins_last_3_years": 0.0,
            "losses_last_3_years": 0.0,
            "finish_rate_last_3_years": 0.0,
        }
    
    df = fight_history
    
    # Locate most recent win
    win_rows = df[df["result"] == "win"]
    if len(win_rows) == 0:
        fights_since_last_win = len(df)
        years_since_last_win = 0.0
        has_ever_won = 0.0
        slump = df
    else:
        last_win_idx = int(win_rows.index[0])
        last_win_date = df.loc[last_win_idx, "event_date_parsed"]
        
        ref_date = pd.to_datetime(as_of_date) if as_of_date is not None else datetime.now()
        fights_since_last_win = last_win_idx
        years_since_last_win = max(0.0, (ref_date - last_win_date).days / 365.25)
        has_ever_won = 1.0
        
        slump = df.iloc[:last_win_idx]
    
    # Losses in slump
    if len(slump) == 0:
        losses_since_last_win = 0.0
        decision_losses_since_last_win = 0.0
        finish_losses_since_last_win = 0.0
    else:
        losses_mask = slump["result"] == "loss"
        method_series = slump["method"].astype(str)
        decision_mask = method_series.str.contains("DEC|Decision", case=False, na=False)
        finish_mask = method_series.str.contains("KO|TKO|SUB|Submission", case=False, na=False)
        
        losses_since_last_win = float(losses_mask.sum())
        decision_losses_since_last_win = float((losses_mask & decision_mask).sum())
        finish_losses_since_last_win = float((losses_mask & finish_mask).sum())
    
    # Career vs recent win rate
    total_fights = len(df)
    career_wins = (df["result"] == "win").sum()
    career_win_rate = safe_divide(career_wins, total_fights)
    
    recent = df.head(5)
    if len(recent) > 0:
        recent_wins = (recent["result"] == "win").sum()
        recent_win_rate_last_5 = safe_divide(recent_wins, len(recent))
    else:
        recent_win_rate_last_5 = 0.0
    
    recent_vs_career_win_rate = recent_win_rate_last_5 - career_win_rate
    
    # Time-windowed (3-year) recent performance
    ref_date = pd.to_datetime(as_of_date) if as_of_date is not None else datetime.now()
    three_years_ago = ref_date - pd.DateOffset(years=3)
    
    recent_window = df[df["event_date_parsed"] >= three_years_ago] if "event_date_parsed" in df.columns else pd.DataFrame()
    
    if len(recent_window) > 0:
        wins_last_3_years = (recent_window["result"] == "win").sum()
        losses_last_3_years = (recent_window["result"] == "loss").sum()
        total_last_3_years = len(recent_window)
        win_rate_last_3_years = safe_divide(wins_last_3_years, total_last_3_years)
        
        method_series_recent = recent_window["method"].astype(str)
        finish_mask_recent = method_series_recent.str.contains(
            "KO|TKO|SUB|Submission", case=False, na=False
        )
        finishes_last_3_years = (
            ((recent_window["result"] == "win") & finish_mask_recent).sum()
        )
        finish_rate_last_3_years = safe_divide(finishes_last_3_years, wins_last_3_years)
    else:
        wins_last_3_years = 0.0
        losses_last_3_years = 0.0
        win_rate_last_3_years = 0.0
        finish_rate_last_3_years = 0.0
    
    return {
        "fights_since_last_win": float(fights_since_last_win),
        "years_since_last_win": float(years_since_last_win),
        "has_ever_won": float(has_ever_won),
        "losses_since_last_win": losses_since_last_win,
        "decision_losses_since_last_win": decision_losses_since_last_win,
        "finish_losses_since_last_win": finish_losses_since_last_win,
        "recent_vs_career_win_rate": float(recent_vs_career_win_rate),
        "win_rate_last_3_years": float(win_rate_last_3_years),
        "wins_last_3_years": float(wins_last_3_years),
        "losses_last_3_years": float(losses_last_3_years),
        "finish_rate_last_3_years": float(finish_rate_last_3_years),
    }


def extract_recent_damage_features(
    fight_history: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract very recent performance, especially bad losses and finishes.
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        
    Returns:
        Dictionary of recent damage features
    """
    if len(fight_history) == 0:
        return {
            "recent_losses_last_2": 0.0,
            "recent_finish_losses_last_2": 0.0,
            "recent_finish_loss_last_fight": 0.0,
            "recent_finish_loss_ratio_last_2": 0.0,
        }
    
    recent_2 = fight_history.head(2)
    recent_1 = fight_history.head(1)
    
    def _is_finish_loss(row) -> bool:
        if row.get("result") != "loss":
            return False
        method = str(row.get("method") or "")
        return bool(pd.notna(method) and (
            "KO" in method.upper() or
            "TKO" in method.upper() or
            "SUB" in method.upper() or
            "SUBMISSION" in method.upper()
        ))
    
    # Losses and finish-losses in last 2
    losses_last_2 = 0.0
    finish_losses_last_2 = 0.0
    for _, row in recent_2.iterrows():
        if row.get("result") == "loss":
            losses_last_2 += 1.0
        if _is_finish_loss(row):
            finish_losses_last_2 += 1.0
    
    # Was the most recent fight a bad finish loss?
    last_fight_finish_loss = 0.0
    if len(recent_1) == 1:
        last_row = recent_1.iloc[0]
        if _is_finish_loss(last_row):
            last_fight_finish_loss = 1.0
    
    fights_considered = max(1, len(recent_2))
    finish_loss_ratio_last_2 = safe_divide(finish_losses_last_2, fights_considered)
    
    return {
        "recent_losses_last_2": losses_last_2,
        "recent_finish_losses_last_2": finish_losses_last_2,
        "recent_finish_loss_last_fight": last_fight_finish_loss,
        "recent_finish_loss_ratio_last_2": finish_loss_ratio_last_2,
    }


def extract_time_decayed_features(
    fight_history: pd.DataFrame,
    lambda_decay: float = 0.3,
    as_of_date: Optional[Union[datetime, str]] = None,
) -> Dict[str, float]:
    """
    Compute time-decayed performance metrics where recent fights are weighted more heavily.
    
    Uses exponential decay: weight = exp(-lambda * years_ago)
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        lambda_decay: Decay rate (higher = faster decay, default 0.3)
        
    Returns:
        Dictionary with time-decayed metrics
    """
    if len(fight_history) == 0:
        return {
            "time_decayed_win_rate": 0.0,
            "time_decayed_finish_rate": 0.0,
            "time_decayed_ko_rate": 0.0,
        }
    
    if "event_date_parsed" not in fight_history.columns:
        # Fallback to simple win rate if dates aren't available
        wins = (fight_history["result"] == "win").sum()
        total = len(fight_history)
        return {
            "time_decayed_win_rate": safe_divide(wins, total),
            "time_decayed_finish_rate": 0.0,
            "time_decayed_ko_rate": 0.0,
        }
    
    now = pd.to_datetime(as_of_date) if as_of_date is not None else datetime.now()
    method_series = fight_history["method"].astype(str)
    finish_mask = method_series.str.contains("KO|TKO|SUB|Submission", case=False, na=False)
    ko_mask = method_series.str.contains("KO|TKO", case=False, na=False)
    
    total_weight = 0.0
    win_weight = 0.0
    finish_weight = 0.0
    ko_weight = 0.0
    
    for _, row in fight_history.iterrows():
        try:
            fight_date = row["event_date_parsed"]
            years_ago = (now - fight_date).days / 365.25
            weight = np.exp(-lambda_decay * years_ago)
            
            total_weight += weight
            
            if row["result"] == "win":
                win_weight += weight
                if finish_mask.loc[row.name]:
                    finish_weight += weight
                if ko_mask.loc[row.name]:
                    ko_weight += weight
        except Exception:
            continue
    
    if total_weight == 0:
        return {
            "time_decayed_win_rate": 0.0,
            "time_decayed_finish_rate": 0.0,
            "time_decayed_ko_rate": 0.0,
        }
    
    time_decayed_win_rate = safe_divide(win_weight, total_weight)
    time_decayed_finish_rate = safe_divide(finish_weight, win_weight)
    time_decayed_ko_rate = safe_divide(ko_weight, win_weight)
    
    return {
        "time_decayed_win_rate": float(time_decayed_win_rate),
        "time_decayed_finish_rate": float(time_decayed_finish_rate),
        "time_decayed_ko_rate": float(time_decayed_ko_rate),
    }


def extract_opponent_quality_adjusted_time_decayed_features(
    fight_history: pd.DataFrame,
    get_fighter_record: Callable[[int], Optional[Dict]],
    lambda_decay: float = 0.3
) -> Dict[str, float]:
    """
    Compute opponent-quality-adjusted time-decayed performance metrics.
    
    Weights wins/losses by opponent quality:
    - Win against strong opponent = higher weight
    - Loss to strong opponent = lower penalty
    - Uses exponential time decay: weight = exp(-lambda * years_ago)
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        get_fighter_record: Function that takes fighter_id and returns record dict
            with keys: wins, losses, draws, total_fights, win_rate
        lambda_decay: Decay rate (higher = faster decay, default 0.3)
        
    Returns:
        Dictionary with opponent-quality-adjusted time-decayed metrics
    """
    if len(fight_history) == 0:
        return {
            "time_decayed_win_rate_adj_opp_quality": 0.0,
        }
    
    if "event_date_parsed" not in fight_history.columns:
        # Fallback to simple win rate if dates aren't available
        wins = (fight_history["result"] == "win").sum()
        total = len(fight_history)
        return {
            "time_decayed_win_rate_adj_opp_quality": safe_divide(wins, total),
        }
    
    if "opponent_id" not in fight_history.columns:
        # Fallback to regular time_decayed_win_rate if opponent info missing
        return {
            "time_decayed_win_rate_adj_opp_quality": 0.0,
        }
    
    now = datetime.now()
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for _, row in fight_history.iterrows():
        try:
            fight_date = row["event_date_parsed"]
            years_ago = (now - fight_date).days / 365.25
            time_weight = np.exp(-lambda_decay * years_ago)
            
            opponent_id = row.get("opponent_id")
            if pd.isna(opponent_id):
                # If opponent info missing, use regular time weight
                opponent_quality_multiplier = 1.0
            else:
                # Get opponent record
                record = get_fighter_record(int(opponent_id))
                if record and record.get("win_rate") is not None:
                    opp_win_rate = record["win_rate"]
                    # Normalize opponent quality: stronger differentiation
                    # Scale from [0, 1] to [0.3, 2.3] so average opponent = 1.3x multiplier
                    # This creates stronger differentiation between elite and weak opponents
                    opponent_quality_multiplier = 0.3 + (opp_win_rate * 2.0)
                else:
                    # Unknown opponent = average quality
                    opponent_quality_multiplier = 1.0
            
            # Calculate score contribution
            if row["result"] == "win":
                # Win: positive score weighted by opponent quality
                # Win against elite opponent (2.3x) counts more than win against weak (0.3x)
                score_contribution = 1.0 * opponent_quality_multiplier
            elif row["result"] == "loss":
                # Loss: negative score, but loss to elite opponent is MUCH less penalized
                # Using exponential decay: -1.0 * exp(-multiplier)
                # Loss to elite (2.0x): -1.0 * exp(-2.0) = -0.135 (very small penalty!)
                # Loss to average (1.3x): -1.0 * exp(-1.3) = -0.273 (moderate penalty)
                # Loss to weak (0.3x): -1.0 * exp(-0.3) = -0.741 (larger penalty)
                score_contribution = -1.0 * np.exp(-opponent_quality_multiplier)
            else:
                # Draw/NC: neutral
                score_contribution = 0.0
            
            # Combined weight: time decay × opponent quality adjustment
            combined_weight = time_weight * opponent_quality_multiplier
            
            total_weighted_score += score_contribution * time_weight
            total_weight += time_weight
            
        except Exception:
            continue
    
    if total_weight == 0:
        return {
            "time_decayed_win_rate_adj_opp_quality": 0.0,
        }
    
    # Normalize score to [0, 1] range (win rate equivalent)
    # With multiplier range [0.3, 2.3] and exponential penalty:
    # - Worst case: all losses to weak (0.3x) = -1.0 * exp(-0.3) ≈ -0.741
    # - Best case: all wins vs elite (2.3x) = 1.0 * 2.3 = +2.3
    # So actual range is approximately [-0.741, 2.3]
    avg_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
    # Normalize from [-0.741, 2.3] to [0, 1]
    # Add 0.741 to shift to [0, 3.041], then divide by 3.041
    normalized_score = (avg_score + 0.741) / 3.041
    normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
    
    return {
        "time_decayed_win_rate_adj_opp_quality": float(normalized_score),
    }


def extract_age_interactions(
    age: float,
    decline_features: Dict[str, float],
    momentum_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Create age × decline and age × activity interaction features.
    
    Args:
        age: Fighter age
        decline_features: Dictionary of decline features
        momentum_features: Dictionary of momentum/activity features
        
    Returns:
        Dictionary of age interaction features
    """
    try:
        age = float(age or 0.0)
        years_since_last_win = float(decline_features.get("years_since_last_win", 0.0) or 0.0)
        fights_since_last_win = float(decline_features.get("fights_since_last_win", 0) or 0)
        recent_vs_career_win_rate = float(decline_features.get("recent_vs_career_win_rate", 0.0) or 0.0)
        
        days_since_last_fight = float(momentum_features.get("days_since_last_fight", 0) or 0)
        fights_in_last_year = float(momentum_features.get("fights_in_last_year", 0) or 0)
        
        # Only penalize decline (negative recent_vs_career means declining)
        decline_magnitude = max(0.0, -recent_vs_career_win_rate)
        
        # Convert days to years for better scaling
        years_since_last_fight = days_since_last_fight / 365.25
        
        return {
            "age_x_years_since_last_win": float(age * years_since_last_win),
            "age_x_fights_since_last_win": float(age * fights_since_last_win),
            "age_x_recent_vs_career_decline": float(age * decline_magnitude),
            "age_x_days_since_last_fight": float(age * days_since_last_fight),
            "age_x_years_since_last_fight": float(age * years_since_last_fight),
            "age_x_fights_in_last_year": float(age * fights_in_last_year),
        }
    except Exception:
        return {
            "age_x_years_since_last_win": 0.0,
            "age_x_fights_since_last_win": 0.0,
            "age_x_recent_vs_career_decline": 0.0,
            "age_x_days_since_last_fight": 0.0,
            "age_x_years_since_last_fight": 0.0,
            "age_x_fights_in_last_year": 0.0,
        }


def extract_youth_form_score(
    age: float,
    win_rate_last_5: float,
    finish_rate_last_5: float
) -> float:
    """
    Calculate youth + recent form interaction score.
    
    Helps capture surging young prospects.
    
    Args:
        age: Fighter age
        win_rate_last_5: Win rate in last 5 fights
        finish_rate_last_5: Finish rate in last 5 fights
        
    Returns:
        Youth form score
    """
    try:
        age = float(age or 0.0)
        win_rate_last_5 = float(win_rate_last_5 or 0.0)
        finish_rate_last_5 = float(finish_rate_last_5 or 0.0)
        
        base_form = max(0.0, 0.8 * win_rate_last_5 + 0.2 * finish_rate_last_5)
        youth_factor = max(0.0, 30.0 - age)
        return float(base_form * youth_factor)
    except Exception:
        return 0.0


def extract_prospect_momentum_score(
    age: float,
    win_rate_last_5: float,
    finish_rate_last_5: float
) -> float:
    """
    Calculate prospect momentum score.
    
    Rewards young fighters with strong recent form and finishing ability.
    Formula: win_rate_last_5 × finish_rate_last_5 × (1 - age/35)
    
    This helps capture rising prospects who are finishing fights and winning,
    while reducing the penalty for facing weaker opponents (common for prospects).
    
    Args:
        age: Fighter age
        win_rate_last_5: Win rate in last 5 fights
        finish_rate_last_5: Finish rate in last 5 fights (finishes / wins)
        
    Returns:
        Prospect momentum score (0.0 to 1.0)
    """
    try:
        age = float(age or 0.0)
        win_rate_last_5 = float(win_rate_last_5 or 0.0)
        finish_rate_last_5 = float(finish_rate_last_5 or 0.0)
        
        # Age factor: (1 - age/35) gives maximum at age 0, decreases to 0 at age 35
        # Clamp to [0, 1] range
        age_factor = max(0.0, min(1.0, 1.0 - (age / 35.0)))
        
        # Prospect momentum: win rate × finish rate × age factor
        # All components are in [0, 1] range, so result is in [0, 1]
        prospect_momentum = win_rate_last_5 * finish_rate_last_5 * age_factor
        
        return float(prospect_momentum)
    except Exception:
        return 0.0


def extract_age_weighted_recent_damage(
    age: float,
    recent_sig_strike_diff_last_3: float
) -> float:
    """
    Calculate age-weighted recent damage.
    
    Recent damage matters more when a fighter is old. This feature multiplies
    recent striking differential by an age decline multiplier that activates
    after age ~32 and ramps steeply after 35.
    
    Formula:
        age_weighted_recent_damage = recent_sig_strike_diff_last_3 × age_decline_multiplier
        where age_decline_multiplier = max(0, (age - 32) / 6)
    
    Args:
        age: Fighter age
        recent_sig_strike_diff_last_3: Average significant strike differential in last 3 fights
        
    Returns:
        Age-weighted recent damage score
    """
    try:
        age = float(age or 0.0)
        recent_sig_strike_diff = float(recent_sig_strike_diff_last_3 or 0.0)
        
        # Age decline multiplier: activates after age 32, ramps steeply after 35
        # At age 32: multiplier = 0
        # At age 35: multiplier = 0.5
        # At age 38: multiplier = 1.0
        # At age 41: multiplier = 1.5
        age_decline_multiplier = max(0.0, (age - 32.0) / 6.0)
        
        # Multiply recent damage by age penalty
        age_weighted_damage = recent_sig_strike_diff * age_decline_multiplier
        
        return float(age_weighted_damage)
    except Exception:
        return 0.0


def extract_durability_collapse_score(
    age: float,
    recent_knockdown_diff_last_3: float,
    recent_finish_losses_last_2: float,
    athleticism_decline: float
) -> float:
    """
    Calculate durability collapse score.
    
    Heavily penalizes older fighters with recent damage, even if their career
    stats are elite. Combines recent knockdowns, finish losses, and athleticism
    decline, then age-gates it (reduces impact for fighters under 33).
    
    Formula:
        durability_collapse_score = (
            (recent_knockdowns_last_3 × 1.5) +
            (recent_finish_losses_last_2 × 2.0) +
            athleticism_decline
        )
        if age < 33: score *= 0.5
    
    Note: recent_knockdown_diff_last_3 is (my_kd - opp_kd), so negative values
    mean the fighter got knocked down more. We use max(0, -recent_knockdown_diff)
    to capture when the fighter was knocked down more (bad for durability).
    
    Args:
        age: Fighter age
        recent_knockdown_diff_last_3: Average knockdown differential in last 3 fights (my_kd - opp_kd)
        recent_finish_losses_last_2: Number of finish losses in last 2 fights
        athleticism_decline: Decline in KO rate (ko_rate - ko_rate_last_3)
        
    Returns:
        Durability collapse score (higher = more concerning)
    """
    try:
        age = float(age or 0.0)
        recent_knockdown_diff = float(recent_knockdown_diff_last_3 or 0.0)
        recent_finish_losses = float(recent_finish_losses_last_2 or 0.0)
        athleticism_decline_val = float(athleticism_decline or 0.0)
        
        # Convert knockdown diff to "knockdowns against fighter" (negative diff = bad)
        # If diff is negative, fighter got knocked down more, which is bad
        recent_knockdowns_against = max(0.0, -recent_knockdown_diff)
        
        # Calculate base score
        score = (
            1.5 * recent_knockdowns_against +
            2.0 * recent_finish_losses +
            athleticism_decline_val
        )
        
        # Age-gate: reduce impact for fighters under 33
        if age < 33.0:
            score *= 0.5
        
        return float(score)
    except Exception:
        return 0.0


def extract_early_finish_advantage(
    first_round_finish_rate: float,
    early_finish_rate_last_3: float,
    time_decayed_ko_rate: float
) -> float:
    """
    Calculate early finish advantage score.
    
    Combines multiple early finishing signals to identify fighters who finish fights quickly.
    Formula: first_round_finish_rate * 0.5 + early_finish_rate_last_3 * 0.3 + time_decayed_ko_rate * 0.2
    
    Args:
        first_round_finish_rate: Rate of finishing in first round (career)
        early_finish_rate_last_3: Rate of finishing in rounds 1-2 (last 3 fights)
        time_decayed_ko_rate: Time-decayed KO rate (recent KOs weighted more)
        
    Returns:
        Early finish advantage score (0.0 to 1.0)
    """
    try:
        first_round_finish_rate = float(first_round_finish_rate or 0.0)
        early_finish_rate_last_3 = float(early_finish_rate_last_3 or 0.0)
        time_decayed_ko_rate = float(time_decayed_ko_rate or 0.0)
        
        # Weighted combination of early finishing signals
        early_finish_advantage = (
            first_round_finish_rate * 0.5 +
            early_finish_rate_last_3 * 0.3 +
            time_decayed_ko_rate * 0.2
        )
        
        # Clamp to [0, 1] range
        return float(max(0.0, min(1.0, early_finish_advantage)))
    except Exception:
        return 0.0


def extract_power_striker_score(
    ko_rate_last_5: float,
    first_round_ko_rate: float,
    knockdowns_per_fight: float,
    head_strike_rate: float
) -> float:
    """
    Calculate power striker score.
    
    Identifies fighters with knockout power and head-hunting style.
    Formula: (ko_rate_last_5 * 0.4 + first_round_ko_rate * 0.3 + knockdowns_per_fight * 0.3) * head_strike_rate
    
    Args:
        ko_rate_last_5: KO rate in last 5 fights (KOs / wins)
        first_round_ko_rate: Rate of first round KOs (career)
        knockdowns_per_fight: Average knockdowns per fight (lifetime or last 3)
        head_strike_rate: Rate of strikes targeting head (lifetime or last 3)
        
    Returns:
        Power striker score (0.0 to 1.0)
    """
    try:
        ko_rate_last_5 = float(ko_rate_last_5 or 0.0)
        first_round_ko_rate = float(first_round_ko_rate or 0.0)
        knockdowns_per_fight = float(knockdowns_per_fight or 0.0)
        head_strike_rate = float(head_strike_rate or 0.0)
        
        # Normalize knockdowns_per_fight to reasonable range [0, 2.0]
        # Most fighters average 0-1 knockdowns per fight, elite power strikers can hit 1.5+
        normalized_knockdowns = min(2.0, knockdowns_per_fight) / 2.0
        
        # Weighted combination of power indicators
        power_components = (
            ko_rate_last_5 * 0.4 +
            first_round_ko_rate * 0.3 +
            normalized_knockdowns * 0.3
        )
        
        # Multiply by head strike rate (power strikers target the head)
        power_striker_score = power_components * head_strike_rate
        
        # Clamp to [0, 1] range
        return float(max(0.0, min(1.0, power_striker_score)))
    except Exception:
        return 0.0

