"""
Grappling Features
Takedowns, submissions, control time, and related grappling metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from database.schema import Fighter, Fight
from datetime import datetime

from .utils import safe_divide, ensure_numeric, parse_control_time_seconds, parse_landed


def extract_grappling_fight_details(fight: Fight, fighter_number: str) -> Optional[Dict[str, float]]:
    """
    Extract grappling metrics from a single fight.
    
    Args:
        fight: Fight database object
        fighter_number: "fighter_1" or "fighter_2"
        
    Returns:
        Dictionary of grappling metrics for this fight, or None if data unavailable
    """
    if not fight.fight_stats:
        return None
    
    totals = fight.fight_stats.fighter_1_totals if fighter_number == "fighter_1" else fight.fight_stats.fighter_2_totals
    opp_totals = fight.fight_stats.fighter_2_totals if fighter_number == "fighter_1" else fight.fight_stats.fighter_1_totals
    
    if not totals or not opp_totals:
        return None
    
    # Parse fight date for sorting
    try:
        dt = datetime.strptime(fight.event.date, "%B %d, %Y")
        fight_key = dt.strftime("%Y%m%d")
    except:
        fight_key = "00000000"
    
    # Duration in minutes
    duration_min = fight.round_finished * 5 if fight.round_finished else 15
    
    # Parse takedowns (format: "X of Y" or just "X")
    def parse_takedown_fraction(td_str):
        """Parse takedown string like '2 of 5' into (landed, attempted)"""
        if not td_str:
            return 0, 0
        td_str = str(td_str).strip()
        if " of " in td_str:
            try:
                parts = td_str.split(" of ")
                return int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, IndexError):
                return parse_landed(td_str), parse_landed(td_str)
        else:
            landed = parse_landed(td_str)
            return landed, max(landed, 1)  # If no attempts given, use landed as minimum
    
    takedowns_landed, takedowns_attempted = parse_takedown_fraction(totals.get("takedowns", "0"))
    opp_takedowns_landed, opp_takedowns_attempted = parse_takedown_fraction(opp_totals.get("takedowns", "0"))
    
    # Parse submission attempts
    submission_attempts = parse_landed(totals.get("submission_attempts", "0"))
    
    # Calculate per-15min rates
    takedown_avg_per_15min = (takedowns_landed / duration_min) * 15 if duration_min > 0 else 0.0
    submission_avg_per_15min = (submission_attempts / duration_min) * 15 if duration_min > 0 else 0.0
    
    # Calculate accuracy
    takedown_accuracy = safe_divide(takedowns_landed, takedowns_attempted, default=0.0)
    
    # Calculate defense (opponent's failed attempts / total attempts)
    # Defense = 1 - (opponent's landed / opponent's attempted)
    takedown_defense = 1.0 - safe_divide(opp_takedowns_landed, opp_takedowns_attempted, default=0.0) if opp_takedowns_attempted > 0 else 0.0
    
    return {
        "fight_key": fight_key,
        "duration_min": duration_min,
        "takedown_avg_per_15min": takedown_avg_per_15min,
        "takedown_accuracy": takedown_accuracy,
        "takedown_defense": takedown_defense,
        "submission_avg_per_15min": submission_avg_per_15min,
        "takedowns_landed": float(takedowns_landed),
        "takedowns_attempted": float(takedowns_attempted),
    }


def extract_grappling_features(context: Dict) -> Dict[str, float]:
    """
    Extract grappling-related features from a fighter.
    
    POINT-IN-TIME SAFE: Uses fight_history filtered by as_of_date to ensure
    features are computed strictly up to the fight date, not including future fights.
    
    Args:
        context: Dictionary containing:
            - fighter: Fighter database object
            - fight_history: DataFrame with fight history (date-filtered by as_of_date)
            - fight_stats_by_fight_id: Dictionary mapping fight_id to FightStats object
            - fighter_id: Fighter ID
            - session: Database session (for querying Fight objects)
            
    Returns:
        Dictionary of grappling features
    """
    fighter = context["fighter"]
    fight_history = context.get("fight_history", pd.DataFrame())
    fight_stats_by_fight_id = context.get("fight_stats_by_fight_id", {})
    fighter_id = context.get("fighter_id", fighter.id)
    
    recent_metrics = []
    
    # Fallback to global stats only if no fight history (for fighters with no fights)
    takedown_avg_per_15min_global = fighter.takedown_avg_per_15min or 0.0
    takedown_accuracy_global = fighter.takedown_accuracy or 0.0
    takedown_defense_global = fighter.takedown_defense or 0.0
    submission_avg_per_15min_global = fighter.submission_avg_per_15min or 0.0
    
    # POINT-IN-TIME SAFE: Use fight_history (already filtered by as_of_date) instead of
    # fighter.fights_as_fighter_1/2 which includes ALL fights regardless of date
    if len(fight_history) > 0 and "fight_id" in fight_history.columns:
        # Get fight IDs from date-filtered fight_history
        fight_ids = [
            int(fid) for fid in fight_history["fight_id"].tolist()
            if pd.notna(fid)
        ]
        
        # Query Fight objects for these specific fights
        session = context.get("session")
        if session is not None:
            fights = (
                session.query(Fight)
                .filter(Fight.id.in_(fight_ids))
                .all()
            )
            
            # Build a mapping of fight_id to Fight object
            fight_by_id = {f.id: f for f in fights}
            
            # Extract metrics from each fight in chronological order
            for _, row in fight_history.iterrows():
                fight_id = int(row["fight_id"])
                fight = fight_by_id.get(fight_id)
                if fight is None:
                    continue
                
                # Determine if this fighter was fighter_1 or fighter_2
                is_fighter_1 = bool(row.get("is_fighter_1", False))
                fighter_number = "fighter_1" if is_fighter_1 else "fighter_2"
                
                fight_metrics = extract_grappling_fight_details(fight, fighter_number)
                if fight_metrics is not None:
                    recent_metrics.append(fight_metrics)
    else:
        # Fallback: if no fight_history provided, use old method (for backward compatibility)
        # This should rarely happen in normal usage
        for fight in fighter.fights_as_fighter_1:
            fight_metrics = extract_grappling_fight_details(fight, "fighter_1")
            if fight_metrics is not None:
                recent_metrics.append(fight_metrics)
        
        for fight in fighter.fights_as_fighter_2:
            fight_metrics = extract_grappling_fight_details(fight, "fighter_2")
            if fight_metrics is not None:
                recent_metrics.append(fight_metrics)
    
    # Compute lifetime averages from fight metrics
    if recent_metrics:
        # Sort by fight_key (most recent first)
        recent_metrics.sort(key=lambda x: x['fight_key'], reverse=True)
        
        # Calculate lifetime averages
        takedown_avg_per_15min_lifetime = np.mean([m['takedown_avg_per_15min'] for m in recent_metrics])
        takedown_accuracy_lifetime = np.mean([m['takedown_accuracy'] for m in recent_metrics])
        takedown_defense_lifetime = np.mean([m['takedown_defense'] for m in recent_metrics])
        submission_avg_per_15min_lifetime = np.mean([m['submission_avg_per_15min'] for m in recent_metrics])
        
        # Use computed values from fight_history
        takedown_avg_per_15min_effective = takedown_avg_per_15min_lifetime
        takedown_accuracy_effective = takedown_accuracy_lifetime
        takedown_defense_effective = takedown_defense_lifetime
        submission_avg_per_15min_effective = submission_avg_per_15min_lifetime
    else:
        # No fight history - use global stats as fallback
        takedown_avg_per_15min_effective = takedown_avg_per_15min_global
        takedown_accuracy_effective = takedown_accuracy_global
        takedown_defense_effective = takedown_defense_global
        submission_avg_per_15min_effective = submission_avg_per_15min_global
    
    features = {
        "takedown_avg_per_15min": float(takedown_avg_per_15min_effective),
        "takedown_accuracy": float(takedown_accuracy_effective),
        "takedown_defense": float(takedown_defense_effective),
        "submission_avg_per_15min": float(submission_avg_per_15min_effective),
    }
    
    return features


def extract_recent_grappling_features(
    fight_history: pd.DataFrame,
    fight_stats_by_fight_id: Dict[int, any],
    fighter_id: int,
    window: int = 3
) -> Dict[str, float]:
    """
    Extract recent grappling performance from FightStats (last N fights).
    
    Args:
        fight_history: DataFrame with fight history (sorted most recent first)
        fight_stats_by_fight_id: Dictionary mapping fight_id to FightStats object
        fighter_id: Fighter ID to extract stats for
        window: Number of recent fights to consider
        
    Returns:
        Dictionary of recent grappling features
    """
    if len(fight_history) == 0 or "fight_id" not in fight_history.columns:
        return {
            "recent_control_time_sec_last_3": 0.0,
            "recent_control_time_diff_last_3": 0.0,
        }
    
    recent = fight_history.head(window)
    fight_ids = [int(fid) for fid in recent["fight_id"].tolist() if pd.notna(fid)]
    
    if not fight_ids:
        return {
            "recent_control_time_sec_last_3": 0.0,
            "recent_control_time_diff_last_3": 0.0,
        }
    
    control_diffs = []
    my_control_times = []
    
    for _, row in recent.iterrows():
        fid = int(row.get("fight_id"))
        stats = fight_stats_by_fight_id.get(fid)
        if not stats:
            continue
        
        is_f1 = bool(row.get("is_fighter_1"))
        my_totals = stats.fighter_1_totals if is_f1 else stats.fighter_2_totals
        opp_totals = stats.fighter_2_totals if is_f1 else stats.fighter_1_totals
        
        if not my_totals or not opp_totals:
            continue
        
        # Control time (seconds)
        my_ct = parse_control_time_seconds(my_totals.get("control_time"))
        opp_ct = parse_control_time_seconds(opp_totals.get("control_time"))
        my_control_times.append(my_ct)
        control_diffs.append(my_ct - opp_ct)
    
    from .utils import safe_mean
    
    return {
        "recent_control_time_sec_last_3": safe_mean(my_control_times),
        "recent_control_time_diff_last_3": safe_mean(control_diffs),
    }

