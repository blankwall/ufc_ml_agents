"""
Feature Exclusions
------------------

Central place to **drop individual features/columns** from the final
training dataset and model input.

Usage:
  - Add base names (e.g. "early_finish_advantage") to EXCLUDED_BASE_FEATURES
    to drop all related columns:
      * f1_early_finish_advantage
      * f2_early_finish_advantage
      * early_finish_advantage_diff
  - Or add exact column names (matching schema/feature_schema.json) to
    EXCLUDED_COLUMNS to drop only those.

After changing this file:
  1. Re-create the training dataset: `python -m features.feature_pipeline --create --feature-set full`
  2. Retrain the model: The schema will be automatically exported after training
  3. (Optional) Export schema from dataset without training: 
     `python scripts/export_schema_from_dataset.py --data-path data/processed/training_data.csv`
"""

from typing import Iterable, List, Set

# Base logical feature names – we will drop any matchup columns derived from
# these (f1_*, f2_*, *_diff).
EXCLUDED_BASE_FEATURES: List[str] = [
    # Example:
    # "early_finish_advantage",
    "years_since_last_win",
    "age_x_years_since_last_win",
    "age_x_recent_sig_strike_diff_last_3",

    # Iteration 1: Remove low-importance binary features to reduce overfitting
    "stance_orthodox",
    "stance_southpaw",
    "stance_switch",
    "both_grapplers",
    "both_strikers",
    "both_finishers",
    "draws",
    "age_in_prime",
    "age_past_prime",

# Striking stats
    # "sig_strikes_landed_per_min",
    "striking_accuracy",
    "striking_defense",
    # "striking_differential",
    "defensive_efficiency",
    # "striking_volume_control",
    "distance_accuracy_last_3",
    "clinch_accuracy_last_3",
    "ground_output_per_min_last_3",
    "leg_strike_rate_last_3",
    "knockdowns_last_3",
    "striking_accuracy_last_3",
    "sig_strikes_landed_per_min_last_3",
    "head_strike_rate_last_3",
    "body_strike_rate_last_3",
    "ground_strike_rate_last_3",
    "distance_strike_rate_last_3",
    "distance_accuracy_lifetime",
    "clinch_accuracy_lifetime",
    "ground_output_per_min_lifetime",
    "leg_strike_rate_lifetime",
    "knockdowns_lifetime",
    "head_strike_rate_lifetime",
    "body_strike_rate_lifetime",
    "ground_strike_rate_lifetime",
    "distance_strike_rate_lifetime",
    # "sig_strikes_landed_per_min_lifetime",
    "striking_accuracy_lifetime",
    
    # Grappling stats
    "takedown_avg_per_15min",
    "takedown_accuracy",
    "takedown_defense",
    "submission_avg_per_15min",

    # Iteration 1: Remove near-zero importance features (importance < 3)
    # These 27 base features (generating 32+ columns) contribute minimal signal
    # and create overfitting noise. Removing them targets both top_25_pct and
    # underdog performance by cleaning the feature space.
    "common_opponent_performance",
    "current_loss_streak",
    "current_win_streak",
    "decision_losses_since_last_win",
    "early_finish_rate_last_3",
    "early_finish_rate_last_5",
    "fights_since_last_win",
    "finish_losses_since_last_win",
    "finish_rate",
    "finish_rate_last_3",
    "finish_rate_last_3_years",
    "finish_rate_last_5",
    "first_round_ko_rate",
    "ko_rate_last_5",
    "losses_last_3_years",
    "losses_since_last_win",
    "opponent_quality_score",
    "power_striker_matchup",
    "recent_finish_loss_last_fight",
    "recent_finish_losses_last_2",
    "recent_form",
    "recent_knockdown_diff_last_3",
    "striker_vs_grappler",
    "weight_lbs",
    "win_rate_last_3",
    "win_rate_last_5",
    "wins_last_3_years",

    # Iteration 2: Remove redundant opponent quality and control time features
    # Keep only adjusted version of time_decayed_win_rate (opponent_quality_adjusted is more predictive)
    "time_decayed_win_rate",
    # Keep avg_beaten_opponent_win_rate (more specific than avg_opponent_win_rate)
    "avg_opponent_win_rate",
    # Keep avg_beaten_opponent_total_fights (more specific than avg_opponent_total_fights)
    "avg_opponent_total_fights",
    # Keep recent_control_time_sec (more explicit than recent_control_time_diff)
    "recent_control_time_diff_last_3",

    # Iteration 3: Remove redundant KO rate time-window features
    # Keep main ko_rate, remove last_3 and last_5 variants to reduce noise
    "ko_rate_last_3",
    "ko_rate_last_5",

    # Iteration 4: Remove physical attributes that duplicate reach advantage
    # Weight classes normalize weight differences, and reach_advantage already captures
    # the relevant physical dimension. Height adds noise without unique predictive signal.
    "height_cm"

    # Iteration 5 (agent_loop): Remove 3 lowest importance features (importance = 1.0)
    # These specific columns have minimal predictive value and add noise to the model.
    # Removing them should improve underdog prediction clarity by eliminating irrelevant signal.
    "long_layoff_over_1yr",
    "recent_losses_last_2",
    "activity_rate",

    # Iteration 6 (agent_loop): Remove entire youth_form_score feature family
    # All three youth_form_score features have very low importance (3.0 each).
    # This concept is not adding value and may be creating noise that hurts underdog prediction,
    # particularly for younger fighters who are often underdogs.
    "youth_form_score",

    # Iteration 7 (agent_loop): REVERTED - Restoring baseline model
    # The removed features contained subtle signals important for underdog predictions.
    # Reverting to restore underdog performance (ROI dropped 16% points after removal).
    # "has_ever_won",
    # "long_layoff_over_1yr",
    # "fights_in_last_year",
    # "age_x_recent_vs_career_decline",
    # "underdog_specialization_advantage",
]

# Exact column names in the final training DataFrame to drop.
# These should match names in schema/feature_schema.json.
EXCLUDED_COLUMNS: List[str] = [
    # Example:
    # "power_striker_matchup",

    # Iteration 5 (agent_loop): Remove specific low-importance columns
    # - f1_long_layoff_over_1yr: binary flag for layoffs > 1 year (importance 1.0)
    # - f2_recent_losses_last_2: count of losses in last 2 fights (importance 1.0)
    # - f2_activity_rate: fights per year metric (importance 1.0)
    "f1_long_layoff_over_1yr",
    "f2_recent_losses_last_2",
    "f2_activity_rate",

    # Iteration 6 (agent_loop): Consolidate redundant striking differential features
    # Remove individual fighter versions (f1_striking_differential: 39.0, f2_striking_differential: 35.0)
    # while keeping striking_differential (38.0) to reduce redundancy and noise without
    # removing low-importance features that support ensemble performance.
    "f1_striking_differential",
    "f2_striking_differential",

    # Iteration 7 (agent_loop): Consolidate redundant age x days since last fight features
    # Remove individual fighter versions (f1_age_x_days_since_last_fight: 25.0, f2_age_x_days_since_last_fight: 29.0)
    # while keeping age_x_days_since_last_fight_diff (24.0) to reduce redundancy without
    # harming ensemble performance. This targets underdog prediction by reducing noise.
    "f1_age_x_days_since_last_fight",
    "f2_age_x_days_since_last_fight",

    # Iteration 8 (agent_loop): REVERTED - Restoring baseline model
    # The removed binary stance matchup feature contained signals important for underdog predictions.
    # Reverting to restore underdog performance.
    # "orthodox_vs_southpaw",

    # Iteration 9 (agent_loop): Remove lowest importance stance and common opponent features
    # Diagnostics show these features have near-zero importance (southpaw_vs_southpaw: 1, orthodox_vs_orthodox: 2, num_common_opponents: 1)
    # Removing them reduces overfitting noise and should improve Top 25% validation accuracy.
    "southpaw_vs_southpaw",
    "orthodox_vs_orthodox",
    "num_common_opponents",
]


def get_columns_to_exclude(all_columns: Iterable[str]) -> List[str]:
    """
    Given a list of DataFrame columns, return the subset that should be dropped.
    
    This supports:
      - Exact name matches from EXCLUDED_COLUMNS
      - Derived names from EXCLUDED_BASE_FEATURES:
          f1_<base>, f2_<base>, <base>_diff
    """
    cols_set: Set[str] = set(all_columns)
    to_drop: Set[str] = set()
    
    # Exact column exclusions
    for col in EXCLUDED_COLUMNS:
        if col in cols_set:
            to_drop.add(col)
    
    # Base feature exclusions (auto-expand to f1_*, f2_*, *_diff)
    for base in EXCLUDED_BASE_FEATURES:
        patterns = [
            base,
            f"f1_{base}",
            f"f2_{base}",
            f"{base}_diff",
        ]
        for p in patterns:
            if p in cols_set:
                to_drop.add(p)
    
    return sorted(to_drop)


def print_exclusion_summary() -> None:
    """Print a summary of configured exclusions before training."""
    print("\n" + "=" * 80)
    print("FEATURE EXCLUSION SUMMARY")
    print("=" * 80)
    
    if EXCLUDED_BASE_FEATURES:
        print(f"\nExcluding {len(EXCLUDED_BASE_FEATURES)} base features (will drop f1_*, f2_*, *_diff variants):")
        for base in EXCLUDED_BASE_FEATURES:
            print(f"  • {base}")
            print(f"    → will exclude: f1_{base}, f2_{base}, {base}_diff")
    else:
        print("\nNo base features excluded.")
    
    if EXCLUDED_COLUMNS:
        print(f"\nExcluding {len(EXCLUDED_COLUMNS)} exact column names:")
        for col in EXCLUDED_COLUMNS:
            print(f"  • {col}")
    else:
        print("\nNo exact columns excluded.")
    
    if not EXCLUDED_BASE_FEATURES and not EXCLUDED_COLUMNS:
        print("\n⚠️  No exclusions configured - all features will be used.")
    
    print("=" * 80 + "\n")


