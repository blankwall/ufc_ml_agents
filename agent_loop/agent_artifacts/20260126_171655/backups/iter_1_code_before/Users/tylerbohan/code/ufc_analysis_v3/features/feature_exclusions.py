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
    "age_x_years_since_last_win"


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
    "submission_avg_per_15min"
]

# Exact column names in the final training DataFrame to drop.
# These should match names in schema/feature_schema.json.
EXCLUDED_COLUMNS: List[str] = [
    # Example:
    # "power_striker_matchup",
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


