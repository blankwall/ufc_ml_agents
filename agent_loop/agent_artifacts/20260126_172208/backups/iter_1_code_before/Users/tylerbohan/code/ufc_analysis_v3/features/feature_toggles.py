"""
Feature Toggles
---------------

Central place to enable/disable **feature groups** used by the registry.

Edit `ENABLED_FEATURE_GROUPS` below and comment out any groups you want to
temporarily disable, then regenerate training data and retrain.

This controls:
- `features.fighter_features.FighterFeatureExtractor` default feature set
- `features.matchup_features.MatchupFeatureExtractor` (via fighter features)
- CLI scripts that use the "custom" feature set option
"""

from typing import List

# These names must match the keys used in `FeatureRegistry.get_feature_function`.
#
# You can comment out any line to disable that entire feature group
# without touching the rest of the code.
#
# Example:
#   To disable opponent-quality features:
#     - comment out "opponent_quality"
#
#   To disable interaction features like power_striker:
#     - comment out "power_striker"
#

ENABLED_FEATURE_GROUPS: List[str] = [
    # --- Physical / core stats ---
    "physical",
    "striking",
    "grappling",

    # --- Experiential / career history ---
    "career_stats",
    "fight_history",
    "early_finishing",
    "round_3",

    # --- Time-based / momentum, decline, time-decayed win rate ---
    "rolling_stats",
    "momentum",
    "decline",
    "recent_damage",
    "time_decayed",
    "time_decayed_adj_opp_quality",

    # --- Opponent quality / strength of schedule ---
    "opponent_quality",

    # --- Interaction / composite scores ---
    "age_interactions",
    "youth_form",
    "prospect_momentum",
    "early_finish_advantage",
    "power_striker",
    "age_weighted_recent_damage",
    "durability_collapse",

    # --- Recent detailed stats (FightStats-based) ---
    "recent_striking",
    "recent_grappling",
]

# Alias used by scripts when `--feature-set custom` is selected.
DEFAULT_FEATURE_SET = ENABLED_FEATURE_GROUPS




