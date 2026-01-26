# Feature Engineering System - Refactored

## Overview

The feature engineering system has been completely refactored into a **modular, maintainable architecture**. Each feature category is now in its own module, making it easy to:

- **Toggle features on/off** by selecting feature sets
- **Add new features** without modifying existing code
- **Test features independently** as pure functions
- **Understand feature dependencies** through clear organization

## Architecture

### Module Structure

```
features/
├── __init__.py              # Package exports
├── utils.py                 # Shared utility functions
├── physical.py              # Age, height, weight, reach, stance
├── striking.py              # Striking stats, accuracy, defense
├── grappling.py             # Takedowns, submissions, control time
├── experiential.py          # Career stats, fight history, finishing rates
├── time_based.py            # Rolling stats, momentum, decline, time-decayed
├── opponent_quality.py      # Strength of schedule, opponent win rates
├── registry.py              # Feature registry and orchestrator
├── fighter_features.py      # FighterFeatureExtractor (backward compatible)
├── matchup_features.py      # MatchupFeatureExtractor (backward compatible)
└── feature_pipeline.py      # FeaturePipeline (backward compatible)
```

### Key Components

#### 1. Feature Modules (`physical.py`, `striking.py`, etc.)

Each module contains **pure functions** that extract features from a specific domain:

```python
from features.physical import extract_physical_features
from features.striking import extract_striking_features

# Pure functions - easy to test
features = extract_physical_features(fighter)
striking = extract_striking_features(fighter)
```

**Principles:**
- Pure functions (no side effects)
- Clear, documented inputs/outputs
- Consistent naming (snake_case)
- Always return numeric values
- Handle missing data gracefully

#### 2. Feature Registry (`registry.py`)

The `FeatureRegistry` class maps feature names to extraction functions and provides predefined feature sets:

```python
from features.registry import FeatureRegistry

# Predefined feature sets
FEATURE_SET_BASE = [
    "physical", "striking", "grappling", 
    "career_stats", "fight_history", "rolling_stats", ...
]

FEATURE_SET_ADVANCED = FEATURE_SET_BASE + [
    "opponent_quality", "age_interactions", ...
]

FEATURE_SET_FULL = FEATURE_SET_ADVANCED + [
    "recent_striking", "recent_grappling"
]
```

#### 3. Feature Builder (`registry.py`)

The `FeatureBuilder` class orchestrates feature extraction:

```python
from features.registry import FeatureBuilder

builder = FeatureBuilder(session, rolling_windows=[3, 5])

# Extract features using a feature set
features = builder.build_features(
    fighter_id=123,
    feature_set=FeatureRegistry.FEATURE_SET_FULL
)
```

## Usage Examples

### Basic Usage (Backward Compatible)

The old interface still works:

```python
from features.fighter_features import FighterFeatureExtractor

extractor = FighterFeatureExtractor(session)
features = extractor.extract_features(fighter_id=123)
```

### Using Feature Sets

```python
from features.registry import FeatureRegistry, FeatureBuilder

builder = FeatureBuilder(session)

# Use predefined feature set
features = builder.build_features(
    fighter_id=123,
    feature_set=FeatureRegistry.FEATURE_SET_BASE
)

# Or create custom feature set
custom_set = [
    "physical",
    "striking",
    "rolling_stats",
    "momentum"
]
features = builder.build_features(
    fighter_id=123,
    feature_set=custom_set
)
```

### Creating Training Dataset with Feature Set

```python
from features.feature_pipeline import FeaturePipeline
from features.registry import FeatureRegistry

pipeline = FeaturePipeline()

# Create dataset with specific feature set
df = pipeline.create_dataset(
    output_path='data/processed/training_data.csv',
    feature_set=FeatureRegistry.FEATURE_SET_ADVANCED
)
```

### Direct Module Usage

You can also use feature modules directly:

```python
from features.physical import extract_physical_features
from features.striking import extract_striking_features
from features.time_based import extract_rolling_stats

# Extract individual feature groups
physical = extract_physical_features(fighter)
striking = extract_striking_features(fighter)
rolling = extract_rolling_stats(fight_history, rolling_windows=[3, 5])

# Combine
all_features = {**physical, **striking, **rolling}
```

## Feature Sets

### FEATURE_SET_BASE
Core features for basic predictions:
- Physical attributes
- Striking and grappling stats
- Career statistics
- Fight history
- Rolling statistics (last 3, 5 fights)
- Momentum and activity

### FEATURE_SET_ADVANCED
Base features plus:
- Opponent quality metrics
- Age interactions
- Youth form score

### FEATURE_SET_FULL
Advanced features plus:
- Recent striking performance (from FightStats)
- Recent grappling performance (from FightStats)

## Adding New Features

### Step 1: Add to Appropriate Module

```python
# features/striking.py

def extract_new_striking_feature(fighter: Fighter) -> Dict[str, float]:
    """
    Extract new striking feature.
    
    Args:
        fighter: Fighter database object
        
    Returns:
        Dictionary with new feature
    """
    return {
        "new_striking_metric": float(calculated_value)
    }
```

### Step 2: Register in FeatureRegistry

```python
# features/registry.py

class FeatureRegistry:
    # Add to appropriate feature set
    FEATURE_SET_STRIKING = [
        "striking",
        "new_striking_feature",  # Add here
    ]
    
    @staticmethod
    def _extract_new_striking_feature(context: Dict) -> Dict[str, float]:
        """Extract new striking feature"""
        fighter = context["fighter"]
        return extract_new_striking_feature(fighter)
    
    @classmethod
    def get_feature_function(cls, feature_name: str):
        feature_map = {
            # ... existing features
            "new_striking_feature": cls._extract_new_striking_feature,
        }
        return feature_map.get(feature_name)
```

### Step 3: Use in Feature Sets

```python
# Update feature sets to include new feature
FEATURE_SET_ADVANCED = (
    FEATURE_SET_BASE +
    ["new_striking_feature"] +  # Add here
    FEATURE_SET_OPPONENT_QUALITY +
    ...
)
```

## Feature Naming Conventions

All features follow consistent naming:

- **snake_case** for all feature names
- **Descriptive names** that indicate what they measure
- **Suffixes** for context:
  - `_last_3`, `_last_5` for rolling windows
  - `_diff` for differential features (matchup level)
  - `_rate` for rates/proportions
  - `_per_min` or `_per_15min` for per-minute metrics

## Testing Features

Since features are pure functions, they're easy to test:

```python
def test_extract_physical_features():
    from features.physical import extract_physical_features
    from database.schema import Fighter
    
    fighter = Fighter(age=30, height_cm=180, weight_lbs=170)
    features = extract_physical_features(fighter)
    
    assert features["age"] == 30.0
    assert features["height_cm"] == 180.0
    assert features["age_in_prime"] == 1.0
```

## Migration Guide

### Old Code
```python
extractor = FighterFeatureExtractor(session)
features = extractor.extract_features(fighter_id)
```

### New Code (Same Interface)
```python
# Still works! Backward compatible
extractor = FighterFeatureExtractor(session)
features = extractor.extract_features(fighter_id)
```

### New Code (With Feature Sets)
```python
# New: Specify feature set
extractor = FighterFeatureExtractor(session)
features = extractor.extract_features(
    fighter_id,
    feature_set=FeatureRegistry.FEATURE_SET_BASE
)
```

## Benefits of Refactoring

1. **Modularity**: Each feature category is isolated
2. **Testability**: Pure functions are easy to unit test
3. **Flexibility**: Easy to toggle features on/off
4. **Maintainability**: Clear organization and dependencies
5. **Extensibility**: Add new features without touching existing code
6. **Documentation**: Each module is self-contained and documented
7. **Consistency**: Enforced naming conventions and patterns

## Next Steps

- [ ] Add unit tests for each feature module
- [ ] Create feature validation scripts
- [ ] Document feature dependencies
- [ ] Add feature importance analysis tools
- [ ] Create feature comparison utilities

