# Feature Vector Builder

The `feature_vector_builder.py` module provides a centralized, schema-enforced way to build feature vectors for model prediction.

## Why This Exists

Before Phase 2, feature alignment was done manually in multiple places:
- `ensemble.py` manually aligned features
- `feature_pipeline.py` had manual alignment logic
- Each prediction script had its own alignment code

This led to:
- Feature drift (244 vs 246 features)
- Inconsistent feature ordering
- Silent bugs when features were missing

## How It Works

The builder:
1. Takes a feature dictionary (from `MatchupFeatureExtractor`)
2. Aligns it to the canonical schema order
3. Fills missing features with 0.0
4. Returns a numpy array in the exact order XGBoost expects

## Usage

### Basic Usage

```python
from features.matchup_features import MatchupFeatureExtractor
from features.feature_vector_builder import build_feature_vector
from models.xgboost_model import XGBoostModel
import numpy as np

# Extract features
extractor = MatchupFeatureExtractor(session)
features = extractor.extract_matchup_features(fighter_1_id, fighter_2_id)
features['is_title_fight'] = 1  # Add fight-specific features

# Build aligned vector
feature_vector = build_feature_vector(features)

# Predict
model = XGBoostModel()
model.load_model('xgboost_model')
prediction = model.model.predict_proba(np.array([feature_vector]))[0, 1]
```

### Convenience Function

```python
from features.feature_vector_builder import build_feature_vector_from_matchup
from models.xgboost_model import XGBoostModel
import numpy as np

# One-liner: extract + align
feature_vector = build_feature_vector_from_matchup(
    fighter_1_id,
    fighter_2_id,
    session,
    is_title_fight=True
)

# Predict
model = XGBoostModel()
model.load_model('xgboost_model')
prediction = model.model.predict_proba(np.array([feature_vector]))[0, 1]
```

### Batch Processing

```python
from features.feature_vector_builder import build_feature_vectors_batch

# Extract features for multiple fights
feature_dicts = []
for fight in fights:
    features = extractor.extract_matchup_features(fight.fighter_1_id, fight.fighter_2_id)
    feature_dicts.append(features)

# Build all vectors at once
feature_matrix = build_feature_vectors_batch(feature_dicts)

# Predict for all fights
predictions = model.model.predict_proba(feature_matrix)
```

## Guarantees

✅ **100% identical order** - Features always in schema order  
✅ **100% identical count** - Always exactly 246 features (or whatever schema says)  
✅ **0% feature drift** - Schema is the single source of truth  
✅ **XGBoost compatibility** - Monotone constraints line up perfectly  
✅ **Consistent across scripts** - Excel export = CLI prediction = training run

## Integration Points

This should be used by:
- `xgboost_predict.py` (Phase 6)
- `scripts/export_predictions_to_excel.py` (Phase 6)
- Any API endpoints
- Model evaluation scripts
- Training dataset creation (optional, but recommended)

## Testing

Test the builder:
```bash
python -m features.feature_vector_builder "Jon Jones" "Daniel Cormier"
```

This will:
1. Find the fighters
2. Extract features
3. Build aligned vector
4. Show schema info

