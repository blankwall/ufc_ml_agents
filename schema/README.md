# Feature Schema

This directory contains the **canonical feature schema** - the master contract between training, prediction, Excel export, and API usage.

## Overview

The feature schema (`feature_schema.json`) defines:
- The exact list of features used by the model
- The canonical order of features
- The schema version

This ensures 100% consistency across:
- Model training
- Predictions
- Excel exports
- API endpoints
- Feature vector builders

## Schema Format

```json
{
  "version": "1.0.0",
  "num_features": 244,
  "features": [
    "f1_height_cm",
    "f1_weight_lbs",
    ...
  ]
}
```

## Generating the Schema

### Method 1: From a Trained Model

After training a model, export the schema:

```python
from models.xgboost_model import XGBoostModel

model = XGBoostModel()
model.load_model("xgboost_model")
model.export_feature_schema(version="1.0.0")
```

Or automatically when saving:

```python
model.save_model("xgboost_model", export_schema=True)
```

### Method 2: From Feature Pipeline

```python
from features.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline()
pipeline.load_pipeline()
pipeline.export_feature_schema(version="1.0.0")
```

### Method 3: Using the Script

```bash
python schema/generate_schema.py --source model --version 1.0.0
```

Options:
- `--source`: `model` (JSON), `pipeline` (PKL), or `booster` (XGBoost)
- `--model-name`: Name of saved model (default: `xgboost_model`)
- `--version`: Schema version (default: `1.0.0`)
- `--output`: Output path (default: `schema/feature_schema.json`)
- `--verify`: Verify schema matches model

## Using the Schema

### Load the Schema

```python
from schema import load_schema, get_feature_list

# Load full schema
schema = load_schema()
print(f"Version: {schema['version']}")
print(f"Features: {schema['num_features']}")

# Get just the feature list
features = get_feature_list()
```

### Validate Feature Vectors

```python
from schema import validate_feature_vector

feature_dict = {"f1_height_cm": 180, "f1_weight_lbs": 170, ...}
is_valid, missing = validate_feature_vector(feature_dict, strict=True)

if not is_valid:
    print(f"Missing features: {missing}")
```

### Align Feature Vectors

```python
from schema import align_feature_vector
import numpy as np

feature_dict = {"f1_height_cm": 180, "f1_weight_lbs": 170, ...}
aligned = align_feature_vector(feature_dict, fill_missing=0.0)
feature_array = np.array(aligned, dtype=float)
```

## Best Practices

1. **Freeze the schema** after training a good model
2. **Version the schema** when making changes
3. **Validate feature vectors** before predictions
4. **Align feature vectors** to ensure correct order
5. **Never modify the schema** without retraining the model

## Schema Versioning

When you retrain the model with new features:
1. Update the version (e.g., `1.0.0` → `1.1.0`)
2. Regenerate the schema
3. Update all consumers (prediction scripts, Excel exports, etc.)

## Integration Points

The schema should be used by:
- `features/feature_vector_builder.py` (Phase 2)
- `xgboost_predict.py`
- `scripts/export_predictions_to_excel.py`
- Any API endpoints
- Model evaluation scripts

