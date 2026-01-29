# Agent Loop Configuration Guide

## Overview

The agent_loop now uses a centralized configuration file (`agent_loop_config.json`) to make it easier to adjust parameters without modifying code.

## Configuration File Location

The default configuration file is located at:
```
agent_loop/agent_loop_config.json
```

## Configuration Structure

The configuration is organized into logical sections:

### Agent Settings
- `model`: Claude model to use for agents
- `agent_cmd`: Command to run the agent CLI
- `timeout_seconds`: Agent execution timeout
- `permission_mode`: Permission mode for agent file operations

### Model Pipeline Settings
- `holdout_from_year`: Year to hold out data from
- `baseline_json`: Path to baseline model configuration
- `train_model_name_prefix`: Prefix for trained model names
- `xgboost_predict_model_name`: Model name for xgboost predictions
- `n_estimators`: Number of estimators for XGBoost
- `max_depth`: Maximum depth for XGBoost trees
- `learning_rate`: Learning rate for training
- `subsample`: Subsample ratio for training
- `colsample_bytree`: Column subsample ratio

### Evaluation Settings
- `eval_min_year`: Minimum year for evaluation
- `odds_path`: Path to odds data file
- `odds_date_tolerance_days`: Tolerance for matching odds by date
- `symmetric`: Whether to use symmetric evaluation

### Introspection Settings
- `non_feature_columns`: List of columns to exclude from feature counts
- `low_importance_threshold`: Threshold for low-importance features
- `top_n_features`: Number of top features to display
- `concentration_thresholds`: Thresholds for feature concentration analysis
- `feature_count_thresholds`: Thresholds for feature count analysis

### Paths
All commonly used paths throughout the agent loop

### Thresholds
- `roi_thresholds`: Thresholds for ROI analysis
- `accuracy_thresholds`: Thresholds for accuracy analysis

## Usage

### Basic Usage

```python
from agent_loop.config import load_config
from pathlib import Path

# Load configuration
cfg = load_config(repo_root=Path("."))

# Access configuration values
model = cfg.agent.model
n_estimators = cfg.model_pipeline.n_estimators
```

### Using with LoopConfig

```python
from agent_loop.orchestrator import LoopConfig, AgentLoop
from pathlib import Path

# LoopConfig automatically loads the config
loop_cfg = LoopConfig(
    repo_root=Path("."),
    fight_url="https://example.com/fight",
    n_iters=5,
)

# The loop_cfg will use defaults from agent_loop_config.json
# unless you specify overrides
loop = AgentLoop(loop_cfg)
loop.loop()
```

### Overriding Config Values

You can override specific config values when creating LoopConfig:

```python
# Override specific values
loop_cfg = LoopConfig(
    repo_root=Path("."),
    fight_url="https://example.com/fight",
    n_iters=5,
    model="claude-opus-4-5-20251101",  # Override agent model
    holdout_from_year=2024,  # Override holdout year
    n_iters=10,  # Override iterations
)
```

## Common Adjustments

### Change Agent Model
Edit `agent_loop_config.json`:
```json
{
  "agent": {
    "model": "claude-opus-4-5-20251101"
  }
}
```

### Adjust Model Training Parameters
Edit `agent_loop_config.json`:
```json
{
  "model_pipeline": {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1
  }
}
```

### Adjust Evaluation Thresholds
Edit `agent_loop_config.json`:
```json
{
  "roi_thresholds": {
    "very_negative": -0.10,
    "positive": 0.05
  }
}
```

## Benefits of Centralized Configuration

1. **Easy adjustments**: Change parameters without editing code
2. **Version control**: Track configuration changes in git
3. **Documentation**: Configuration is self-documenting
4. **Reusability**: Share configurations across runs
5. **Testing**: Easy to test different parameter combinations
