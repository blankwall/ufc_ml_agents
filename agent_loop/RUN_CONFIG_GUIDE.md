# Agent Loop Run Configuration Guide

## Quick Start

The run configuration lets you easily specify **what to optimize for** in each run without editing code.

## Simple Workflow

### 1. Choose or Create a Run Config

```bash
# Use the default run config
cp agent_loop/run_config.json agent_loop/my_run_config.json

# Or use an example
cp agent_loop/run_configs/example_underdog_focus.json agent_loop/my_run_config.json
```

### 2. Edit the Run Config

Open `agent_loop/my_run_config.json` and set your goals:

```json
{
  "run_goal": ["underdog"],           // What to focus on
  "priorities": {
    "prioritize": ["underdog"],       // Segments to optimize
    "avoid": ["favorites"]            // Segments to ignore
  },
  "success_criteria": {
    "min_roi_improvement": 0.01,      // Minimum ROI improvement
    "required_improvements": [        // Must meet these to keep changes
      {
        "segment": "underdog",
        "metric": "roi",
        "min_delta": 0.02
      }
    ]
  }
}
```

### 3. Run with Your Config

```python
from pathlib import Path
from agent_loop.orchestrator import LoopConfig, AgentLoop

cfg = LoopConfig(
    repo_root=Path("."),
    fight_url=None,  # Goal mode
    n_iters=10,
    run_config_path=Path("agent_loop/my_run_config.json"),  # Your config!
)

loop = AgentLoop(cfg)
loop.loop()
```

## Available Run Goals

### `underdog`
Focus on improving underdog predictions:
- Accuracy on underdogs
- ROI when betting on underdogs
- Calibration of underdog probabilities

### `top_25_pct`
Focus on high-confidence predictions:
- Accuracy on top 25% confidence fights
- ROI on high-confidence bets
- Precision of probability estimates

### `overall_accuracy`
Focus on overall model performance:
- Overall accuracy
- Brier score
- Log loss
- AUC

### `calibration`
Focus on probability calibration:
- Brier score improvement
- Log loss improvement
- Reliability diagram alignment

## Priority Options

### Segments to Prioritize
- `"underdog"` - Underdog predictions
- `"top_25_pct"` - Top 25% confidence predictions
- `"overall_accuracy"` - Overall accuracy
- `"favorites"` - Favorite predictions

### Segments to Avoid
- `"favorites"` - Don't optimize for favorites
- `"underdog"` - Don't optimize for underdogs
- Or leave empty to avoid nothing

## Success Criteria

### Available Metrics
- `"accuracy"` - Prediction accuracy
- `"roi"` - Return on investment
- `"brier_score"` - Brier score (lower is better)
- `"log_loss"` - Logarithmic loss (lower is better)

### Example Success Criteria

```json
{
  "success_criteria": {
    "min_roi_improvement": 0.01,  // Need at least 1% ROI improvement
    "required_improvements": [
      {
        "segment": "underdog",
        "metric": "accuracy",
        "min_delta": 0.005  // 0.5% accuracy improvement
      },
      {
        "segment": "underdog",
        "metric": "roi",
        "min_delta": 0.02  // 2% ROI improvement
      }
    ]
  }
}
```

## Example Run Configs

### Underdog Focus
```json
{
  "run_goal": ["underdog"],
  "priorities": {
    "prioritize": ["underdog"],
    "avoid": ["favorites"]
  }
}
```

### Top 25% Focus
```json
{
  "run_goal": ["top_25_pct"],
  "priorities": {
    "prioritize": ["top_25_pct"],
    "avoid": []
  }
}
```

### Balanced (Underdog + Top 25%)
```json
{
  "run_goal": ["underdog", "top_25_pct"],
  "priorities": {
    "prioritize": ["underdog", "top_25_pct"],
    "avoid": []
  }
}
```

### Calibration Focus
```json
{
  "run_goal": ["calibration"],
  "priorities": {
    "prioritize": ["overall_accuracy"],
    "avoid": []
  }
}
```

## Constraints

### Avoid Features
Prevent certain types of features from being added:

```json
{
  "constraints": {
    "avoid_features": [
      "quality differences",
      "ratio features with small denominators",
      "interaction terms"
    ]
  }
}
```

### Limit New Features
```json
{
  "constraints": {
    "max_new_features_per_iteration": 5,
    "prefer_existing_features_over_new": true
  }
}
```

## Tips

1. **Start narrow**: Focus on one goal at a time (e.g., just "underdog")
2. **Check results**: After each run, review what improved
3. **Adjust criteria**: If too many changes are kept, tighten success criteria
4. **Use examples**: Start from the example configs in `agent_loop/run_configs/`

## Troubleshooting

### Too many changes kept?
Tighten success criteria:
```json
{
  "success_criteria": {
    "min_roi_improvement": 0.02,  // Increase from 0.01
    "required_improvements": [
      {
        "segment": "underdog",
        "metric": "roi",
        "min_delta": 0.03  // Increase from 0.02
      }
    ]
  }
}
```

### Too few changes kept?
Loosen success criteria or reduce required improvements.

### Not focusing on the right segment?
Check your `prioritize` and `avoid` settings match your goal.
