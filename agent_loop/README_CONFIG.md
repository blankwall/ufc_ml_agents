# Agent Loop Configuration

## Two Types of Configuration

### 1. Global Configuration (`agent_loop_config.json`)
**Purpose**: Base settings that rarely change
- Agent model and timeout
- Model training parameters (n_estimators, max_depth, etc.)
- Paths and file locations
- Evaluation thresholds

**When to edit**: When you want to change fundamental behavior

### 2. Run Configuration (`run_config.json` or custom)
**Purpose**: Per-run goal settings
- What to optimize for (underdog, top 25%, etc.)
- Priorities and constraints
- Success criteria for keeping changes

**When to edit**: Before each run to specify your goal

## Quick Start

### Option 1: Use Default Run Config
```python
from pathlib import Path
from agent_loop.orchestrator import LoopConfig, AgentLoop

cfg = LoopConfig(
    repo_root=Path("."),
    fight_url=None,
    n_iters=10,
    # Uses default agent_loop/run_config.json
)

AgentLoop(cfg).loop()
```

### Option 2: Specify Custom Run Config
```python
from pathlib import Path
from agent_loop.orchestrator import LoopConfig, AgentLoop

cfg = LoopConfig(
    repo_root=Path("."),
    fight_url=None,
    n_iters=10,
    run_config_path=Path("agent_loop/run_configs/example_underdog_focus.json"),
)

AgentLoop(cfg).loop()
```

## Example Run Configs

See `agent_loop/run_configs/` for examples:
- `example_underdog_focus.json` - Optimize underdog predictions
- `example_top25_focus.json` - Optimize high-confidence predictions
- `example_balanced.json` - Balance multiple goals
- `example_calibration.json` - Improve probability calibration

## Workflow

1. **Copy an example** or use the template:
   ```bash
   cp agent_loop/run_config.template.json agent_loop/my_run.json
   ```

2. **Edit your run config** to set goals and success criteria

3. **Run with your config**:
   ```python
   cfg = LoopConfig(
       repo_root=Path("."),
       fight_url=None,
       n_iters=10,
       run_config_path=Path("agent_loop/my_run.json"),
   )
   AgentLoop(cfg).loop()
   ```

## Documentation

- `CONFIG_GUIDE.md` - Global configuration reference
- `RUN_CONFIG_GUIDE.md` - Run configuration reference with examples
