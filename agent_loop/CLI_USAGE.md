# CLI Usage for Agent Loop

## Basic Usage

### Without Run Config (uses defaults)
```bash
python -m agent_loop.run --goal "Improve the model" --n 5
```

### With Run Config (recommended!)
```bash
python -m agent_loop.run \
  --goal "Improve the model" \
  --n 5 \
  --run-config agent_loop/run_config_underdog_focus.json
```

## Common Examples

### Underdog Focus
```bash
python -m agent_loop.run \
  --goal "Improve underdog predictions" \
  --n 10 \
  --run-config agent_loop/run_config_underdog_focus.json
```

### Top 25% Focus
```bash
python -m agent_loop.run \
  --goal "Improve high-confidence predictions" \
  --n 10 \
  --run-config agent_loop/run_configs/example_top25_focus.json
```

### Balanced (Underdog + Top 25%)
```bash
python -m agent_loop.run \
  --goal "Improve overall performance" \
  --n 10 \
  --run-config agent_loop/run_configs/example_balanced.json
```

## Available Run Configs

- `agent_loop/run_config_underdog_focus.json` - Underdog focus (recommended)
- `agent_loop/run_configs/example_underdog_focus.json` - Underdog example
- `agent_loop/run_configs/example_top25_focus.json` - Top 25% example
- `agent_loop/run_configs/example_balanced.json` - Balanced example
- `agent_loop/run_configs/example_calibration.json` - Calibration focus

## Other Useful Flags

```bash
# Verbose output
python -m agent_loop.run --goal "Improve model" --n 5 --run-config agent_loop/run_config_underdog_focus.json --verbose

# Resume from previous run
python -m agent_loop.run --goal "Improve model" --resume-run 20260126_172555 --run-config agent_loop/run_config_underdog_focus.json

# Fork from previous run
python -m agent_loop.run --goal "Improve model" --fork-run 20260126_172555 --run-config agent_loop/run_config_underdog_focus.json

# Use different model
python -m agent_loop.run --goal "Improve model" --n 5 --run-config agent_loop/run_config_underdog_focus.json --model claude-opus-4-5-20251101
```

## Quick Start

1. Edit your run config (or use the underdog one):
   ```bash
   cp agent_loop/run_config_underdog_focus.json agent_loop/my_run.json
   # Edit agent_loop/my_run.json if needed
   ```

2. Run with your config:
   ```bash
   python -m agent_loop.run --goal "Improve underdog performance" --n 10 --run-config agent_loop/my_run.json
   ```
