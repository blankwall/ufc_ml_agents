## Agent Loop

This directory contains an **agentic optimization loop** that orchestrates Cursor CLI agents
to iteratively improve feature engineering + monotone constraints, then retrain/evaluate models.

### High-level flow

1. **planning agent**
   - Ingest fight into DB (via fight-details URL → event → DB)
   - Run `xgboost_predict.py` for the fight
   - Produce a plan JSON at `agent_loop/agent_artifacts/<run_ts>/plan.json`

2. **Loop N times**
   - **feature_creator agent**: makes **one logical change** (features/schema/constraints)
   - **feature_builder (python)**: rebuild dataset → train model → evaluate model
   - **tester agent**: decides keep/revert and updates plan for next iteration

3. **summarizer agent**
   - Produces `report.html` summarizing all iterations

### Running

```bash
python3 -m agent_loop.run \
  --fight-url "http://ufcstats.com/fight-details/fa4b3f5ce8055921" \
  --n 3 \
  --model gpt-5 \
  --verbose
```

### Notes
- Uses the **local Cursor CLI** agent command (`agent`).
- Uses `--print --output-format json` for non-interactive runs.
- Creates backups before mutating key files and before rebuilding long-running artifacts.


