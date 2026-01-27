## Agent Loop

This directory contains an **agentic optimization loop** that orchestrates Claude Code CLI agents
to iteratively improve feature engineering + monotone constraints, then retrain/evaluate models.

### High-level flow

1. **planning agent**
   - **Fight mode**: Scrape fight details from UFCStats (without adding to DB) + Run `xgboost_predict.py`
   - **Goal mode**: Use provided goal text directly
   - Produce a plan JSON at `agent_loop/agent_artifacts/<run_ts>/plan.json`

2. **Loop N times**
   - **feature_creator agent**: makes **one logical change** (features/schema/constraints)
   - **feature_builder (python)**: rebuild dataset → train model → evaluate model
   - **tester agent**: decides keep/revert and updates plan for next iteration

3. **summarizer agent**
   - Produces `report.html` summarizing all iterations

### Important: Data Leakage Prevention

In **fight mode**, the fight is **NOT added to the database**. This prevents data leakage:
- The fight remains out-of-sample for the entire optimization loop
- Only fighter historical data (from prior fights) is used
- The model is trained on existing data, then evaluated on how well it would have predicted the analyzed fight
- This ensures genuine model improvement rather than overfitting to a specific fight

### Running

**Fight mode** (analyze a specific fight):
```bash
python3 -m agent_loop.run \
  --fight-url "http://ufcstats.com/fight-details/fa4b3f5ce8055921" \
  --n 3 \
  --model claude-sonnet-4-5-20250929 \
  --verbose
```

**Goal mode** (generic optimization):
```bash
python3 -m agent_loop.run \
  --goal "Improve Top 25% and underdog performance; avoid new opponent-quality-difference features." \
  --n 3 \
  --model claude-sonnet-4-5-20250929
```

**Goal from file**:
```bash
python3 -m agent_loop.run --goal-file goals.txt --n 3 --model claude-sonnet-4-5-20250929
```

### Resume vs Fork

- **Resume**: continue in the same run directory (appends new iterations to the same history).

```bash
python3 -m agent_loop.run --fight-url "<fight>" --resume-run 20260126_172555 --n 3
```

- **Fork**: branch into a new run directory seeded from the latest plan/context of an existing run.

```bash
python3 -m agent_loop.run --fight-url "<fight>" --fork-run 20260126_172555 --n 3
```

### Manual Mode (--manual)

If your Claude CLI binary is not working (e.g., macOS sandboxing issues), use `--manual` mode:

```bash
python3 -m agent_loop.run --goal "your goal text" --n 3 --manual --verbose
```

In manual mode:
1. Loop writes prompt files to `agent_loop/agent_artifacts/<run_ts>/logs/prompt_*.txt`
2. Loop pauses and waits for you to create each expected artifact (e.g., `change.json`, `decision.json`)
3. You open the prompt file in Claude Code IDE, run the agent, and let it write artifacts
4. Once the expected artifact appears, press Enter to continue to the next step

This lets you control agent execution directly in the IDE while the orchestrator handles:
- Feature building and model training
- Keeping/reverting code changes
- Updating plans and history between iterations

### Notes
- Uses the **local Claude Code CLI** (`claude` command)
- Uses `-p` (print) with `--output-format json` for non-interactive runs
- Creates backups before mutating key files and before rebuilding long-running artifacts
- Fighters should already exist in the database from their prior fights before using fight mode

