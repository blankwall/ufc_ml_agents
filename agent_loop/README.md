# Agent Loop

An agentic optimization system that iteratively improves ML feature engineering through role-separated micro-agents, deterministic evaluation boundaries, and durable artifact-based state.

The system treats agents as short-lived task executors operating against shared structured state — not long-running conversational entities. Each iteration proposes a scoped change, validates it deterministically, and either commits or rolls it back.

State lives in artifacts. Evaluation is owned by deterministic code. Agents do not judge themselves.

---

## Quick Start

```bash
# Goal mode (generic optimization)
python -m agent_loop.run \
  --goal "Improve Top 25% and underdog performance" \
  --n 5 \
  --run-config agent_loop/run_config_underdog_focus.json \
  --verbose

# Fight mode (analyze specific fight - prevents data leakage)
python -m agent_loop.run \
  --fight-url "http://ufcstats.com/fight-details/fa4b3f5ce8055921" \
  --n 3
```

**Key Flags**: `--resume-run <timestamp>` | `--fork-run <timestamp>` | `--manual` (IDE-based execution)

---
## Architecture Overview

### Core Design Principles

* **Role-separated micro-agents** (planning → mutation → validation → evaluation → summary)
* **Deterministic execution boundary** (training + metrics computed outside the LLM)
* **Scoped iterative mutation with rollback**
* **Durable artifact-based state** (filesystem-backed, not conversational)

---
### Agent Responsibilities

| Agent               | Responsibility                                                      | Design Intent                |
| ------------------- | ------------------------------------------------------------------- | ---------------------------- |
| **Planning**        | Defines optimization hypothesis and produces structured `plan.json` | Establish direction          |
| **Feature Creator** | Proposes a single scoped change per iteration                       | Constrain mutation surface   |
| **Validator**       | Performs fast syntax/schema checks                                  | Fail early                   |
| **Tester**          | Compares deterministic metrics, decides keep/revert                 | Separate proposer from judge |
| **Summarizer**      | Produces audit report of iterations                                 | Traceability                 |

**Role separation prevents role collapse** — agents that propose changes never evaluate them.

---

### Deterministic Execution Boundary

```
LLM (Proposes)
  → Suggest feature/schema/constraint change

        ↓

Python Pipeline (Deterministic)
  → Rebuild dataset
  → Train XGBoost (fixed params)
  → Compute metrics (held-out set)
  → Emit structured JSON

        ↓

LLM (Decision)
  → Compare metrics vs baseline
  → Keep or revert
  → Update next plan
```

The LLM does not evaluate its own success. Metrics are computed deterministically and used as structured input to decision-making.

---

### State & Reproducibility

Each run creates an isolated artifact directory:

```
agent_artifacts/<timestamp>/
├── plan.json
├── history.json
├── iter_n/
│   ├── change.json
│   ├── analysis.json
│   ├── decision.json
│   └── plan_next.json
├── kept_changes/
└── backups/
```

Key properties:

* All iterations are reproducible from artifacts.
* Runs can be resumed or forked.
* Every mutation is reversible.
* State is explicit, inspectable, and versionable.

---

### Context Strategy

Agents are short-lived. State lives on disk.

Each iteration:

1. Load structured state (`plan.json`, `history.json`)
2. Invoke fresh agent with scoped context
3. Produce structured artifact
4. Run deterministic evaluation
5. Commit or revert
6. Persist updated state

**Context windows are not treated as state.**
Long-running conversational memory was unreliable; structured artifacts proved deterministic and reproducible.

---

## Lessons Learned

## Lessons Learned

### 1. Signal > Model Intelligence

Most failures were due to **ambiguous evaluation criteria**, not weak model reasoning. When success conditions were underspecified, the agent optimized noisy proxies.

**Resolution:** Explicit, structured success criteria in run configs.

```json
{
  "success_criteria": {
    "required_improvements": [
      {"segment": "underdog", "metric": "roi", "min_delta": 0.02}
    ]
  }
}
```

---

### 2. Role Separation Reduces Bias

Single-agent loops reinforced their own hypotheses. When an agent proposes and evaluates a change, confirmation bias emerges.

**Resolution:** Separate proposer and evaluator roles with fresh context per invocation.

---

### 3. Deterministic Systems Must Own Evaluation

Allowing the LLM to self-assess improvements produced false positives. The model would misinterpret metrics or hallucinate gains.

**Resolution:** Deterministic Python pipeline computes metrics. The LLM only compares structured numeric output.

---

### 4. Autonomy Requires Rollback

Without rollback, errors compound across iterations and obscure root causes.

**Resolution:** Automatic backups before mutation. Each iteration is reversible. Accepted changes tracked explicitly.

---

### 5. Context Windows Are Not Durable State

Long-running conversational context degraded reliability. Agents lost track of constraints or hallucinated prior changes.

**Resolution:** Filesystem-backed JSON artifacts as the source of truth. Each agent invocation starts from structured state.

---

### 6. Small Mutations Improve Stability

Large multi-feature changes reduced interpretability and increased validation failures.

**Resolution:** One logical change per iteration (add, modify, or remove — never multiple categories).

---

### 7. Fresh Invocation Prevents Strategy Loops

Agents with accumulated conversational history tended to repeat failed strategies.

**Resolution:** Short-lived agent executions operating on explicit state instead of growing context.

---

## Configuration

### Base Config (`agent_loop_config.json`)

```json
{
  "agent": {
    "model": "claude-sonnet-4-5-20250929",
    "agent_cmd": "claude",
    "timeout": 300
  },
  "model_pipeline": {
    "xgboost_params": {...},
    "holdout_from_year": 2025
  },
  "evaluation": {
    "min_roi_improvement": 0.01
  }
}
```

### Run Config (per-run override)

```json
{
  "run_goal": ["improve_underdog_roi"],
  "prioritize": ["underdog", "top_25_pct"],
  "avoid": ["new_opponent_quality_features"],
  "constraints": {
    "avoid_features": ["quality differences"],
    "max_new_features_per_iteration": 5
  },
  "success_criteria": {
    "required_improvements": [
      {"segment": "underdog", "metric": "roi", "min_delta": 0.02}
    ]
  }
}
```

---

## Data Leakage Prevention (Fight Mode)

In fight mode, the analyzed fight is **NOT added to the database**. This prevents data leakage:

- The fight remains out-of-sample for the entire optimization loop
- Only fighter historical data (from prior fights) is used
- The model is trained on existing data, then evaluated on how well it would have predicted the analyzed fight
- This ensures genuine model improvement rather than overfitting to a specific fight

---

## Summary

This project implements a structured autonomy pattern: separating reasoning from execution, enforcing deterministic evaluation, and maintaining reproducible state across long-running optimization loops.

Agents are short-lived task executors operating against explicit filesystem-backed state — not persistent conversational entities. The LLM proposes changes; deterministic systems evaluate them. Proposal and judgment are role-separated.

This architecture generalizes to any iterative improvement domain:

* **Pentesting** — LLM proposes exploit, system validates
* **Code modification** — LLM proposes patch, tests verify
* **System tuning** — LLM proposes config, benchmarks decide

The core principle is simple: reasoning is probabilistic; evaluation must not be.
