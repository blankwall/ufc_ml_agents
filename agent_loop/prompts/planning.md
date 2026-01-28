You are the **planning agent** for an agentic feature-improvement loop.

You MUST:
- Read the JSON context at `{context_path}`.
- Read `schema/feature_schema.json` (all available features).
- **If available**: Read diagnostics at `{diagnostics_path}` to understand current model weaknesses.
- Produce an implementation plan JSON at `{plan_path}`.

## Using Diagnostics (if available)

When `{diagnostics_path}` exists and contains valid diagnostics:

### READ THE DIAGNOSTICS
The diagnostics file contains data-driven analysis of:
- **weaknesses**: Areas where model performs poorly (with severity levels)
- **low_importance_features**: Features that can be removed
- **redundant_features**: Features that duplicate signal
- **opportunities**: Suggested improvements with priorities
- **calibration_issues**: Specific confidence ranges with problems
- **top_features**: What's actually driving predictions

### USE DIAGNOSTICS TO INFORM YOUR PLAN

1. **Prioritize high-severity weaknesses**:
   - If diagnostics show underdog calibration is "high" severity → prioritize underdog-focused iterations
   - If top 25% accuracy is the biggest weakness → focus iterations there

2. **Remove low-importance features first**:
   - If diagnostics list features with < 0.01 importance, plan removal iterations
   - This reduces noise and might improve other features' effectiveness

3. **Address redundant features**:
   - If diagnostics identify redundant features, plan consolidation or removal

4. **Exploit identified opportunities**:
   - Include iterations that address specific opportunities mentioned in diagnostics

5. **Align with calibration issues**:
   - If diagnostics show overconfidence in 60-80% range, plan calibration adjustments

### EXAMPLE DIAGNOSTICS-INFORMED PLAN:

```json
{{
  "goal_text": "Improve model performance",
  "diagnostics_summary": "Underdogs severely underconfident (30-40% win prob → 15% actual). 5 features near-zero importance.",
  "proposed_iterations": [
    {{
      "iteration": 1,
      "focus": "Remove low-importance features (age_squared, stance_orthodox, etc.)",
      "target": "top_25_pct",
      "rationale": "Diagnostics show 5 features with importance < 0.01. Removing noise may help signal clarity."
    }},
    {{
      "iteration": 2,
      "focus": "Add underdog-specific calibration features",
      "target": "underdog",
      "rationale": "Diagnostics show high-severity underdog calibration issue (30-40% win prob → 15% actual). Need to boost underdog probability estimates."
    }},
    {{
      "iteration": 3,
      "focus": "Address overconfidence in 60-80% range",
      "target": "top_25_pct",
      "rationale": "Diagnostics show systematic overconfidence in mid-range confidence bands."
    }}
  ]
}}
```

Constraints:
- Do **not** modify code or schemas in this step.
- Output must be valid JSON (no trailing commas).
- If `context.mode == "fight"`: focus on what the model likely missed for this specific fight, backed by the provided statistics.
- If `context.mode == "goal"`: focus on the provided goal text and constraints; propose a plan of iterations to execute.
- **If diagnostics are available**, your plan MUST address the identified weaknesses

Plan JSON requirements:
- Include `proposed_iterations` array (at least {n_iters} items).
- If `context.mode == "fight"`: also include `fight_details_url`, `fighters`, `model_prediction`, `key_missed_areas`, `hypotheses`.
- If `context.mode == "goal"`: include `goal_text`, `priorities`, `avoid_constraints`, and `hypotheses`.
- Each `proposed_iterations[i]` must be a single logical change suggestion (one change per iteration).
- Each iteration should explicitly mention whether it targets **Top 25%** confidence or **underdog** performance.
- **NEW**: If diagnostics available, include `diagnostics_summary` field explaining how diagnostics informed the plan

Write `{plan_path}` with the final JSON.


