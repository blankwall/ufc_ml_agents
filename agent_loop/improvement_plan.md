# Agent Loop Improvement Plan

**Status**: In Progress
**Last Updated**: 2025-01-27
**Purpose**: Enhance repeatability, add diff tracking, improve agent architecture

---

## SUMMARY OF IMPROVEMENTS

### Category 1: Repeatability & Diff Tracking (HIGH PRIORITY)
- Generate actual code diffs for each iteration
- Create summary directory of kept changes
- Link diffs in final report
- Enable easy reapplication of successful changes

### Category 2: Agent Architecture (MEDIUM PRIORITY)
- Add Validation Agent (catch errors before expensive builds)
- Add Diagnostics Agent (data-driven planning)
- Split Tester into Evaluator + Decision agents
- Add Refactor Agent (prevent code entropy)

---

## IMPLEMENTATION TRACKER

✅ = Complete | 🚧 = In Progress | ⬜ = Not Started

---

## HIGH PRIORITY - Quick Wins (2-6 hours total)

### ✅ Item 1: Generate Code Diffs for Each Iteration
**Effort**: 2 hours | **Impact**: High
**Status**: ✅ COMPLETED (2025-01-27)

**Problem**: Currently only have `change.json` metadata, no actual code differences

**Solution**: Generate git diff after each feature creator agent runs

**Implementation Notes**:
- Added `generate_diff()` method to `AgentLoop` class in `orchestrator.py`
- Integrated diff generation into `loop()` method after feature_creator completes
- Added `diff_path` to history tracking for easy reference
- Handles new files, deleted files, and modified files
- Output: `iter_<i>/code_diff.patch` in each iteration directory

**Implementation**:
```python
# In orchestrator.py - add new method

def generate_diff(self, iteration: int, backup_dir: str) -> str:
    """
    Generate git diff between backup and current state

    Args:
        iteration: Current iteration number
        backup_dir: Path to iteration backup directory

    Returns:
        Path to generated diff file
    """
    import subprocess
    from pathlib import Path

    diff_path = f"{self.run_dir}/iter_{iteration}/code_diff.patch"
    Path(diff_path).parent.mkdir(parents=True, exist_ok=True)

    with open(diff_path, 'w') as f:
        f.write(f"# Code Diff for Iteration {iteration}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

        # Diff feature_schema.json
        result = subprocess.run(
            ['git', 'diff', '--no-color',
             f'{backup_dir}/schema/feature_schema.json',
             'schema/feature_schema.json'],
            capture_output=True, text=True
        )
        if result.stdout:
            f.write(f"=== schema/feature_schema.json ===\n")
            f.write(result.stdout)
            f.write("\n")

        # Diff monotone_constraints.json
        result = subprocess.run(
            ['git', 'diff', '--no-color',
             f'{backup_dir}/schema/monotone_constraints.json',
             'schema/monotone_constraints.json'],
            capture_output=True, text=True
        )
        if result.stdout:
            f.write(f"=== schema/monotone_constraints.json ===\n")
            f.write(result.stdout)
            f.write("\n")

        # Diff features/ directory
        result = subprocess.run(
            ['git', 'diff', '--no-color',
             f'{backup_dir}/features/',
             'features/'],
            capture_output=True, text=True
        )
        if result.stdout:
            f.write(f"=== features/ ===\n")
            f.write(result.stdout)
            f.write("\n")

    return diff_path
```

**Integration Point**: Call in `loop()` method after feature creator completes
**Output**: `agent_artifacts/<timestamp>/iter_<i>/code_diff.patch`

**Additional Fix** (2025-01-27):
- Added `--permission-mode acceptEdits` to agent command
- This allows agents to write files (plan.json, change.json, etc.) without interactive prompting
- Required for autonomous operation
- Location: `orchestrator.py` line 91 in `run_agent()` method

---

### ✅ Item 2: Create "Kept Changes" Summary Directory
**Effort**: 1 hour | **Impact**: High
**Status**: ✅ COMPLETED (2025-01-27)

**Problem**: Hard to see which iterations succeeded at a glance

**Solution**: Create summary directory with only kept changes

**Implementation**:
```python
# In orchestrator.py - add method

def update_kept_changes(self, iteration: int, decision: str,
                       change_summary: str, diff_path: str) -> None:
    """
    Update kept_changes directory with successful iterations

    Args:
        iteration: Current iteration number
        decision: "keep" or "revert"
        change_summary: Summary of what changed
        diff_path: Path to diff file
    """
    if decision != "keep":
        return

    kept_dir = f"{self.run_dir}/kept_changes"
    os.makedirs(kept_dir, exist_ok=True)

    # Copy diff with descriptive name
    safe_summary = change_summary[:30].replace(" ", "_").replace("/", "_")
    diff_filename = f"iter_{iteration}_{safe_summary}.patch"
    shutil.copy(diff_path, f"{kept_dir}/{diff_filename}")

    # Update index
    index_path = f"{kept_dir}/index.json"
    if os.path.exists(index_path):
        index = utils.read_json(index_path)
    else:
        index = {"kept_iterations": [], "cumulative_diff_path": "cumulative_changes.patch"}

    index["kept_iterations"].append({
        "iteration": iteration,
        "diff_file": diff_filename,
        "summary": change_summary,
        "timestamp": datetime.now().isoformat()
    })

    utils.write_json(index_path, index)

    # Regenerate cumulative diff
    self.generate_cumulative_diff(kept_dir, index["kept_iterations"])
```

**Output Structure**:
```
agent_artifacts/<timestamp>/kept_changes/
├── index.json
├── iter_1_removed_redundant_age_features.patch
├── iter_3_calibrated_opponent_quality.patch
└── cumulative_changes.patch
```

**Implementation Notes**:
- Added `update_kept_changes()` method to `AgentLoop` class in `orchestrator.py`
- Added `generate_cumulative_diff()` method to combine all kept diffs
- Integrated into `loop()` method after keep/revert decision
- Automatically creates `kept_changes/` subdirectory in run directory
- Maintains `index.json` with metadata about each kept iteration
- Generates cumulative patch file for easy reapplication of all changes
- Called only when decision == "keep" to avoid clutter with reverted changes

**Integration Point**: Called in `loop()` method after decision when decision == "keep"
**Output**: `agent_artifacts/<timestamp>/kept_changes/index.json` and patch files
├── index.json
├── iter_1_removed_redundant_age_features.patch
├── iter_3_calibrated_opponent_quality.patch
└── cumulative_changes.patch
```

---

### ⬜ Item 3: Generate Cumulative Diff
**Effort**: 2 hours | **Impact**: High

**Problem**: Want to apply all successful changes at once

**Solution**: Combine all kept diffs into single patch file

**Implementation**:
```python
def generate_cumulative_diff(self, kept_dir: str,
                             kept_iterations: list) -> None:
    """
    Generate cumulative diff of all kept changes

    Args:
        kept_dir: Path to kept_changes directory
        kept_iterations: List of kept iteration metadata
    """
    cumulative_path = f"{kept_dir}/cumulative_changes.patch"

    # Sort by iteration number
    kept_iterations = sorted(kept_iterations, key=lambda x: x["iteration"])

    with open(cumulative_path, 'w') as out:
        out.write("# Cumulative Changes - All Kept Iterations\n")
        out.write(f"# Generated: {datetime.now().isoformat()}\n\n")

        for iter_info in kept_iterations:
            iter_path = f"{kept_dir}/{iter_info['diff_file']}"
            if os.path.exists(iter_path):
                with open(iter_path, 'r') as f:
                    out.write(f"\n{'='*60}\n")
                    out.write(f"# Iteration {iter_info['iteration']}: {iter_info['summary']}\n")
                    out.write(f"{'='*60}\n\n")
                    out.write(f.read())
```

**Usage**:
```bash
# Reapply all successful changes to a fresh baseline
git checkout baseline_branch
git apply agent_loop/agent_artifacts/<timestamp>/kept_changes/cumulative_changes.patch
```

---

### ✅ Item 4: Enhanced Report with Diff Links
**Effort**: 30 minutes | **Impact**: Medium
**Status**: ✅ COMPLETED (2025-01-27)

**Problem**: Final report doesn't link to actual code changes

**Solution**: Update summarizer prompt to include diff links

**Implementation**:
```markdown
# Update agent_loop/prompts/summarizer.md

## ADD to report.html requirements:

### Successful Changes Section

For each iteration where decision == "keep":
- Iteration number
- Change summary (from change.json)
- Key metrics improvement
- Link: `<a href="../iter_<N>/code_diff.patch" target="_blank">View Code Diff</a>`
- Tester reasoning excerpt

### Download All Changes

At the top of report, add:
```html
<div class="download-section">
  <h3>📥 Download Successful Changes</h3>
  <ul>
    <li><a href="kept_changes/cumulative_changes.patch" download>
        Download All Kept Changes (cumulative.patch)
      </a></li>
    <li><a href="kept_changes/index.json" download>
        Download Change Index (index.json)
      </a></li>
  </ul>
  <p><small>To apply: git apply kept_changes/cumulative_changes.patch</small></p>
</div>
```
```

**Files to Modify**:
- `agent_loop/prompts/summarizer.md`

**Implementation Notes**:
- Updated `agent_loop/prompts/summarizer.md` with enhanced reporting requirements
- Added "Download Successful Changes" section at top of report
  - Links to `kept_changes/cumulative_changes.patch` for easy download
  - Links to `kept_changes/index.json` for metadata
  - Shows git apply command for easy reapplication
- Added diff links to iterations table
  - Each kept iteration gets a "📄 View Code Diff" link
  - Links to `../iter_<N>/code_diff.patch` (relative path from report in root)
  - Opens in new tab with `target="_blank"`
- Summarizer agent will read `history.json` for iteration info and `kept_changes/index.json` for metadata
- Improved styling instructions (green for kept, red for reverted)

**Integration Point**: Automatically used by summarizer agent at end of each run
**Output**: Enhanced HTML report with diff links and download section

---

## MEDIUM PRIORITY - Agent Architecture (6-10 hours total)

### ✅ Item 5: Add Validation Agent
**Effort**: 3 hours | **Impact**: High
**Status**: ✅ COMPLETED (2025-01-27)

**Problem**: If feature creator produces invalid code/syntax, waste 10-minute build

**Solution**: Add validation step between feature creator and build

**Implementation**:

**Step 1: Create validation prompt**
```markdown
# agent_loop/prompts/validator.md

You are the validation agent.

Your job: Verify the feature_creator's changes are valid BEFORE building.

## MUST CHECK:

1. JSON Syntax
   - schema/feature_schema.json is valid JSON
   - schema/monotone_constraints.json is valid JSON

2. Python Syntax
   - All modified .py files have valid syntax
   - All imports resolve successfully

3. Schema Consistency
   - Every feature in feature_schema.json exists in code
   - No duplicate feature names
   - All features in schema have implementations

4. Feature Naming
   - Feature names follow conventions (snake_case)
   - No reserved words used

## OUTPUT:

Write to {validation_path}:
```json
{
  "status": "pass" | "fail",
  "errors": ["list of critical errors"],
  "warnings": ["list of non-critical issues"],
  "checks_performed": {
    "json_syntax": "pass" | "fail",
    "python_syntax": "pass" | "fail",
    "schema_consistency": "pass" | "fail",
    "feature_naming": "pass" | "fail"
  },
  "can_proceed": true
}
```

## IMPORTANT:
- If status == "fail", the orchestrator will revert changes
- Be strict - catch issues before expensive build
- Provide specific error messages for debugging
```

**Step 2: Add validation to orchestrator**
```python
# In orchestrator.py

def validation(self, iteration: int) -> bool:
    """
    Run validation agent to check code quality

    Returns:
        True if validation passed, False otherwise
    """
    validation_path = f"{self.run_dir}/iter_{iteration}/validation.json"

    # Prepare validation prompt
    prompt_template = self.read_prompt("validator.md")
    prompt = prompt_template.format(
        validation_path=validation_path,
        iteration=iteration
    )

    # Run validation agent
    agent_cmd = f"claude-code '{prompt}'"
    result = subprocess.run(agent_cmd, shell=True,
                          capture_output=True, text=True)

    # Read validation result
    validation = utils.read_json(validation_path)

    if not validation.get("can_proceed", False):
        print(f"❌ Validation failed: {validation['errors']}")
        # Auto-revert
        self.restore_code_backup(iteration)
        return False

    return True
```

**Step 3: Integrate into loop**
```python
# In loop() method, after feature_creator:

# Run validation
if not self.validation(iteration):
    # Revert changes, log failure, skip to next iteration
    self.record_iteration_failure(iteration, "validation_failed")
    continue

# Only proceed with build if validation passed
self.feature_builder(iteration)
```

**Output**: `iter_<i>/validation.json`

**Implementation Notes**:
- Created `agent_loop/prompts/validator.md` with comprehensive validation checks
- Added `validation()` method to `AgentLoop` class in `orchestrator.py`
- Integrated into loop after feature_creator, before feature_builder
- Automatically reverts changes on validation failure
- Records validation failure in history.json
- Continues to next iteration with same plan (no progress lost)
- Validates:
  - JSON syntax (feature_schema.json, monotone_constraints.json)
  - Python syntax (all modified .py files)
  - Schema consistency (features in schema have implementations)
  - Feature naming (snake_case, no reserved words)
  - Monotone constraints consistency
- Saves ~10 minutes per failed iteration by catching errors before expensive build
- Output: `iter_<i>/validation.json` with pass/fail status and detailed error messages

**Integration Point**: Called in `loop()` method after diff generation, before build
**Benefits**:
- Prevents wasted build time on invalid code
- Catches syntax errors early
- Provides specific error messages for debugging
- Auto-recovers and continues to next iteration

---

### ⬜ Item 6: Add Diagnostics Agent
**Effort**: 6 hours | **Impact**: Medium-High

**Problem**: Planning agent makes assumptions without data

**Solution**: Add diagnostic analysis before planning phase

**Implementation**:

**Step 1: Create diagnostics prompt**
```markdown
# agent_loop/prompts/diagnostics.md

You are the diagnostics agent.

Your job: Analyze the current model to identify weaknesses and opportunities.

## INPUTS:
- {baseline_eval_path}: Current model evaluation
- {feature_importance_path}: Feature importance scores
- {training_stats_path}: Training data statistics

## ANALYZE:

1. Feature Importance
   - Which features are most/least important?
   - Are there many near-zero importance features (candidates for removal)?

2. Model Performance by Segment
   - Where does accuracy suffer? (favorites vs underdogs)
   - Which confidence buckets are poorly calibrated?
   - Any specific odds bands with issues?

3. Redundancy Detection
   - Look for features with similar names/patterns
   - Check for interaction features that might duplicate base features

4. Data Quality
   - Missing value patterns
   - Outlier detection
   - Distribution skews

## OUTPUT:

Write to {diagnostics_path}:
```json
{
  "weaknesses": [
    {
      "area": "underdog_calibration",
      "severity": "high" | "medium" | "low",
      "evidence": "Underdogs with 30-40% win prob have only 15% actual win rate",
      "suggested_focus": "Features affecting underdog probability estimation",
      "metrics": {
        "underdog_win_prob_30_40": 0.15,
        "expected": 0.35,
        "delta": -0.20
      }
    }
  ],
  "redundant_features": [
    {
      "features": ["age_x_opponent_quality", "age_x_opponent_quality_diff"],
      "importance_scores": [0.012, 0.008],
      "recommendation": "Consider removing one - they capture similar signal"
    }
  ],
  "low_importance_features": [
    {
      "feature": "fighter_stance_orthodox",
      "importance": 0.001,
      "recommendation": "Candidate for removal"
    }
  ],
  "opportunities": [
    {
      "area": "high_variance_predictions",
      "description": "Model shows poor calibration in 60-80% confidence range",
      "potential_fix": "Add confidence interval features"
    }
  ]
}
```
```

**Step 2: Integrate before planning**
```python
# In orchestrator.py, before planning phase:

def run_diagnostics(self) -> dict:
    """Run diagnostics agent to analyze current model"""
    diagnostics_path = f"{self.run_dir}/diagnostics.json"

    # Gather input paths
    baseline_eval = "models/saved/baseline_metrics.json"
    feature_importance = "models/saved/baseline_feature_importance.csv"

    # Prepare prompt
    prompt_template = self.read_prompt("diagnostics.md")
    prompt = prompt_template.format(
        baseline_eval_path=baseline_eval,
        feature_importance_path=feature_importance,
        diagnostics_path=diagnostics_path
    )

    # Run diagnostics agent
    self.run_agent("diagnostics", prompt)

    return utils.read_json(diagnostics_path)

# In main flow:
diagnostics = self.run_diagnostics()
# Pass diagnostics to planning agent
self.planning(diagnostics=diagnostics)
```

**Step 3: Update planning prompt**
```markdown
# In agent_loop/prompts/planning.md, add:

## ADDITIONAL INPUT:
You will also receive {diagnostics_path} with model analysis.

## USE DIAGNOSTICS TO:
- Focus on identified weaknesses
- Remove suggested low-importance features
- Address redundant features
- Exploit identified opportunities
```

**Output**: `diagnostics.json` in run directory

---

### ⬜ Item 7: Split Tester → Evaluator + Decision Agents
**Effort**: 4 hours | **Impact**: Medium

**Problem**: Tester agent has too many responsibilities

**Solution**: Separate into pure analysis (Evaluator) and strategy (Decision)

**Implementation**:

**Step 1: Create evaluator.md**
```markdown
# agent_loop/prompts/evaluator.md

You are the evaluator agent.

Your job: Pure metrics analysis. NO decision making.

## INPUTS:
- {eval_path}: New model evaluation
- {baseline_path}: Baseline model evaluation
- {change_path}: What was changed

## ANALYZE:

1. Overall Metrics
   - Accuracy delta
   - Brier score delta
   - AUC delta

2. Top 25% Performance
   - Accuracy change
   - Calibration change

3. Underdog Performance
   - Accuracy change
   - ROI change
   - Calibration by odds band

4. By Segment
   - Favorites performance
   - Underdogs performance

## OUTPUT:

Write to {analysis_path}:
```json
{
  "overall": {
    "accuracy_delta": +0.0123,
    "brier_delta": -0.0045,
    "auc_delta": +0.0089,
    "improved": true
  },
  "top_25_pct": {
    "accuracy_delta": +0.0322,
    "calibration_delta": -0.02,
    "improved": true
  },
  "underdog": {
    "accuracy_delta": -0.0156,
    "roi_delta": -0.05,
    "calibration_issue": "Underdogs still underconfident",
    "improved": false
  },
  "key_insights": [
    "Top 25% improved significantly (+3.22%)",
    "Underdog performance degraded (-1.56% accuracy, -5% ROI)",
    "Quality-difference features appear to amplify penalties on underdogs"
  ],
  "metrics_summary": {
    "baseline": {...},
    "new_model": {...},
    "deltas": {...}
  }
}
```

## IMPORTANT:
- Be objective and thorough
- Do NOT make keep/revert recommendations
- Focus on accurate metrics comparison
- Highlight both improvements and regressions
```

**Step 2: Modify decision.md** (renamed from tester.md)
```markdown
# agent_loop/prompts/decision.md

You are the decision agent.

Your job: Make strategic keep/revert decisions based on evaluation.

## INPUTS:
- {analysis_path}: Evaluator's objective analysis
- {change_path}: What was changed
- {plan_path}: Current plan
- {history_path}: Previous iterations

## DECISION FRAMEWORK:

PRIORITIES (in order):
1. Underdog performance (TOP PRIORITY)
2. Top 25% accuracy
3. Overall accuracy
4. Calibration quality

DECISION LOGIC:
- If underdog improved significantly → KEEP
- If underdog degraded significantly → REVERT (unless Top 25% improved massively)
- If underdog neutral, Top 25% improved → KEEP
- If both degraded → REVERT

## OUTPUT:

Write to {decision_path}:
```json
{
  "decision": "keep" | "revert",
  "reasoning": "Clear explanation of decision",
  "key_considerations": [
    "Underdog ROI decreased by 5%",
    "Top 25% accuracy improved by 3.22%",
    "Prioritizing underdog performance per goals"
  ],
  "metrics_considered": {
    "underdog_roi_delta": -0.05,
    "top_25_accuracy_delta": +0.0322
  },
  "next_steps": "What to try next iteration",
  "lessons_learned": "Key insights to apply to future iterations"
}
```

Also write to {plan_next_path}: Updated plan based on this decision
```

**Step 3: Update orchestrator**
```python
# In loop() method, replace tester() with:

def evaluation_phase(self, iteration: int):
    """Run evaluator and decision agents"""

    # 1. Run evaluator (pure analysis)
    self.evaluator(iteration)
    analysis = utils.read_json(f"{self.run_dir}/iter_{iteration}/analysis.json")

    # 2. Run decision agent (strategy)
    self.decision(iteration, analysis)
    decision_data = utils.read_json(f"{self.run_dir}/iter_{iteration}/decision.json")

    # 3. Handle decision
    if decision_data["decision"] == "revert":
        self.restore_code_backup(iteration)

    return decision_data["decision"]
```

**Output**:
- `iter_<i>/analysis.json` (from evaluator)
- `iter_<i>/decision.json` (from decision agent)

---

## LOWER PRIORITY - Nice to Have (8-10 hours total)

### ⬜ Item 8: Add Refactor Agent
**Effort**: 8 hours | **Impact**: Medium (long-term)

**Problem**: Code entropy accumulates after many iterations

**Solution**: Periodic cleanup and consolidation

**Implementation**:
```markdown
# agent_loop/prompts/refactor.md

You are the refactor agent.

Your job: Clean up and consolidate code after iterations.

## TRIGGERS:
- Every 5 iterations automatically
- Or explicit --refactor flag

## TASKS:

1. Remove Dead Code
   - Find features in schema but not used
   - Remove orphaned utility functions

2. Consolidate Redundant Features
   - Merge similar features
   - Remove near-duplicates

3. Improve Organization
   - Group related features
   - Improve naming consistency

4. Update Documentation
   - Update feature_schema.json descriptions
   - Add comments to complex features

5. Standardize Patterns
   - Ensure consistent feature naming
   - Standardize quality adjustment patterns

## CONSTRAINTS:
- Must NOT change model behavior
- Run full evaluation after refactor
- If metrics degrade, revert changes
```

**Integration**:
```python
# In orchestrator.py

def should_run_refactor(self, iteration: int) -> bool:
    """Check if refactor should run"""
    return (iteration % 5 == 0) or self.config.run_refactor

def refactor(self, iteration: int) -> bool:
    """Run refactor agent and verify changes"""
    # Run refactor agent
    # Build and evaluate
    # If metrics degraded, revert
    # Return success status
```

---

### ⬜ Item 9: Add Reapply Utility
**Effort**: 2 hours | **Impact**: Low (convenience)

**Problem**: Want to easily reapply successful changes

**Solution**: Command-line utility for reapplication

**Implementation**:
```python
# In agent_loop/utils.py

def reapply_changes(run_dir: str, target_baseline: str = None):
    """
    Reapply all kept changes from a run

    Usage:
        python -m agent_loop.utils --reapply agent_artifacts/20260126_190719 --baseline main

    Args:
        run_dir: Path to agent_artifacts/<timestamp>/
        target_baseline: Git ref to reset to before applying (optional)
    """
    import subprocess

    # Reset to baseline if specified
    if target_baseline:
        print(f"Resetting to {target_baseline}...")
        subprocess.run(['git', 'checkout', target_baseline], check=True)

    # Apply cumulative diff
    kept_dir = f"{run_dir}/kept_changes"
    cumulative_diff = f"{kept_dir}/cumulative_changes.patch"

    if not os.path.exists(cumulative_diff):
        print(f"Error: {cumulative_diff} not found")
        return False

    print(f"Applying {cumulative_diff}...")
    result = subprocess.run(['git', 'apply', cumulative_diff],
                           capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error applying diff:\n{result.stderr}")
        return False

    print("✅ Changes applied successfully")
    return True

# Add CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reapply', type=str)
    parser.add_argument('--baseline', type=str, default=None)
    args = parser.parse_args()

    if args.reapply:
        reapply_changes(args.reapply, args.baseline)
```

**Usage**:
```bash
python -m agent_loop.utils --reapply agent_loop/agent_artifacts/20260126_190719 --baseline main
```

---

### ⬜ Item 10: Agent Performance Tracking
**Effort**: 3 hours | **Impact**: Low (monitoring)

**Problem**: Want to measure agent success rates over time

**Solution**: Track agent performance metrics

**Implementation**:
```python
# In orchestrator.py, add to LoopConfig:

@dataclass
class AgentMetrics:
    """Track agent performance"""
    validation_failures: int = 0
    revert_rate: float = 0.0
    average_iteration_time: float = 0.0
    kept_changes_per_iteration: float = 0.0

# Track throughout run:
def record_agent_performance(self, iteration: int,
                            validation_passed: bool,
                            decision: str,
                            iteration_time: float):
    """Record agent performance metrics"""

    if not hasattr(self, 'agent_metrics'):
        self.agent_metrics = AgentMetrics()

    if not validation_passed:
        self.agent_metrics.validation_failures += 1

    if decision == "keep":
        self.agent_metrics.kept_changes_per_iteration += 1

    # Update averages
    n = iteration
    self.agent_metrics.average_iteration_time = (
        (self.agent_metrics.average_iteration_time * (n-1) + iteration_time) / n
    )
    self.agent_metrics.revert_rate = (
        (n - self.agent_metrics.kept_changes_per_iteration) / n
    )

# Save to run directory at end:
def save_agent_metrics(self):
    """Save agent performance metrics"""
    metrics_path = f"{self.run_dir}/agent_metrics.json"
    utils.write_json(metrics_path, asdict(self.agent_metrics))
```

**Output**: `agent_metrics.json` in run directory

---

## TESTING CHECKLIST

For each improvement, verify:

- [ ] Code compiles and runs
- [ ] New artifacts are generated correctly
- [ ] Report includes new information
- [ ] Diff files are valid patches
- [ ] Can reapply changes successfully
- [ ] Validation catches errors correctly
- [ ] Diagnostics provide useful insights
- [ ] Evaluator produces objective analysis
- [ ] Decision agent makes reasonable choices
- [ ] No regressions in existing functionality

---

## ROLLBACK PLAN

If any improvement introduces issues:

1. **Revert specific commit**: `git revert <commit-hash>`
2. **Restore previous orchestrator**: `git checkout HEAD~1 agent_loop/orchestrator.py`
3. **Report issue**: Document in improvement_plan.md under "Issues Encountered"

---

## PROGRESS TRACKING

### Completed Items
- ✅ Item 1: Generate Code Diffs (2025-01-27)
- ✅ Item 2: Create "Kept Changes" Summary Directory (2025-01-27)
- ✅ Item 3: Generate Cumulative Diff (2025-01-27) - included in Item 2
- ✅ Item 4: Enhanced Report with Diff Links (2025-01-27)
- ✅ Item 5: Add Validation Agent (2025-01-27)

### In Progress
- 🚧 None currently

### Next Up
- ⬜ Item 6: Add Diagnostics Agent (6 hours - Medium-High Impact)
- ⬜ Item 7: Split Tester → Evaluator + Decision Agents (4 hours - Medium Impact)
- ⬜ Item 8: Add Refactor Agent (8 hours - Medium long-term)
- ⬜ Item 9: Add Reapply Utility (2 hours - Low convenience)
- ⬜ Item 10: Agent Performance Tracking (3 hours - Low monitoring)

### Blocked
- None

---

## NOTES

- Implement one item at a time
- Test thoroughly before moving to next
- Update this file as you go
- Add lessons learned to each item
- Feel free to adjust priorities based on experience
