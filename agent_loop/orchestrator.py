from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from agent_loop.utils import (
    utc_timestamp,
    read_text,
    write_text,
    write_json,
    read_json,
    run_cmd,
    backup_paths,
    restore_paths,
    newest_matching,
)
from agent_loop.tools import (
    scrape_fight_details_json,
    run_xgboost_predict,
)


@dataclass(frozen=True)
class LoopConfig:
    repo_root: Path
    fight_url: Optional[str]
    n_iters: int
    goal_text: Optional[str] = None
    model: str = "claude-sonnet-4-5-20250929"
    agent_cmd: str = "claude"
    verbose: bool = False
    manual: bool = False  # If True, pause for manual agent execution instead of running agent binary

    # Model/pipeline settings
    holdout_from_year: int = 2025
    baseline_json: Path = Path("models/baseline.json")
    train_model_name_prefix: str = "agent_loop_model"
    xgboost_predict_model_name: str = "baseline_jan_11_2026_age_feature_add_striking_landed"

    # Evaluation settings
    eval_min_year: int = 2025
    odds_path: Path = Path("ufc_2025_odds.csv")
    odds_date_tolerance_days: int = 5


class AgentLoop:
    def __init__(self, cfg: LoopConfig, *, run_dir: Optional[Path] = None):
        self.cfg = cfg
        if run_dir is not None:
            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.run_ts = self.run_dir.name
        else:
            self.run_ts = utc_timestamp()
            self.run_dir = cfg.repo_root / "agent_loop" / "agent_artifacts" / self.run_ts
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def dbg(self, msg: str) -> None:
        """Simple, switchable debug logger."""
        if self.cfg.verbose:
            print(f"[agent_loop] {msg}")

    # -----------------------
    # Agent runner
    # -----------------------
    def run_agent(self, prompt: str, wait_for_file: Optional[Path] = None) -> None:
        """
        Run Claude Code CLI in non-interactive mode.
        Uses: -p (print) --output-format json to make it script-friendly.
        We still rely on the agent to write artifact files to disk.

        Args:
            prompt: The prompt to send to the agent
            wait_for_file: Optional path to a file that should exist after completion
        """
        # Write prompt to a temp file for reference
        prompt_file = self.run_dir / "logs" / f"prompt_{utc_timestamp()}.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")
        
        # Claude Code CLI: cwd handles workspace, prompt passed via stdin
        # --permission-mode acceptEdits: Allow agents to write files (plan.json, change.json, etc.)
        #   without interactive prompting. This is required for autonomous operation.
        cmd = [
            self.cfg.agent_cmd,
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",  # Allow agents to write files without prompting
            "--model",
            self.cfg.model,
        ]
        # Don't pass prompt as argument - pass via stdin instead
        # This is more robust for long prompts and avoids shell escaping issues
        
        # Store raw agent output for debugging
        out_path = self.run_dir / "logs" / f"agent_{utc_timestamp()}.json"
        stderr_path = out_path.with_suffix(".stderr.txt")
        self.dbg(f"Running agent: model={self.cfg.model}, output={out_path}")
        self.dbg(f"Prompt file (for reference): {prompt_file}")
        
        try:
            # Pass prompt via stdin, with timeout (10 minutes)
            proc = subprocess.run(
                cmd,
                cwd=str(self.cfg.repo_root),
                text=True,
                input=prompt,
                stdout=open(out_path, "w", encoding="utf-8"),
                stderr=open(stderr_path, "w", encoding="utf-8"),
                check=True,
                timeout=600,  # 10 minutes
            )
            self.dbg(f"Agent completed: {out_path}")
        except subprocess.TimeoutExpired as e:
            stderr_content = ""
            if stderr_path.exists():
                stderr_content = stderr_path.read_text(encoding="utf-8")[:3000]
            raise RuntimeError(
                f"Agent command timed out after 10 minutes. Stderr:\n{stderr_content}\n\n"
                f"Prompt file: {prompt_file}\n\n"
                f"Hint: The agent may be waiting for input or trying to connect to a service."
            ) from e
        except subprocess.CalledProcessError as e:
            # Show stderr if available for debugging
            stderr_content = ""
            if stderr_path.exists():
                stderr_content = stderr_path.read_text(encoding="utf-8")[:3000]  # First 3KB
            raise RuntimeError(
                f"Agent command failed with code {e.returncode}. Stderr:\n{stderr_content}\n\n"
                f"Prompt file: {prompt_file}"
            ) from e
        except Exception as e:
            if stderr_path.exists():
                stderr_content = stderr_path.read_text(encoding="utf-8")[:3000]
                raise RuntimeError(
                    f"Agent command failed unexpectedly. Stderr:\n{stderr_content}\n\n"
                    f"Original error: {e}\n"
                    f"Prompt file: {prompt_file}"
                ) from e
            raise

    def _render_prompt(self, template_path: Path, **kwargs) -> str:
        tpl = read_text(template_path)
        return tpl.format(**{k: str(v) for k, v in kwargs.items()})

    # -----------------------
    # Core steps
    # -----------------------
    def build_context(self) -> Path:
        """
        Build context for agents:
        - Goal mode: use goal text directly
        - Fight mode: scrape fight details and run xgboost_predict (without adding fight to DB)
        """
        context_path = self.run_dir / "context.json"
        self.dbg(f"Run dir: {self.run_dir}")
        if self.cfg.fight_url:
            self.dbg(f"Building context for fight: {self.cfg.fight_url}")
        else:
            self.dbg("Building context for generic goal (non-fight mode)")

        # Goal-only mode: no scraping/prediction ingestion, just seed context with goals/constraints.
        if not self.cfg.fight_url:
            if not self.cfg.goal_text:
                raise RuntimeError("goal_text is required when fight_url is not provided")

            context = {
                "mode": "goal",
                "goal_text": self.cfg.goal_text,
                "constraints": {
                    "prioritize": ["top_25_pct", "underdog"],
                    "avoid": [
                        "Do NOT add new features involving quality differences (thresholds/interactions/quality-adjusted weighting).",
                    ],
                },
                "schema_paths": {
                    "feature_schema": "schema/feature_schema.json",
                    "monotone_constraints": "schema/monotone_constraints.json",
                    "feature_exclusions": "features/feature_exclusions.py",
                },
            }
            write_json(context_path, context)
            self.dbg(f"Wrote context.json: {context_path}")
            return context_path

        # Fight mode: scrape fight details WITHOUT adding to database (prevents data leakage)
        # The fight remains out-of-sample for analysis and model improvement

        # 1) Scrape fight details for metadata (fighters, result, stats)
        fight_meta = scrape_fight_details_json(self.cfg.fight_url)
        self.dbg(f"Fight meta: {fight_meta}")
        fighters = fight_meta.get("fighters") or []
        if len(fighters) < 2 or not fighters[0].get("ufcstats_id") or not fighters[1].get("ufcstats_id"):
            raise RuntimeError("Unable to extract both fighter UFCStats IDs from fight-details page.")
        self.dbg(
            f"Fighters: "
            f"F1={fighters[0].get('name')} ({fighters[0].get('ufcstats_id')}), "
            f"F2={fighters[1].get('name')} ({fighters[1].get('ufcstats_id')})"
        )

        # 2) Run xgboost_predict for this matchup (store raw output)
        # Note: Fighters should already exist in DB from their prior fights
        # The fight itself is NOT added to DB, remaining out-of-sample
        pred_out = self.run_dir / "xgboost_predict.txt"
        self.dbg(f"Running xgboost_predict → {pred_out}")
        self.dbg("Fight remains out-of-sample (not added to training data)")
        run_xgboost_predict(
            self.cfg.repo_root,
            fighter_1_ufcstats_id=fighters[0]["ufcstats_id"],
            fighter_2_ufcstats_id=fighters[1]["ufcstats_id"],
            model_name=self.cfg.xgboost_predict_model_name,
            out_path=pred_out,
        )
        self.dbg("xgboost_predict complete.")

        context = {
            "mode": "fight",
            "fight_details_url": self.cfg.fight_url,
            "fighters": fighters,
            "xgboost_predict_model_name": self.cfg.xgboost_predict_model_name,
            "xgboost_predict_output_path": str(pred_out),
        }
        write_json(context_path, context)
        self.dbg(f"Wrote context.json: {context_path}")
        return context_path

    def planning(self, context_path: Path) -> Path:
        plan_path = self.run_dir / "plan.json"
        self.dbg(f"Planning step: writing plan → {plan_path}")
        prompt = self._render_prompt(
            self.cfg.repo_root / "agent_loop" / "prompts" / "planning.md",
            context_path=context_path,
            plan_path=plan_path,
            n_iters=self.cfg.n_iters,
        )
        self.run_agent(prompt)
        if not plan_path.exists():
            raise RuntimeError(f"Planning agent did not create plan file: {plan_path}")
        self.dbg("Planning complete.")
        return plan_path

    def feature_builder(self, iter_idx: int) -> Path:
        """
        Non-agent step:
        - backup current data/processed outputs
        - build dataset
        - train model
        - evaluate model
        Returns path to newest reports_strict/model_eval_*.json
        """
        logs_dir = self.run_dir / "logs" / f"iter_{iter_idx}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.dbg(f"[iter {iter_idx}] Feature builder starting. logs_dir={logs_dir}")

        # Backup data outputs before rebuild (as requested)
        backup_dir = self.run_dir / "backups" / f"iter_{iter_idx}_data_before"
        self.dbg(f"[iter {iter_idx}] Backing up data/processed → {backup_dir}")
        backup_paths(
            [
                self.cfg.repo_root / "data" / "processed",
            ],
            backup_dir,
        )

        # Build full feature set dataset
        self.dbg(f"[iter {iter_idx}] Building training_data.csv (this can take ~10 min)…")
        run_cmd(
            ["python3", "-m", "features.feature_pipeline", "--create", "--feature-set", "full"],
            cwd=self.cfg.repo_root,
            stdout_path=logs_dir / "feature_pipeline.stdout.txt",
            stderr_path=logs_dir / "feature_pipeline.stderr.txt",
            check=True,
        )

        # Train model (name includes run_ts + iteration)
        model_name = f"{self.cfg.train_model_name_prefix}_{self.run_ts}_iter{iter_idx}"
        self.dbg(f"[iter {iter_idx}] Training model: {model_name}")
        train_cmd = [
            "python3",
            "-m",
            "models.xgboost_model",
            "--train",
            "--evaluate",
            "--check-calibration",
            "--save-plots",
            "--export-schema",
            "--data-path",
            "data/processed/training_data.csv",
            "--n-estimators",
            "200",
            "--max-depth",
            "4",
            "--holdout-from-year",
            str(self.cfg.holdout_from_year),
            "--learning-rate",
            "0.05",
            "--subsample",
            "0.8",
            "--colsample-bytree",
            "0.8",
            "--model-name",
            model_name,
        ]
        run_cmd(
            train_cmd,
            cwd=self.cfg.repo_root,
            stdout_path=logs_dir / "train.stdout.txt",
            stderr_path=logs_dir / "train.stderr.txt",
            check=True,
        )

        # Evaluate model (compare-to-baseline)
        self.dbg(f"[iter {iter_idx}] Evaluating model vs baseline: {self.cfg.baseline_json}")
        eval_cmd = [
            "python3",
            "-m",
            "evaluation.evaluate_model",
            "--data-path",
            "data/processed/training_data.csv",
            "--odds-path",
            str(self.cfg.odds_path),
            "--min-year",
            str(self.cfg.eval_min_year),
            "--output-dir",
            "reports_strict",
            "--odds-date-tolerance-days",
            str(self.cfg.odds_date_tolerance_days),
            "--model-name",
            model_name,
            "--symmetric",
            "--compare-to-baseline",
            "--baseline-path",
            str(self.cfg.baseline_json),
        ]
        run_cmd(
            eval_cmd,
            cwd=self.cfg.repo_root,
            stdout_path=logs_dir / "evaluate.stdout.txt",
            stderr_path=logs_dir / "evaluate.stderr.txt",
            check=True,
        )

        latest = newest_matching(self.cfg.repo_root / "reports_strict", "model_eval_*.json")
        if not latest:
            raise RuntimeError("Could not find reports_strict/model_eval_*.json after evaluation")

        # Copy eval json into run_dir for immutability
        dest = self.run_dir / f"iter_{iter_idx}_model_eval.json"
        dest.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
        self.dbg(f"[iter {iter_idx}] Wrote eval JSON snapshot: {dest}")
        return dest

    def generate_diff(self, iteration: int, backup_dir: Path) -> Path:
        """
        Generate git diff between backup and current state.

        Captures actual code changes made by the feature_creator agent.
        This enables repeatability and easy review of what changed.

        Args:
            iteration: Current iteration number
            backup_dir: Path to iteration backup directory

        Returns:
            Path to generated diff file
        """
        diff_path = self.run_dir / f"iter_{iteration}" / "code_diff.patch"
        diff_path.parent.mkdir(parents=True, exist_ok=True)

        with open(diff_path, 'w') as f:
            f.write(f"# Code Diff for Iteration {iteration}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Backup: {backup_dir}\n\n")

            # Helper function to find backup file (handles absolute path structure)
            def get_backup_path(relative_path: Path) -> Path:
                """Get backup file path, accounting for backup_paths() full path preservation"""
                # backup_paths() preserves full path: backup_dir/Users/username/repo/...
                # So we need to construct the full path
                full_path = self.cfg.repo_root / relative_path
                # Convert to relative path (strip leading /) for backup lookup
                rel_key = full_path.as_posix().lstrip("/")
                return backup_dir / rel_key

            # Diff feature_schema.json
            current_schema = self.cfg.repo_root / "schema" / "feature_schema.json"
            backup_schema = get_backup_path(Path("schema/feature_schema.json"))
            if backup_schema.exists() and current_schema.exists():
                result = subprocess.run(
                    ['git', 'diff', '--no-color', str(backup_schema), str(current_schema)],
                    capture_output=True, text=True, cwd=str(self.cfg.repo_root)
                )
                if result.stdout.strip():
                    f.write(f"=== schema/feature_schema.json ===\n")
                    f.write(result.stdout)
                    f.write("\n")
                elif current_schema.exists() and not backup_schema.exists():
                    f.write(f"=== schema/feature_schema.json (NEW FILE) ===\n")
                    f.write(current_schema.read_text(encoding='utf-8'))
                    f.write("\n")

            # Diff monotone_constraints.json
            current_constraints = self.cfg.repo_root / "schema" / "monotone_constraints.json"
            backup_constraints = get_backup_path(Path("schema/monotone_constraints.json"))
            if backup_constraints.exists() and current_constraints.exists():
                result = subprocess.run(
                    ['git', 'diff', '--no-color', str(backup_constraints), str(current_constraints)],
                    capture_output=True, text=True, cwd=str(self.cfg.repo_root)
                )
                if result.stdout.strip():
                    f.write(f"=== schema/monotone_constraints.json ===\n")
                    f.write(result.stdout)
                    f.write("\n")
                elif current_constraints.exists() and not backup_constraints.exists():
                    f.write(f"=== schema/monotone_constraints.json (NEW FILE) ===\n")
                    f.write(current_constraints.read_text(encoding='utf-8'))
                    f.write("\n")

            # Diff features/ directory
            backup_features_base = get_backup_path(Path("features"))
            current_features = self.cfg.repo_root / "features"

            if backup_features_base.exists() and current_features.exists():
                # Get list of Python files in current features/
                for py_file in sorted(current_features.rglob("*.py")):
                    rel_path = py_file.relative_to(current_features)
                    backup_py = backup_features_base / rel_path

                    if backup_py.exists():
                        result = subprocess.run(
                            ['git', 'diff', '--no-color', str(backup_py), str(py_file)],
                            capture_output=True, text=True, cwd=str(self.cfg.repo_root)
                        )
                        if result.stdout.strip():
                            f.write(f"=== features/{rel_path} ===\n")
                            f.write(result.stdout)
                            f.write("\n")
                    else:
                        # New file
                        f.write(f"=== features/{rel_path} (NEW FILE) ===\n")
                        f.write(py_file.read_text(encoding='utf-8'))
                        f.write("\n")

                # Check for deleted files
                for backup_py in sorted(backup_features_base.rglob("*.py")):
                    rel_path = backup_py.relative_to(backup_features_base)
                    current_py = current_features / rel_path
                    if not current_py.exists():
                        f.write(f"=== features/{rel_path} (DELETED) ===\n")
                        f.write(f"# File was removed\n")
                        f.write(f"# Previously at: {backup_py}\n\n")

        self.dbg(f"[iter {iteration}] Generated diff: {diff_path}")
        return diff_path

    def generate_cumulative_diff(self, kept_dir: Path,
                                 kept_iterations: List[Dict]) -> None:
        """
        Generate cumulative diff of all kept changes.

        Args:
            kept_dir: Path to kept_changes directory
            kept_iterations: List of kept iteration metadata
        """
        cumulative_path = kept_dir / "cumulative_changes.patch"

        # Sort by iteration number
        kept_iterations = sorted(kept_iterations, key=lambda x: x["iteration"])

        with open(cumulative_path, 'w') as out:
            out.write("# Cumulative Changes - All Kept Iterations\n")
            out.write(f"# Generated: {datetime.now().isoformat()}\n")
            out.write(f"# Total iterations: {len(kept_iterations)}\n\n")

            for iter_info in kept_iterations:
                iter_path = kept_dir / iter_info['diff_file']
                if iter_path.exists():
                    with open(iter_path, 'r') as f:
                        out.write(f"\n{'='*60}\n")
                        out.write(f"# Iteration {iter_info['iteration']}: {iter_info['summary']}\n")
                        out.write(f"# Timestamp: {iter_info['timestamp']}\n")
                        out.write(f"{'='*60}\n\n")
                        out.write(f.read())

        self.dbg(f"Generated cumulative diff: {cumulative_path}")

    def update_kept_changes(self, iteration: int, decision: str,
                           change_summary: str, diff_path: Path) -> None:
        """
        Update kept_changes directory with successful iterations.

        Args:
            iteration: Current iteration number
            decision: "keep" or "revert"
            change_summary: Summary of what changed
            diff_path: Path to diff file
        """
        if decision != "keep":
            return

        kept_dir = self.run_dir / "kept_changes"
        kept_dir.mkdir(parents=True, exist_ok=True)

        # Copy diff with descriptive name
        safe_summary = change_summary[:30].replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_summary = ''.join(c if c.isalnum() or c in '_-' else '_' for c in safe_summary)
        diff_filename = f"iter_{iteration}_{safe_summary}.patch"
        shutil.copy(str(diff_path), str(kept_dir / diff_filename))

        # Update index
        index_path = kept_dir / "index.json"
        if index_path.exists():
            index = read_json(index_path)
        else:
            index = {"kept_iterations": [], "cumulative_diff_path": "cumulative_changes.patch"}

        index["kept_iterations"].append({
            "iteration": iteration,
            "diff_file": diff_filename,
            "summary": change_summary,
            "timestamp": datetime.now().isoformat()
        })

        write_json(index_path, index)
        self.dbg(f"[iter {iteration}] Added to kept_changes: {diff_filename}")

        # Regenerate cumulative diff
        self.generate_cumulative_diff(kept_dir, index["kept_iterations"])

    def validation(self, iteration: int, iter_dir: Path) -> bool:
        """
        Run validation agent to check code quality before building.

        Catches syntax errors, JSON issues, schema inconsistencies, etc.
        Saves ~10 minutes per failed iteration by catching errors early.

        Args:
            iteration: Current iteration number
            iter_dir: Path to iteration directory

        Returns:
            True if validation passed, False otherwise (auto-reverts on failure)
        """
        validation_path = iter_dir / "validation.json"

        self.dbg(f"[iter {iteration}] Running validation...")

        # Prepare validation prompt
        prompt = self._render_prompt(
            self.cfg.repo_root / "agent_loop" / "prompts" / "validator.md",
            repo_root=self.cfg.repo_root,
            iteration=iteration,
            iteration_dir=iter_dir,
            validation_path=validation_path,
        )

        # Run validation agent
        self.run_agent(prompt, wait_for_file=validation_path)

        # Read validation result
        if not validation_path.exists():
            self.dbg(f"[iter {iteration}] ❌ Validation agent did not create validation.json")
            return False

        validation_result = read_json(validation_path)
        status = validation_result.get("status", "fail")
        can_proceed = validation_result.get("can_proceed", False)
        summary = validation_result.get("summary", "")

        self.dbg(f"[iter {iteration}] Validation result: status={status}, can_proceed={can_proceed}")

        if not can_proceed:
            errors = validation_result.get("errors", [])
            self.dbg(f"[iter {iteration}] ❌ Validation failed:")
            for error in errors:
                if error.get("severity") == "critical":
                    self.dbg(f"[iter {iteration}]   - {error.get('check')}: {error.get('message')}")

            # Auto-revert changes
            code_backup = self.run_dir / "backups" / f"iter_{iteration}_code_before"
            self.dbg(f"[iter {iteration}] Auto-reverting changes due to validation failure...")
            restore_paths(code_backup, self.cfg.repo_root)
            self.dbg(f"[iter {iteration}] Revert complete.")

            return False

        self.dbg(f"[iter {iteration}] ✅ Validation passed: {summary}")
        return True

    def loop(self) -> None:
        cfg = self.cfg
        self.dbg(f"Starting loop: N={cfg.n_iters}, model={cfg.model}, holdout_from_year={cfg.holdout_from_year}")
        run_config_path = self.run_dir / "run_config.json"
        if not run_config_path.exists():
            write_json(run_config_path, json.loads(json.dumps({
                "fight_url": cfg.fight_url,
                "n_iters": cfg.n_iters,
                "model": cfg.model,
                "holdout_from_year": cfg.holdout_from_year,
                "baseline_json": str(cfg.baseline_json),
                "train_model_name_prefix": cfg.train_model_name_prefix,
            })))
            self.dbg(f"Wrote run_config.json: {run_config_path}")

        history_path = self.run_dir / "history.json"
        if history_path.exists():
            hist = read_json(history_path)
            completed = len(hist.get("iterations", [])) if isinstance(hist, dict) else 0
            self.dbg(f"Resuming run_dir={self.run_dir} with {completed} completed iterations")
        else:
            write_json(history_path, {"iterations": []})
            self.dbg(f"Initialized history.json: {history_path}")

        # Context
        context_path = self.run_dir / "context.json"
        if context_path.exists():
            self.dbg(f"Reusing existing context.json: {context_path}")
        else:
            context_path = self.build_context()

        # Plan
        plan_path = self._resolve_latest_plan_path()
        if plan_path is None:
            plan_path = self.planning(context_path)
        else:
            self.dbg(f"Reusing latest plan: {plan_path}")

        # Paths to backup each iteration’s code changes (for revert)
        key_code_paths = [
            cfg.repo_root / "schema" / "feature_schema.json",
            cfg.repo_root / "schema" / "monotone_constraints.json",
            cfg.repo_root / "features",
        ]

        start_iter = self._next_iteration_index(history_path)
        end_iter = start_iter + cfg.n_iters - 1
        for i in range(start_iter, end_iter + 1):
            iter_dir = self.run_dir / f"iter_{i}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            self.dbg(f"=== Iteration {i}/{cfg.n_iters} ===")

            # Backup code paths before agent edits (enables revert without git)
            code_backup = self.run_dir / "backups" / f"iter_{i}_code_before"
            self.dbg(f"[iter {i}] Backing up code paths → {code_backup}")
            backup_paths(key_code_paths, code_backup)

            change_path = iter_dir / "change.json"
            self.dbg(f"[iter {i}] feature_creator expected to write: {change_path}")
            creator_prompt = self._render_prompt(
                cfg.repo_root / "agent_loop" / "prompts" / "feature_creator.md",
                plan_path=plan_path,
                context_path=context_path,
                history_path=history_path,
                change_path=change_path,
            )
            self.run_agent(creator_prompt, wait_for_file=change_path)
            if not change_path.exists():
                raise RuntimeError(f"feature_creator did not write change file: {change_path}")
            self.dbg(f"[iter {i}] feature_creator wrote change.json")

            # Generate diff of changes made by feature_creator
            self.dbg(f"[iter {i}] Generating code diff...")
            diff_path = self.generate_diff(i, code_backup)

            # Validation: check code quality before expensive build
            if not self.validation(i, iter_dir):
                # Validation failed, changes already reverted by validation()
                # Record failure in history and skip to next iteration
                hist = read_json(history_path)
                hist["iterations"].append(
                    {
                        "iteration": i,
                        "change_path": str(change_path),
                        "decision_path": str(iter_dir / "validation.json"),
                        "decision": "revert",
                        "reason": "validation_failed",
                        "diff_path": str(diff_path),
                    }
                )
                write_json(history_path, hist)

                # Create next plan to continue loop
                next_plan_path = iter_dir / "plan_next.json"
                self.dbg(f"[iter {i}] Creating next plan after validation failure...")
                # Reuse current plan for next iteration
                shutil.copy(str(plan_path), str(next_plan_path))
                plan_path = next_plan_path
                self.dbg(f"[iter {i}] Skipping to next iteration due to validation failure")
                continue

            # Build/train/eval (only if validation passed)
            eval_path = self.feature_builder(i)

            # Tester decision
            decision_path = iter_dir / "decision.json"
            next_plan_path = iter_dir / "plan_next.json"
            self.dbg(f"[iter {i}] tester expected to write: {decision_path} and {next_plan_path}")
            tester_prompt = self._render_prompt(
                cfg.repo_root / "agent_loop" / "prompts" / "tester.md",
                plan_path=plan_path,
                change_path=change_path,
                eval_path=eval_path,
                baseline_path=cfg.repo_root / cfg.baseline_json,
                history_path=history_path,
                next_plan_path=next_plan_path,
                decision_path=decision_path,
            )
            self.run_agent(tester_prompt, wait_for_file=decision_path)
            if not decision_path.exists() or not next_plan_path.exists():
                raise RuntimeError("tester agent did not write decision and next plan JSON")

            decision = read_json(decision_path)
            self.dbg(f"[iter {i}] tester decision: {decision.get('decision')}")
            # Update history
            hist = read_json(history_path)
            hist["iterations"].append(
                {
                    "iteration": i,
                    "change_path": str(change_path),
                    "eval_path": str(eval_path),
                    "decision_path": str(decision_path),
                    "decision": decision.get("decision"),
                    "diff_path": str(diff_path),
                }
            )
            write_json(history_path, hist)

            # Keep/revert behavior
            if decision.get("decision") == "revert":
                self.dbg(f"[iter {i}] Reverting via backup restore: {code_backup}")
                restore_paths(code_backup, cfg.repo_root)
                self.dbg(f"[iter {i}] Revert complete.")
            else:
                self.dbg(f"[iter {i}] Keeping change.")
                # Update kept_changes summary
                change_data = read_json(change_path)
                change_summary = change_data.get("summary", "No summary provided")
                self.update_kept_changes(i, "keep", change_summary, diff_path)

            # Advance plan
            plan_path = next_plan_path
            self.dbg(f"[iter {i}] Next plan: {plan_path}")

        # Summarize
        report_path = self.run_dir / "report.html"
        self.dbg(f"Summarizer writing report → {report_path}")
        summarizer_prompt = self._render_prompt(
            cfg.repo_root / "agent_loop" / "prompts" / "summarizer.md",
            run_dir=self.run_dir,
            baseline_path=cfg.repo_root / cfg.baseline_json,
            report_path=report_path,
        )
        self.run_agent(summarizer_prompt, wait_for_file=report_path)
        if not report_path.exists():
            raise RuntimeError("summarizer did not write report.html")
        self.dbg("Loop complete.")

    def _next_iteration_index(self, history_path: Path) -> int:
        if not history_path.exists():
            return 1
        hist = read_json(history_path)
        if not isinstance(hist, dict):
            return 1
        iters = hist.get("iterations") or []
        if not isinstance(iters, list) or not iters:
            return 1
        # Next iteration = max recorded iteration + 1
        try:
            return int(max(i.get("iteration", 0) for i in iters)) + 1
        except Exception:
            return len(iters) + 1

    def _resolve_latest_plan_path(self) -> Optional[Path]:
        """
        If resuming, prefer the most recent iter_*/plan_next.json.
        Fallback to run_dir/plan.json if present.
        """
        # Prefer latest plan_next.json
        iter_dirs = sorted([p for p in self.run_dir.glob("iter_*") if p.is_dir()])
        if iter_dirs:
            # iterate from newest to oldest
            for d in reversed(iter_dirs):
                p = d / "plan_next.json"
                if p.exists():
                    return p
        p0 = self.run_dir / "plan.json"
        return p0 if p0.exists() else None


