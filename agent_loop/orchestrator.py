from __future__ import annotations

import json
from dataclasses import dataclass
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
    ingest_event_for_fight,
    scrape_fight_details_json,
    run_xgboost_predict,
)


@dataclass(frozen=True)
class LoopConfig:
    repo_root: Path
    fight_url: str
    n_iters: int
    model: str = "gpt-5"
    agent_cmd: str = "agent"
    verbose: bool = False

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
    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
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
    def run_agent(self, prompt: str) -> None:
        """
        Run Cursor CLI agent in non-interactive mode.
        Uses: --print --output-format json to make it script-friendly.
        We still rely on the agent to write artifact files to disk.
        """
        cmd = [
            self.cfg.agent_cmd,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.cfg.model,
            "--workspace",
            str(self.cfg.repo_root),
            prompt,
        ]
        # Store raw agent output for debugging
        out_path = self.run_dir / "logs" / f"agent_{utc_timestamp()}.json"
        self.dbg(f"Running agent: model={self.cfg.model}, output={out_path}")
        run_cmd(cmd, cwd=self.cfg.repo_root, stdout_path=out_path, stderr_path=out_path.with_suffix(".stderr.txt"))
        self.dbg(f"Agent completed: {out_path}")

    def _render_prompt(self, template_path: Path, **kwargs) -> str:
        tpl = read_text(template_path)
        return tpl.format(**{k: str(v) for k, v in kwargs.items()})

    # -----------------------
    # Core steps
    # -----------------------
    def build_context(self) -> Path:
        """
        Ingest fight into DB, run xgboost_predict, and write a context.json for agents.
        """
        context_path = self.run_dir / "context.json"
        self.dbg(f"Run dir: {self.run_dir}")
        self.dbg(f"Building context for fight: {self.cfg.fight_url}")

        # 1) Ingest event containing this fight (ensures DB has it)
        self.dbg("Ingesting event into DB (includes fight stats + validation)…")
        event_url = ingest_event_for_fight(
            self.cfg.repo_root,
            self.cfg.fight_url,
            include_fight_stats=True,
            validate=True,
            validate_details=True,
        )
        self.dbg(f"Ingest complete. event_url={event_url}")

        # 2) Scrape fight details minimal metadata
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

        # 3) Run xgboost_predict for this matchup (store raw output)
        pred_out = self.run_dir / "xgboost_predict.txt"
        self.dbg(f"Running xgboost_predict → {pred_out}")
        run_xgboost_predict(
            self.cfg.repo_root,
            fighter_1_ufcstats_id=fighters[0]["ufcstats_id"],
            fighter_2_ufcstats_id=fighters[1]["ufcstats_id"],
            model_name=self.cfg.xgboost_predict_model_name,
            out_path=pred_out,
        )
        self.dbg("xgboost_predict complete.")

        context = {
            "fight_details_url": self.cfg.fight_url,
            "event_url": event_url,
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

    def loop(self) -> None:
        cfg = self.cfg
        self.dbg(f"Starting loop: N={cfg.n_iters}, model={cfg.model}, holdout_from_year={cfg.holdout_from_year}")
        write_json(self.run_dir / "run_config.json", json.loads(json.dumps({
            "fight_url": cfg.fight_url,
            "n_iters": cfg.n_iters,
            "model": cfg.model,
            "holdout_from_year": cfg.holdout_from_year,
            "baseline_json": str(cfg.baseline_json),
            "train_model_name_prefix": cfg.train_model_name_prefix,
        })))

        context_path = self.build_context()
        plan_path = self.planning(context_path)

        history_path = self.run_dir / "history.json"
        write_json(history_path, {"iterations": []})
        self.dbg(f"Initialized history.json: {history_path}")

        # Paths to backup each iteration’s code changes (for revert)
        key_code_paths = [
            cfg.repo_root / "schema" / "feature_schema.json",
            cfg.repo_root / "schema" / "monotone_constraints.json",
            cfg.repo_root / "features",
        ]

        for i in range(1, cfg.n_iters + 1):
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
            self.run_agent(creator_prompt)
            if not change_path.exists():
                raise RuntimeError(f"feature_creator did not write change file: {change_path}")
            self.dbg(f"[iter {i}] feature_creator wrote change.json")

            # Build/train/eval
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
            self.run_agent(tester_prompt)
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
        self.run_agent(summarizer_prompt)
        if not report_path.exists():
            raise RuntimeError("summarizer did not write report.html")
        self.dbg("Loop complete.")


