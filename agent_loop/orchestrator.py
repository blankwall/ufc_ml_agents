from __future__ import annotations

import csv
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

    def planning(self, context_path: Path, diagnostics_path: Optional[Path] = None) -> Path:
        plan_path = self.run_dir / "plan.json"
        self.dbg(f"Planning step: writing plan → {plan_path}")
        prompt = self._render_prompt(
            self.cfg.repo_root / "agent_loop" / "prompts" / "planning.md",
            context_path=context_path,
            diagnostics_path=diagnostics_path or "None",
            plan_path=plan_path,
            n_iters=self.cfg.n_iters,
        )
        self.run_agent(prompt)
        if not plan_path.exists():
            raise RuntimeError(f"Planning agent did not create plan file: {plan_path}")
        self.dbg("Planning complete.")
        return plan_path

    def introspection(self, model_name: Optional[str] = None, output_path: Optional[Path] = None) -> Path:
        """
        Run introspection analysis on repository and model.

        Provides comprehensive overview for manual debugging:
        - Feature landscape (count, categories, types)
        - Feature importance (top/bottom features)
        - Model performance (overall, segments, calibration)
        - Training data snapshot
        - Model configuration
        - Key insights and recommendations

        Args:
            model_name: Name of model to introspect (if None, uses baseline)
            output_path: Where to write the report (if None, uses run_dir/introspection.md)

        Returns:
            Path to the introspection report
        """
        if output_path is None:
            output_path = self.run_dir / "introspection.md"

        self.dbg(f"Running introspection (model={model_name or 'baseline'})...")

        # Start building the report
        report_lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        report_lines.append(f"# Model Introspection Report")
        report_lines.append("")
        report_lines.append(f"**Model**: `{model_name or 'baseline'}`")
        report_lines.append(f"**Generated**: {timestamp}")
        report_lines.append(f"**Repository**: `{self.cfg.repo_root}`")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # === Feature Landscape ===
        report_lines.append("## Feature Landscape")
        report_lines.append("")

        # First, try to get the ACTUAL features from training data
        training_data_path = self.cfg.repo_root / "data" / "processed" / "training_data.csv"
        actual_features = None
        actual_feature_count = 0

        if training_data_path.exists():
            try:
                import pandas as pd
                df_sample = pd.read_csv(training_data_path, nrows=1)
                # Exclude all non-feature columns (IDs, targets, metadata)
                non_feature_cols = [
                    'fighter1_id', 'fighter2_id', 'fighter_1_id', 'fighter_2_id',
                    'result', 'target', 'event_date', 'dataset',
                    'event_id', 'fight_id', 'method', 'weight_class'
                ]
                actual_features = [c for c in df_sample.columns if c not in non_feature_cols]
                actual_feature_count = len(actual_features)
            except Exception as e:
                self.dbg(f"Could not read training data: {e}")

        # Also read schema for comparison
        feature_schema_path = self.cfg.repo_root / "schema" / "feature_schema.json"
        schema_feature_count = None

        if feature_schema_path.exists():
            feature_schema = read_json(feature_schema_path)
            if isinstance(feature_schema, dict) and "num_features" in feature_schema:
                schema_feature_count = feature_schema["num_features"]

        # Report the findings
        report_lines.append("### Actual vs Theoretical Feature Counts")
        report_lines.append("")
        if actual_feature_count > 0:
            report_lines.append(f"- **Actual features in training data**: {actual_feature_count} ⭐ **(USE THIS NUMBER)**")
        if schema_feature_count and schema_feature_count != actual_feature_count:
            report_lines.append(f"- **Schema definition**: {schema_feature_count} features")
            report_lines.append(f"- **Gap**: {schema_feature_count - actual_feature_count} features in schema but not in training data")
        report_lines.append("")
        report_lines.append("**Note**: The schema file lists all theoretical features (auto-generated), but only")
        report_lines.append("features that are implemented, enabled, and successfully computed appear in training data.")
        report_lines.append("")

        # Show feature breakdown from actual data if available
        if actual_features:
            # Count by prefix
            f1_count = sum(1 for f in actual_features if f.startswith('f1_'))
            f2_count = sum(1 for f in actual_features if f.startswith('f2_'))
            diff_count = sum(1 for f in actual_features if f.endswith('_diff'))
            interaction_count = sum(1 for f in actual_features if '_x_' in f)

            report_lines.append("**Feature Breakdown** (from training data):")
            report_lines.append(f"- Fighter 1 features (f1_): {f1_count}")
            report_lines.append(f"- Fighter 2 features (f2_): {f2_count}")
            report_lines.append(f"- Difference features (_diff): {diff_count}")
            report_lines.append(f"- Interaction features (_x_): {interaction_count}")
            report_lines.append("")

            # Show first few features
            report_lines.append("**Sample Features** (first 15 from training data):")
            for feat in sorted(actual_features)[:15]:
                report_lines.append(f"- `{feat}`")
            if len(actual_features) > 15:
                report_lines.append(f"- ... and {len(actual_features) - 15} more")
            report_lines.append("")
        elif feature_schema_path.exists():
            # Fallback to schema if training data not available
            feature_schema = read_json(feature_schema_path)

            # Handle different schema structures
            if isinstance(feature_schema, dict):
                if "features" in feature_schema:
                    feature_list = feature_schema["features"]
                    num_features = feature_schema.get("num_features", len(feature_list))
                    version = feature_schema.get("version", "unknown")
                else:
                    feature_list = list(feature_schema.keys())
                    num_features = len(feature_list)
                    version = "legacy"
            elif isinstance(feature_schema, list):
                feature_list = feature_schema
                num_features = len(feature_list)
                version = "list"
            else:
                feature_list = []
                num_features = 0
                version = "unknown"

            # Count features by type based on naming patterns
            feature_types = {
                "interaction": 0,
                "ratio": 0,
                "polynomial": 0,
                "simple/aggregate": 0,
                "other": 0
            }

            for feat_name in feature_list:
                # Classify feature type by naming pattern
                if "_x_" in feat_name:
                    feature_types["interaction"] += 1
                elif "_ratio" in feat_name or "_over_" in feat_name or "_div_" in feat_name:
                    feature_types["ratio"] += 1
                elif "_squared" in feat_name or "_cubed" in feat_name:
                    feature_types["polynomial"] += 1
                elif any(x in feat_name for x in ["_avg", "_sum", "_count", "_total", "_diff", "_delta"]):
                    feature_types["simple/aggregate"] += 1
                else:
                    feature_types["other"] += 1

            total_features = num_features
            report_lines.append(f"**Total Features** (from schema): {total_features}")
            report_lines.append(f"**Schema Version**: {version}")
            report_lines.append("")
            report_lines.append("⚠️ *Using schema data - training data not available*")
            report_lines.append("")

            # By type
            report_lines.append("**Feature Types** (by naming pattern):")
            for ftype, count in feature_types.items():
                if count > 0:
                    pct = (count / total_features * 100) if total_features > 0 else 0
                    report_lines.append(f"- {ftype.title()}: {count} ({pct:.1f}%)")
            report_lines.append("")

            # Show first few features as examples
            report_lines.append("**Sample Features** (first 10 from schema):")
            for feat in feature_list[:10]:
                report_lines.append(f"- `{feat}`")
            if len(feature_list) > 10:
                report_lines.append(f"- ... and {len(feature_list) - 10} more")
            report_lines.append("")
        else:
            report_lines.append("⚠️ No feature schema found")
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

        # === Feature Importance ===
        report_lines.append("## Feature Importance")
        report_lines.append("")

        # Find feature importance file
        importance_path = None
        if model_name:
            # Try model-specific importance
            for ext in ["_feature_importance.csv", "_importance.csv"]:
                candidate = self.cfg.repo_root / "models" / "saved" / f"{model_name}{ext}"
                if candidate.exists():
                    importance_path = candidate
                    break

        # Fallback to most recent importance file
        if not importance_path:
            models_dir = self.cfg.repo_root / "models" / "saved"
            if models_dir.exists():
                importance_files = list(models_dir.glob("*feature_importance.csv"))
                if importance_files:
                    importance_path = max(importance_files, key=lambda p: p.stat().st_mtime)

        if importance_path:
            report_lines.append(f"**Source**: `{importance_path.relative_to(self.cfg.repo_root)}`")
            report_lines.append("")

            # Read and parse importance
            features_with_importance = []
            with open(importance_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    feat = row.get('feature', row.get('name', ''))
                    gain = float(row.get('gain', row.get('importance', 0)))
                    features_with_importance.append((feat, gain))

            # Sort by gain
            features_with_importance.sort(key=lambda x: x[1], reverse=True)

            if features_with_importance:
                # Top 10
                report_lines.append("### Top 10 Features")
                report_lines.append("")
                report_lines.append("| Rank | Feature | Gain |")
                report_lines.append("|------|---------|------|")
                for i, (feat, gain) in enumerate(features_with_importance[:10], 1):
                    report_lines.append(f"| {i} | `{feat}` | {gain:.2f} |")
                report_lines.append("")

                # Bottom features (near zero)
                bottom_features = [(f, g) for f, g in features_with_importance if g < 0.01]
                if bottom_features:
                    report_lines.append(f"### Near-Zero Importance Features ({len(bottom_features)} features)")
                    report_lines.append("")
                    report_lines.append("These features contribute very little to predictions:")
                    report_lines.append("")
                    for feat, gain in bottom_features[:20]:  # Show first 20
                        report_lines.append(f"- `{feat}`: {gain:.4f}")
                    if len(bottom_features) > 20:
                        report_lines.append(f"- ... and {len(bottom_features) - 20} more")
                    report_lines.append("")

                # Feature concentration
                total_gain = sum(g for _, g in features_with_importance)
                top_10_gain = sum(g for _, g in features_with_importance[:10])
                concentration = (top_10_gain / total_gain * 100) if total_gain > 0 else 0
                report_lines.append(f"**Feature Concentration**: Top 10 features account for {concentration:.1f}% of total gain")
                report_lines.append("")
        else:
            report_lines.append("⚠️ No feature importance file found")
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

        # === Model Performance ===
        report_lines.append("## Model Performance")
        report_lines.append("")

        # Find model evaluation - priority order:
        # 1. Model-specific evaluation in reports_strict/
        # 2. Latest model_eval_*.json in reports_strict/
        # 3. models/baseline.json (fallback)
        eval_path = None

        if model_name:
            # Try to find model-specific evaluation
            patterns = [
                f"model_eval_{model_name}*.json",  # Exact model name match
                f"{model_name}_eval.json",
                f"{model_name}_metrics.json",
            ]
            for pattern in patterns:
                matches = list((self.cfg.repo_root / "reports_strict").glob(pattern))
                if matches:
                    eval_path = max(matches, key=lambda p: p.stat().st_mtime)
                    self.dbg(f"Found model-specific eval: {eval_path.name}")
                    break

        # If no model-specific eval, try latest model_eval
        if not eval_path:
            matches = list((self.cfg.repo_root / "reports_strict").glob("model_eval_*.json"))
            if matches:
                eval_path = max(matches, key=lambda p: p.stat().st_mtime)
                self.dbg(f"Found latest eval: {eval_path.name}")

        # Last resort: use baseline config (may be stale)
        if not eval_path:
            eval_path = self.cfg.repo_root / self.cfg.baseline_json
            self.dbg(f"Using baseline config (may be stale): {eval_path}")

        if eval_path and eval_path.exists():
            eval_data = read_json(eval_path)

            # Overall metrics
            report_lines.append("### Overall Metrics")
            report_lines.append("")
            if "overall" in eval_data:
                overall = eval_data["overall"]
                report_lines.append(f"- **Accuracy**: {overall.get('accuracy', 'N/A'):.4f}" if isinstance(overall.get('accuracy'), (int, float)) else f"- **Accuracy**: {overall.get('accuracy', 'N/A')}")
                report_lines.append(f"- **Brier Score**: {overall.get('brier', 'N/A'):.4f}" if isinstance(overall.get('brier'), (int, float)) else f"- **Brier Score**: {overall.get('brier', 'N/A')}")
                report_lines.append(f"- **AUC**: {overall.get('auc', 'N/A'):.4f}" if isinstance(overall.get('auc'), (int, float)) else f"- **AUC**: {overall.get('auc', 'N/A')}")
                report_lines.append(f"- **Log Loss**: {overall.get('log_loss', 'N/A'):.4f}" if isinstance(overall.get('log_loss'), (int, float)) else f"- **Log Loss**: {overall.get('log_loss', 'N/A')}")
                report_lines.append("")

            # Top 25% by confidence
            if "by_confidence" in eval_data and "top_25_pct" in eval_data["by_confidence"]:
                report_lines.append("### Top 25% Confidence Performance")
                report_lines.append("")
                top_25 = eval_data["by_confidence"]["top_25_pct"]
                report_lines.append(f"- **Accuracy**: {top_25.get('accuracy', 'N/A'):.4f}" if isinstance(top_25.get('accuracy'), (int, float)) else f"- **Accuracy**: {top_25.get('accuracy', 'N/A')}")
                report_lines.append(f"- **Min Confidence**: {top_25.get('min_p', 'N/A'):.4f}" if isinstance(top_25.get('min_p'), (int, float)) else f"- **Min Confidence**: {top_25.get('min_p', 'N/A')}")
                report_lines.append(f"- **Fights**: {top_25.get('n', 'N/A')}")
                report_lines.append("")

            # Underdog performance (favorites vs underdogs)
            if "by_bucket" in eval_data:
                report_lines.append("### Favorites vs Underdogs")
                report_lines.append("")
                by_bucket = eval_data["by_bucket"]
                if "favorites" in by_bucket:
                    fav = by_bucket["favorites"]
                    report_lines.append(f"- **Favorites Accuracy**: {fav.get('accuracy', 'N/A'):.4f}" if isinstance(fav.get('accuracy'), (int, float)) else f"- **Favorites Accuracy**: {fav.get('accuracy', 'N/A')}")
                    report_lines.append(f"- **Favorites Count**: {fav.get('n', 'N/A')}")
                    report_lines.append("")

                if "underdogs" in by_bucket:
                    under = by_bucket["underdogs"]
                    report_lines.append(f"- **Underdogs Accuracy**: {under.get('accuracy', 'N/A'):.4f}" if isinstance(under.get('accuracy'), (int, float)) else f"- **Underdogs Accuracy**: {under.get('accuracy', 'N/A')}")
                    report_lines.append(f"- **Underdogs Count**: {under.get('n', 'N/A')}")
                    report_lines.append("")

            # ROI data
            if "roi_flat_edge_gt_0" in eval_data:
                roi = eval_data["roi_flat_edge_gt_0"]
                report_lines.append(f"- **Overall ROI (flat bet, edge > 0)**: {roi:.4f}" if isinstance(roi, (int, float)) else f"- **Overall ROI (flat bet, edge > 0)**: {roi}")
                report_lines.append("")

            # Underdog odds bands (if available)
            if "underdog_odds_bands" in eval_data:
                report_lines.append("### Underdog Performance by Odds Band")
                report_lines.append("")
                report_lines.append("| Odds Band | Accuracy | ROI | Bets |")
                report_lines.append("|-----------|----------|-----|------|")
                for band, data in sorted(eval_data["underdog_odds_bands"].items()):
                    acc = data.get('accuracy', 'N/A')
                    roi_val = data.get('roi', 'N/A')
                    bets = data.get('n_bets', 'N/A')
                    if isinstance(acc, float):
                        report_lines.append(f"| {band} | {acc:.3f} | {roi_val:.2f} | {bets} |")
                    else:
                        report_lines.append(f"| {band} | {acc} | {roi_val} | {bets} |")
                report_lines.append("")

            # Underdog confidence breakdown (if available)
            if "underdog_confidence_breakdown" in eval_data:
                report_lines.append("### Underdog Performance by Confidence")
                report_lines.append("")
                report_lines.append("| Confidence | Accuracy | ROI | Bets |")
                report_lines.append("|------------|----------|-----|------|")
                for conf, data in sorted(eval_data["underdog_confidence_breakdown"].items()):
                    acc = data.get('accuracy', 'N/A')
                    roi_val = data.get('roi', 'N/A')
                    bets = data.get('n_bets', 'N/A')
                    if isinstance(acc, float):
                        report_lines.append(f"| {conf} | {acc:.3f} | {roi_val:.2f} | {bets} |")
                    else:
                        report_lines.append(f"| {conf} | {acc} | {roi_val} | {bets} |")
                report_lines.append("")
        else:
            report_lines.append(f"⚠️ No evaluation data found (looked for: {eval_path})")
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

        # === Key Insights ===
        report_lines.append("## Key Insights")
        report_lines.append("")

        insights = []

        # Use actual feature count for insights
        effective_feature_count = actual_feature_count if actual_feature_count > 0 else (schema_feature_count if schema_feature_count else 0)

        # Feature complexity insight
        if effective_feature_count > 0:
            if effective_feature_count > 200:
                insights.append(f"**High feature count**: {effective_feature_count} features - consider feature selection to reduce noise")
            elif effective_feature_count < 30:
                insights.append(f"**Low feature count**: {effective_feature_count} features - may be missing important signals")
            else:
                insights.append(f"**Feature count**: {effective_feature_count} features - balanced complexity")

        # Schema vs actual gap
        if actual_feature_count > 0 and schema_feature_count and schema_feature_count != actual_feature_count:
            gap = schema_feature_count - actual_feature_count
            gap_pct = (gap / schema_feature_count * 100) if schema_feature_count > 0 else 0
            insights.append(f"**Schema vs reality gap**: {gap} features ({gap_pct:.0f}%) in schema but not in training data - schema may be auto-generating theoretical features")

        # Importance concentration insight
        if importance_path and features_with_importance:
            if concentration > 70:
                insights.append(f"**High feature concentration**: Top 10 features drive {concentration:.0f}% of predictions - model relies heavily on few features")
            elif concentration < 40:
                insights.append(f"**Distributed feature importance**: Top 10 features only {concentration:.0f}% of gain - signal is well distributed")
            else:
                insights.append(f"**Moderate feature concentration**: Top 10 features account for {concentration:.0f}% of gain - good balance")

        # Low importance features
        if importance_path and bottom_features and effective_feature_count > 0:
            if len(bottom_features) > effective_feature_count * 0.2:
                insights.append(f"**Many low-importance features**: {len(bottom_features)} features ({len(bottom_features)/effective_feature_count*100:.1f}%) with < 0.01 gain - candidates for removal")

        # Model performance insights
        if eval_path and eval_path.exists():
            eval_data = read_json(eval_path)
            if "roi_flat_edge_gt_0" in eval_data:
                roi = eval_data["roi_flat_edge_gt_0"]
                if roi < -0.05:
                    insights.append(f"**Negative ROI**: {roi:.1%} - betting predictions are losing money")
                elif roi > 0.02:
                    insights.append(f"**Positive ROI**: {roi:.1%} - betting predictions are profitable")

            if "overall" in eval_data and isinstance(eval_data["overall"].get("accuracy"), (int, float)):
                acc = eval_data["overall"]["accuracy"]
                if acc > 0.70:
                    insights.append(f"**Strong accuracy**: {acc:.2%} - model predictions are reliable")
                elif acc < 0.60:
                    insights.append(f"Weak accuracy: {acc:.2%} - model needs improvement")

        if not insights:
            insights.append("No major concerns identified from available data")

        for i, insight in enumerate(insights, 1):
            report_lines.append(f"{i}. {insight}")
        report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

        # === Recommendations ===
        report_lines.append("## Recommendations")
        report_lines.append("")

        recommendations = []

        if importance_path and bottom_features and len(bottom_features) > 10:
            recommendations.append(f"**Remove low-importance features**: {len(bottom_features)} features with near-zero importance could be removed to reduce noise")

        if eval_path and eval_path.exists():
            eval_data = read_json(eval_path)
            if "underdog" in eval_data and eval_data["underdog"].get("roi", 0) < -0.05:
                recommendations.append("**Focus on underdog calibration**: Underdog performance is the top priority - consider adding underdog-specific features or calibration adjustments")

        if effective_feature_count > 150:
            recommendations.append("**Consider feature selection**: High feature count may lead to overfitting - use diagnostics to identify redundant features")

        if not recommendations:
            recommendations.append("Continue monitoring model performance and feature importance on each iteration")

        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines), encoding="utf-8")

        self.dbg(f"Introspection report written to: {output_path}")
        return output_path

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

    def evaluation_phase(self, iteration: int, iter_dir: Path, change_path: Path,
                        eval_path: Path, plan_path: Path, history_path: Path,
                        diff_path: Path, code_backup: Path) -> str:
        """
        Run evaluator and decision agents for post-build analysis.

        Splits the old tester agent into two phases:
        1. Evaluator: Pure metrics analysis (objective)
        2. Decision: Strategic keep/revert choice

        Args:
            iteration: Current iteration number
            iter_dir: Path to iteration directory
            change_path: Path to change.json
            eval_path: Path to evaluation JSON
            plan_path: Path to current plan
            history_path: Path to history JSON
            diff_path: Path to code diff
            code_backup: Path to code backup for revert

        Returns:
            Decision: "keep" or "revert"
        """
        # Phase 1: Run evaluator (objective analysis)
        analysis_path = iter_dir / "analysis.json"
        self.dbg(f"[iter {iteration}] Running evaluator (objective analysis)...")

        evaluator_prompt = self._render_prompt(
            self.cfg.repo_root / "agent_loop" / "prompts" / "evaluator.md",
            eval_path=eval_path,
            baseline_path=self.cfg.repo_root / self.cfg.baseline_json,
            change_path=change_path,
            analysis_path=analysis_path,
        )
        self.run_agent(evaluator_prompt, wait_for_file=analysis_path)
        if not analysis_path.exists():
            raise RuntimeError(f"Evaluator did not create analysis file: {analysis_path}")

        # Phase 2: Run decision agent (strategy)
        decision_path = iter_dir / "decision.json"
        next_plan_path = iter_dir / "plan_next.json"
        self.dbg(f"[iter {iteration}] Running decision agent (strategy)...")

        decision_prompt = self._render_prompt(
            self.cfg.repo_root / "agent_loop" / "prompts" / "decision.md",
            analysis_path=analysis_path,
            change_path=change_path,
            plan_path=plan_path,
            history_path=history_path,
            decision_path=decision_path,
            next_plan_path=next_plan_path,
        )
        self.run_agent(decision_prompt, wait_for_file=decision_path)
        if not decision_path.exists() or not next_plan_path.exists():
            raise RuntimeError("Decision agent did not write decision and next plan JSON")

        decision_data = read_json(decision_path)
        decision = decision_data.get("decision")
        self.dbg(f"[iter {iteration}] Decision: {decision}")

        # Update history
        hist = read_json(history_path)
        hist["iterations"].append(
            {
                "iteration": iteration,
                "change_path": str(change_path),
                "eval_path": str(eval_path),
                "analysis_path": str(analysis_path),
                "decision_path": str(decision_path),
                "decision": decision,
                "diff_path": str(diff_path),
            }
        )
        write_json(history_path, hist)

        # Keep/revert behavior
        if decision == "revert":
            self.dbg(f"[iter {iteration}] Reverting via backup restore: {code_backup}")
            restore_paths(code_backup, self.cfg.repo_root)
            self.dbg(f"[iter {iteration}] Revert complete.")
        else:
            self.dbg(f"[iter {iteration}] Keeping change.")
            # Update kept_changes summary
            change_data = read_json(change_path)
            change_summary = change_data.get("summary", "No summary provided")
            self.update_kept_changes(iteration, "keep", change_summary, diff_path)

        # Return next plan path for loop to use
        return next_plan_path

    def run_diagnostics(self) -> Path:
        """
        Run diagnostics agent to analyze current model weaknesses.

        Should be run before planning to provide data-driven insights.

        Returns:
            Path to diagnostics JSON output
        """
        diagnostics_path = self.run_dir / "diagnostics.json"

        # Check if we already have diagnostics from a previous run
        if diagnostics_path.exists():
            self.dbg(f"Reusing existing diagnostics: {diagnostics_path}")
            return diagnostics_path

        self.dbg("Running diagnostics to analyze current model...")

        # Find baseline evaluation and feature importance files
        baseline_eval = self.cfg.repo_root / self.cfg.baseline_json
        if not baseline_eval.exists():
            self.dbg(f"Warning: Baseline eval not found at {baseline_eval}")
            # Create a placeholder diagnostics with missing data info
            placeholder = {
                "weaknesses": [],
                "redundant_features": [],
                "low_importance_features": [],
                "top_features": [],
                "opportunities": [],
                "calibration_issues": [],
                "summary": "No baseline data available - diagnostics will be updated after first model evaluation",
                "data_available": False
            }
            write_json(diagnostics_path, placeholder)
            return diagnostics_path

        # Try to find feature importance from the most recent model
        # Check models/saved/ directory for the baseline model
        feature_importance = None
        models_dir = self.cfg.repo_root / "models" / "saved"
        if models_dir.exists():
            # Look for most recent feature importance CSV
            importance_files = list(models_dir.glob("*feature_importance.csv"))
            if importance_files:
                feature_importance = max(importance_files, key=lambda p: p.stat().st_mtime)
                self.dbg(f"Found feature importance: {feature_importance.name}")

        # Prepare diagnostics prompt
        prompt = self._render_prompt(
            self.cfg.repo_root / "agent_loop" / "prompts" / "diagnostics.md",
            baseline_eval_path=baseline_eval,
            feature_importance_path=feature_importance or "NOT_FOUND",
            baseline_metrics=self.cfg.baseline_json,
            diagnostics_path=diagnostics_path,
        )

        # Run diagnostics agent
        self.run_agent(prompt, wait_for_file=diagnostics_path)

        if not diagnostics_path.exists():
            self.dbg("Warning: Diagnostics agent did not create diagnostics.json")
            # Create placeholder
            placeholder = {
                "weaknesses": [],
                "redundant_features": [],
                "low_importance_features": [],
                "top_features": [],
                "opportunities": [],
                "calibration_issues": [],
                "summary": "Diagnostics generation failed - proceeding without diagnostic insights",
                "data_available": False
            }
            write_json(diagnostics_path, placeholder)

        self.dbg(f"Diagnostics complete: {diagnostics_path}")
        return diagnostics_path

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

        # Diagnostics: analyze current model before planning
        diagnostics_path = self.run_dir / "diagnostics.json"
        if not diagnostics_path.exists():
            self.dbg("Running diagnostics to inform planning...")
            diagnostics_path = self.run_diagnostics()
        else:
            self.dbg(f"Reusing existing diagnostics: {diagnostics_path}")

        # Plan (informed by diagnostics)
        plan_path = self._resolve_latest_plan_path()
        if plan_path is None:
            plan_path = self.planning(context_path, diagnostics_path=diagnostics_path)
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

            # Evaluation phase: evaluator (analysis) + decision (strategy)
            next_plan_path = self.evaluation_phase(
                iteration=i,
                iter_dir=iter_dir,
                change_path=change_path,
                eval_path=eval_path,
                plan_path=plan_path,
                history_path=history_path,
                diff_path=diff_path,
                code_backup=code_backup,
            )

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


