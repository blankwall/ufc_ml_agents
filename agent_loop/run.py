from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from agent_loop.orchestrator import AgentLoop, LoopConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Run the agentic optimization loop.")
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fight-url", type=str, help="UFCStats fight-details URL")
    input_group.add_argument("--goal", type=str, help="Generic optimization goal text (non-fight mode)")
    input_group.add_argument("--goal-file", type=str, help="Path to a text/markdown file containing the goal (non-fight mode)")
    input_group.add_argument("--introspect", type=str, nargs="?", const="", metavar="MODEL_NAME",
                           help="Introspection mode: analyze repository and model state. Optionally specify model name (default: baseline)")

    p.add_argument("--n", type=int, default=3, help="Number of iterations")
    p.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929", help="Claude model (e.g., claude-sonnet-4-5-20250929)")
    p.add_argument("--agent-cmd", type=str, default="claude", help="Agent command (default: claude)")
    p.add_argument("--verbose", action="store_true", help="Print step-by-step progress (debug flow)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress printing (overrides --verbose)")
    p.add_argument(
        "--manual",
        action="store_true",
        help="Manual mode: pause for each agent step so you can run them in Claude Code IDE instead of the CLI binary",
    )
    p.add_argument(
        "--resume-run",
        type=str,
        default=None,
        help="Resume an existing run by timestamp (e.g. 20260126_172555) or by full path to the run dir",
    )
    p.add_argument(
        "--fork-run",
        type=str,
        default=None,
        help="Fork an existing run (timestamp or path) into a new run dir and continue from its latest plan",
    )
    p.add_argument("--introspect-output", type=str, help="Custom output path for introspection report (default: agent_loop/agent_artifacts/<timestamp>/introspection.md)")

    p.add_argument("--holdout-from-year", type=int, default=2025)
    p.add_argument("--baseline-json", type=str, default="models/baseline.json")
    p.add_argument("--train-model-name-prefix", type=str, default="agent_loop_model")
    p.add_argument(
        "--xgboost-predict-model-name",
        type=str,
        default="baseline_jan_11_2026_age_feature_add_striking_landed",
        help="Model name to use for xgboost_predict context step",
    )
    p.add_argument("--eval-min-year", type=int, default=2025)
    p.add_argument("--odds-path", type=str, default="ufc_2025_odds.csv")
    p.add_argument("--odds-date-tolerance-days", type=int, default=5)

    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Handle introspection mode (separate from normal loop)
    if args.introspect is not None:
        from agent_loop.utils import utc_timestamp

        # Create minimal config for introspection
        model_name = args.introspect if args.introspect else None
        output_path = Path(args.introspect_output) if args.introspect_output else None

        # Create a temporary run dir for introspection output
        if output_path is None:
            introspection_run_dir = repo_root / "agent_loop" / "agent_artifacts" / utc_timestamp()
            introspection_run_dir.mkdir(parents=True, exist_ok=True)
            output_path = introspection_run_dir / "introspection.md"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a minimal LoopConfig (not used for loop, just for AgentLoop instantiation)
        cfg = LoopConfig(
            repo_root=repo_root,
            fight_url=None,
            goal_text="Introspection mode",
            n_iters=0,
            model=args.model,
            agent_cmd=args.agent_cmd,
            verbose=True,  # Always verbose for introspection
            manual=False,
            holdout_from_year=int(args.holdout_from_year),
            baseline_json=Path(args.baseline_json),
        )

        # Run introspection
        loop = AgentLoop(cfg)
        result_path = loop.introspection(model_name=model_name, output_path=output_path)

        print(f"\n✅ Introspection complete!")
        print(f"📄 Report: {result_path}")
        print(f"\nTo view the report:")
        print(f"  cat {result_path}")
        if "EDITOR" in os.environ:
            print(f"  # Or open in your editor:")
            print(f"  $EDITOR {result_path}")

        return 0

    goal_text = None
    if args.goal_file:
        goal_text = Path(args.goal_file).read_text(encoding="utf-8")
    elif args.goal:
        goal_text = args.goal

    cfg = LoopConfig(
        repo_root=repo_root,
        fight_url=args.fight_url,
        goal_text=goal_text,
        n_iters=int(args.n),
        model=args.model,
        agent_cmd=args.agent_cmd,
        verbose=bool(args.verbose) and not bool(args.quiet),
        manual=bool(args.manual),
        holdout_from_year=int(args.holdout_from_year),
        baseline_json=Path(args.baseline_json),
        train_model_name_prefix=args.train_model_name_prefix,
        xgboost_predict_model_name=args.xgboost_predict_model_name,
        eval_min_year=int(args.eval_min_year),
        odds_path=Path(args.odds_path),
        odds_date_tolerance_days=int(args.odds_date_tolerance_days),
    )

    def _resolve_run_dir(run_arg: str) -> Path:
        maybe = Path(run_arg)
        if maybe.exists():
            return maybe
        return repo_root / "agent_loop" / "agent_artifacts" / run_arg

    run_dir = None

    # Fork has priority: it creates a new run directory seeded from an existing run.
    if args.fork_run:
        seed_dir = _resolve_run_dir(args.fork_run)
        if not seed_dir.exists():
            raise SystemExit(f"--fork-run not found: {seed_dir}")

        # Mirror orchestrator timestamp format from utils.
        from agent_loop.utils import utc_timestamp

        new_dir = repo_root / "agent_loop" / "agent_artifacts" / utc_timestamp()
        new_dir.mkdir(parents=True, exist_ok=True)

        # Copy context if present
        seed_context = seed_dir / "context.json"
        if seed_context.exists():
            shutil.copy2(seed_context, new_dir / "context.json")

        # Copy latest plan (prefer latest iter_*/plan_next.json)
        latest_plan = None
        iter_dirs = sorted([p for p in seed_dir.glob("iter_*") if p.is_dir()])
        for d in reversed(iter_dirs):
            pth = d / "plan_next.json"
            if pth.exists():
                latest_plan = pth
                break
        if latest_plan is None:
            p0 = seed_dir / "plan.json"
            latest_plan = p0 if p0.exists() else None

        if latest_plan is not None and latest_plan.exists():
            shutil.copy2(latest_plan, new_dir / "plan.json")

        # Seed history: keep provenance and snapshot of prior history, but start new iterations list.
        seed_history = seed_dir / "history.json"
        if seed_history.exists():
            shutil.copy2(seed_history, new_dir / "seed_history.json")

        import json
        provenance = {
            "seed_run_dir": str(seed_dir),
            "seed_plan_path": str(latest_plan) if latest_plan else None,
        }
        (new_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        run_dir = new_dir

    elif args.resume_run:
        run_dir = _resolve_run_dir(args.resume_run)

    AgentLoop(cfg, run_dir=run_dir).loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


