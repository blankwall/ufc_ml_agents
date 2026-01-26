from __future__ import annotations

import argparse
from pathlib import Path

from agent_loop.orchestrator import AgentLoop, LoopConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Run the agentic optimization loop.")
    p.add_argument("--fight-url", type=str, required=True, help="UFCStats fight-details URL")
    p.add_argument("--n", type=int, default=3, help="Number of iterations")
    p.add_argument("--model", type=str, default="gpt-5", help="Cursor agent model (e.g., gpt-5)")
    p.add_argument("--agent-cmd", type=str, default="agent", help="Agent command (default: agent)")
    p.add_argument("--verbose", action="store_true", help="Print step-by-step progress (debug flow)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress printing (overrides --verbose)")

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

    cfg = LoopConfig(
        repo_root=repo_root,
        fight_url=args.fight_url,
        n_iters=int(args.n),
        model=args.model,
        agent_cmd=args.agent_cmd,
        verbose=bool(args.verbose) and not bool(args.quiet),
        holdout_from_year=int(args.holdout_from_year),
        baseline_json=Path(args.baseline_json),
        train_model_name_prefix=args.train_model_name_prefix,
        xgboost_predict_model_name=args.xgboost_predict_model_name,
        eval_min_year=int(args.eval_min_year),
        odds_path=Path(args.odds_path),
        odds_date_tolerance_days=int(args.odds_date_tolerance_days),
    )

    AgentLoop(cfg).loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


