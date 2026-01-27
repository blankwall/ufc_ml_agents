#!/usr/bin/env python3
"""
Standalone script to test the tester agent in isolation.
Usage: python3 agent_loop/test_tester.py <run_timestamp>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_loop.utils import read_text, write_text
from agent_loop.orchestrator import AgentLoop, LoopConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 agent_loop/test_tester.py <run_timestamp>")
        print("Example: python3 agent_loop/test_tester.py 20260126_200944")
        sys.exit(1)
    
    run_ts = sys.argv[1]
    repo_root = PROJECT_ROOT
    run_dir = repo_root / "agent_loop" / "agent_artifacts" / run_ts
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Check required files exist
    required = {
        "plan": run_dir / "plan.json",
        "change": run_dir / "iter_1" / "change.json",
        "eval": run_dir / "iter_1_model_eval.json",
        "baseline": repo_root / "models" / "baseline.json",
        "history": run_dir / "history.json",
    }
    
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        print(f"Error: Missing required files: {missing}")
        for k in missing:
            print(f"  - {k}: {required[k]}")
        sys.exit(1)
    
    # Render tester prompt
    tester_template = repo_root / "agent_loop" / "prompts" / "tester.md"
    tpl = read_text(tester_template)
    
    prompt = tpl.format(
        plan_path=required["plan"],
        change_path=required["change"],
        eval_path=required["eval"],
        baseline_path=required["baseline"],
        history_path=required["history"],
        next_plan_path=run_dir / "iter_1" / "plan_next.json",
        decision_path=run_dir / "iter_1" / "decision.json",
    )
    
    # Write prompt to file
    prompt_file = run_dir / "logs" / "test_tester_prompt.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    write_text(prompt_file, prompt)
    
    print(f"✓ Prompt generated: {prompt_file}")
    print(f"\nRun the tester agent with:\n")
    print(f"  agent --print --output-format json --model auto < {prompt_file}")
    print(f"\nOr pass it as an argument (may need to escape quotes):\n")
    print(f"  agent --print --output-format json --model auto \"$(cat {prompt_file})\"")
    print(f"\nExpected outputs:")
    print(f"  - {run_dir / 'iter_1' / 'decision.json'}")
    print(f"  - {run_dir / 'iter_1' / 'plan_next.json'}")

if __name__ == "__main__":
    main()

