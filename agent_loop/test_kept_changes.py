#!/usr/bin/env python3
"""
Test script for Item 2: Kept Changes functionality

This simulates the agent loop behavior to verify:
1. kept_changes directory is created
2. Diffs are copied with descriptive names
3. index.json is updated correctly
4. cumulative_changes.patch combines all kept diffs
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def create_test_diff(path: Path, iteration: int, content: str) -> None:
    """Create a fake diff file for testing"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Code Diff for Iteration {iteration}\n"
                    f"# Generated: {datetime.now().isoformat()}\n\n"
                    f"{content}\n")


def create_test_change_json(path: Path, summary: str) -> None:
    """Create a fake change.json for testing"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "changes": ["test_change_1", "test_change_2"],
        "feature_name": "test_feature"
    }, indent=2))


def test_kept_changes():
    """Test the kept_changes functionality"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from agent_loop.orchestrator import AgentLoop, LoopConfig

    print("="*60)
    print("Testing Item 2: Kept Changes Functionality")
    print("="*60)

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal config
        cfg = LoopConfig(
            repo_root=tmpdir,
            fight_url=None,
            n_iters=3,
            goal_text="Test goal",
            agent_cmd="echo",  # Dummy command
        )

        # Create agent loop instance
        loop = AgentLoop(cfg)
        print(f"\n✓ Created test AgentLoop with run_dir: {loop.run_dir}")

        # Simulate iterations
        test_iterations = [
            {"iter": 1, "summary": "removed_redundant_age_features", "diff_content": "=== schema/feature_schema.json ===\n- old_feature\n+ new_feature"},
            {"iter": 2, "summary": "calibrated_opponent_quality", "diff_content": "=== features/opponent_quality.py ===\n+ def calibrated_feature():\n+     return x * 0.95"},
            {"iter": 3, "summary": "added_strike_defense_metrics", "diff_content": "=== features/striking.py ===\n+ strike_defense_ratio"},
        ]

        print("\nSimulating 3 iterations...")

        for test_case in test_iterations:
            i = test_case["iter"]
            summary = test_case["summary"]
            diff_content = test_case["diff_content"]

            # Create fake diff and change.json
            diff_path = loop.run_dir / f"iter_{i}" / "code_diff.patch"
            change_path = loop.run_dir / f"iter_{i}" / "change.json"

            create_test_diff(diff_path, i, diff_content)
            create_test_change_json(change_path, summary)

            print(f"\n  Iteration {i}:")
            print(f"    - Created diff: {diff_path}")
            print(f"    - Created change.json: {change_path}")

            # Test update_kept_changes
            loop.update_kept_changes(i, "keep", summary, diff_path)

            # Verify kept_changes directory
            kept_dir = loop.run_dir / "kept_changes"
            assert kept_dir.exists(), f"kept_changes directory not created for iter {i}"
            print(f"    ✓ kept_changes directory exists")

            # Verify diff was copied
            expected_patch = kept_dir / f"iter_{i}_{summary}.patch"
            assert expected_patch.exists(), f"Diff not copied: {expected_patch}"
            print(f"    ✓ Diff copied: {expected_patch.name}")

            # Verify index.json
            index_path = kept_dir / "index.json"
            assert index_path.exists(), "index.json not created"
            index = json.loads(index_path.read_text())
            assert len(index["kept_iterations"]) == i, f"Expected {i} iterations, got {len(index['kept_iterations'])}"
            print(f"    ✓ index.json has {i} iteration(s)")

            # Verify cumulative diff
            cumulative_path = kept_dir / "cumulative_changes.patch"
            assert cumulative_path.exists(), "cumulative_changes.patch not created"
            cumulative = cumulative_path.read_text()
            assert f"Iteration {i}" in cumulative, f"Iteration {i} not in cumulative diff"
            print(f"    ✓ cumulative_changes.patch updated")

        # Final verification
        print("\n" + "="*60)
        print("Final Verification")
        print("="*60)

        index = json.loads((kept_dir / "index.json").read_text())
        cumulative = (kept_dir / "cumulative_changes.patch").read_text()

        print(f"\n📁 kept_changes directory structure:")
        for f in sorted(kept_dir.iterdir()):
            size = f.stat().st_size
            print(f"  {f.name:50} ({size:,} bytes)")

        print(f"\n📊 index.json summary:")
        print(f"  Total kept iterations: {len(index['kept_iterations'])}")
        for iter_info in index["kept_iterations"]:
            print(f"    - Iteration {iter_info['iteration']}: {iter_info['summary']}")

        print(f"\n📝 cumulative_changes.patch preview:")
        lines = cumulative.split('\n')
        for line in lines[:15]:
            print(f"  {line}")
        if len(lines) > 15:
            print(f"  ... ({len(lines)} total lines)")

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

        # Show what the output structure looks like
        print(f"\nExpected output structure in real run:")
        print(f"  {cfg.repo_root}/agent_loop/agent_artifacts/<timestamp>/kept_changes/")
        print(f"    ├── index.json")
        print(f"    ├── iter_1_<summary>.patch")
        print(f"    ├── iter_2_<summary>.patch")
        print(f"    ├── iter_3_<summary>.patch")
        print(f"    └── cumulative_changes.patch")


if __name__ == "__main__":
    test_kept_changes()
