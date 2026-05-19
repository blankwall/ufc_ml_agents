#!/usr/bin/env python3
"""
Create a cited evidence review from v_agent_packet_evidence.

By default this builds a deterministic scaffold that groups evidence into
support, caution, context, and audit buckets. With --llm it can pass that
scaffold plus the underlying evidence rows to an LLM, which must return
evidence-only cited reasoning with explicit evidence_id citations and no
recommendations.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.context_packet import DEFAULT_POOL, find_target  # noqa: E402
from backtest.elo_analysis import DEFAULT_ALIAS_SOURCES, load_aliases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--fight-pool-id", type=int, default=None)
    parser.add_argument("--fighter1", default=None)
    parser.add_argument("--fighter2", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--llm", action="store_true", help="Use the LLM-backed evidence review harness.")
    parser.add_argument("--max-evidence", type=int, default=40, help="Maximum evidence rows to send to the LLM.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for LLM review mode.")
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def resolve_fight_pool_id(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    if args.fight_pool_id is not None:
        return args.fight_pool_id
    if not args.fighter1 or not args.fighter2:
        raise SystemExit("Pass --fight-pool-id or both --fighter1 and --fighter2.")
    aliases = load_aliases(DEFAULT_ALIAS_SOURCES)
    target, _ = find_target(
        conn,
        fighter1=args.fighter1,
        fighter2=args.fighter2,
        date=args.date,
        season=None,
        aliases=aliases,
    )
    return int(target["id"])


def load_evidence(conn: sqlite3.Connection, fight_pool_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT evidence_id, evidence_role, evidence_type, summary, data_json, source_table, source_key
        FROM v_agent_packet_evidence
        WHERE fight_pool_id = ?
        ORDER BY evidence_role, evidence_type, evidence_id
        """,
        (fight_pool_id,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"No evidence rows found for fight_pool_id={fight_pool_id}")
    return [dict(row) for row in rows]


def parse_payload(row: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(row["data_json"])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def classify(row: dict[str, Any]) -> str:
    payload = parse_payload(row)
    if row["evidence_type"] == "pattern_stat":
        roi = payload.get("roi")
        win_rate = payload.get("win_rate")
        sample_size = payload.get("sample_size") or 0
        if sample_size >= 30 and (roi or 0.0) > 0 and (win_rate or 0.0) >= 0.65:
            return "support"
        if sample_size >= 20 and (roi or 0.0) <= 0:
            return "caution"
    if row["evidence_type"] == "elo_snapshot":
        if (payload.get("pick_elo_diff") or 0) >= 50:
            return "support"
        if (payload.get("pick_elo_diff") or 0) <= -50:
            return "caution"
    if row["evidence_type"] == "elo_triangle":
        if payload.get("model_market_elo_triangle") == "model_and_market_under_elo":
            return "support"
        if payload.get("model_market_elo_triangle") == "model_and_market_over_elo":
            return "caution"
    if row["evidence_type"] == "opponent_quality":
        if (payload.get("pick_opponent_quality_diff") or 0) >= 50:
            return "support"
        if (payload.get("pick_opponent_quality_diff") or 0) <= -50:
            return "caution"
    if row["evidence_type"] == "trait_delta":
        deltas = payload.get("deltas", {})
        if (deltas.get("cardio_score_diff") or 0) >= 10 or (deltas.get("striking_efficiency_score_diff") or 0) >= 10:
            return "support"
        if (deltas.get("cardio_score_diff") or 0) <= -10 or (deltas.get("defensive_exposure_score_diff") or 0) >= 10:
            return "caution"
    if row["evidence_type"] in {"target_context", "recent_fight"}:
        return "audit"
    return "context"


def build_review(evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"support": [], "caution": [], "context": [], "audit": []}
    target = None
    for row in evidence_rows:
        item = {
            "evidence_id": row["evidence_id"],
            "evidence_type": row["evidence_type"],
            "summary": row["summary"],
            "source": f"{row['source_table']}:{row['source_key']}",
        }
        bucket = classify(row)
        buckets[bucket].append(item)
        if row["evidence_type"] == "target_context":
            target = parse_payload(row)

    return {
        "review_mode": "deterministic_evidence_review_v0",
        "decision_scope": "empirical_evidence_only",
        "not_a_recommendation": True,
        "target": target,
        "support": buckets["support"],
        "caution": buckets["caution"],
        "context": buckets["context"],
        "audit": buckets["audit"][:8],
        "review_notes": [
            "Evidence is grouped deterministically from v_agent_packet_evidence.",
            "A downstream LLM/agent must cite evidence_id values for any claim.",
            "This review does not create probabilities, stakes, or bet/skip recommendations.",
        ],
    }


LLM_SYSTEM_PROMPT = """You are reviewing a UFC evidence packet.

You must produce an evidence-only review. You are NOT allowed to recommend bets,
predict winners, assign probabilities, or invent facts not present in the
evidence rows.

You MUST respond with valid JSON only in exactly this structure:
{
  "summary": "<2-3 sentence evidence-only summary>",
  "support": [{"claim": "<hedged evidence-backed claim>", "citations": [123, 456]}],
  "caution": [{"claim": "<hedged evidence-backed caution>", "citations": [789]}],
  "context": [{"claim": "<neutral contextual note>", "citations": [321]}],
  "audit": [{"claim": "<audit/provenance/data-quality note>", "citations": [654]}],
  "review_notes": [
    "Evidence-only; not a recommendation.",
    "Every claim is backed by explicit evidence_id citations."
  ]
}

Rules:
- Each citations array must contain one or more evidence_id integers from the supplied evidence list.
- Use only supplied evidence.
- Use hedged language such as "suggests", "supports", "historically", "contextually".
- Do not say a fighter will win, should be bet, is the play, is safe, or similar.
- Prefer 1-3 items per section.
- If a section has no grounded claim, return an empty list for that section.
"""


def load_zai_client():
    zai_path = Path.home() / "code" / "freshflowAI"
    if str(zai_path) not in sys.path:
        sys.path.insert(0, str(zai_path))
    try:
        from chat_z import ZAIClient  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit("LLM review requires freshflowAI/chat_z (ZAIClient) in ~/code/freshflowAI.") from exc
    return ZAIClient


def prompt_evidence_rows(evidence_rows: list[dict[str, Any]], *, max_evidence: int) -> list[dict[str, Any]]:
    prompt_rows = []
    for row in evidence_rows[:max_evidence]:
        prompt_rows.append(
            {
                "evidence_id": row["evidence_id"],
                "evidence_role": row["evidence_role"],
                "evidence_type": row["evidence_type"],
                "summary": row["summary"],
                "payload": parse_payload(row),
                "source": f"{row['source_table']}:{row['source_key']}",
            }
        )
    return prompt_rows


def build_llm_prompt(review: dict[str, Any], evidence_rows: list[dict[str, Any]], *, max_evidence: int) -> str:
    target = review.get("target") or {}
    prompt_payload = {
        "task": "Create an evidence-only cited review for this fight context packet.",
        "target": {
            "fight_pool_id": target.get("fight_pool_id"),
            "date": target.get("date"),
            "fighter1": target.get("fighter1"),
            "fighter2": target.get("fighter2"),
            "pick": target.get("pick"),
            "pick_prob": target.get("pick_prob"),
            "pick_odds": target.get("pick_odds"),
            "current_decision": target.get("current_decision"),
        },
        "deterministic_scaffold": {
            "support": review.get("support", []),
            "caution": review.get("caution", []),
            "context": review.get("context", []),
            "audit": review.get("audit", []),
        },
        "evidence_rows": prompt_evidence_rows(evidence_rows, max_evidence=max_evidence),
    }
    return json.dumps(prompt_payload, indent=2, sort_keys=True)


def _extract_text(response: dict[str, Any]) -> str:
    content = response.get("content", [])
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _strip_fences(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return clean.strip()


def _validate_claim_group(name: str, items: Any, *, allowed_ids: set[int]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"{name} must be a list")
    validated = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{name} items must be objects")
        claim = str(item.get("claim", "")).strip()
        citations = item.get("citations")
        if not claim:
            raise ValueError(f"{name} claim is required")
        if not isinstance(citations, list) or not citations:
            raise ValueError(f"{name} citations must be a non-empty list")
        citation_ids: list[int] = []
        for citation in citations:
            try:
                evidence_id = int(citation)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} citations must be integers") from exc
            if evidence_id not in allowed_ids:
                raise ValueError(f"{name} citation E{evidence_id} not found in evidence rows")
            citation_ids.append(evidence_id)
        validated.append({"claim": claim, "citations": citation_ids})
    return validated


def parse_llm_review(raw_text: str, *, allowed_ids: set[int], target: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(_strip_fences(raw_text))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM review payload must be an object")

    review_notes = parsed.get("review_notes")
    if not isinstance(review_notes, list) or not all(isinstance(note, str) for note in review_notes):
        raise ValueError("review_notes must be a list of strings")

    return {
        "review_mode": "llm_evidence_review_v0",
        "decision_scope": "empirical_evidence_only",
        "not_a_recommendation": True,
        "target": target,
        "summary": str(parsed.get("summary", "")).strip(),
        "support": _validate_claim_group("support", parsed.get("support", []), allowed_ids=allowed_ids),
        "caution": _validate_claim_group("caution", parsed.get("caution", []), allowed_ids=allowed_ids),
        "context": _validate_claim_group("context", parsed.get("context", []), allowed_ids=allowed_ids),
        "audit": _validate_claim_group("audit", parsed.get("audit", []), allowed_ids=allowed_ids),
        "review_notes": review_notes,
    }


def build_llm_review(
    deterministic_review: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    max_evidence: int,
    temperature: float,
) -> dict[str, Any]:
    ZAIClient = load_zai_client()
    client = ZAIClient()
    prompt = build_llm_prompt(deterministic_review, evidence_rows, max_evidence=max_evidence)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system=LLM_SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=temperature,
    )
    raw_text = _extract_text(response)
    allowed_ids = {int(row["evidence_id"]) for row in evidence_rows[:max_evidence]}
    return parse_llm_review(raw_text, allowed_ids=allowed_ids, target=deterministic_review.get("target") or {})


def print_review(review: dict[str, Any]) -> None:
    target = review.get("target") or {}
    print("=" * 90)
    print("AGENT EVIDENCE REVIEW")
    print("=" * 90)
    if target:
        print(
            f"{target.get('date')} {target.get('fighter1')} vs {target.get('fighter2')} | "
            f"pick={target.get('pick')} prob={target.get('pick_prob'):.1%} "
            f"odds={target.get('pick_odds')} decision={target.get('current_decision')}"
        )
    print("Scope: empirical evidence only; not a recommendation.")
    if review.get("summary"):
        print(f"\nSummary\n  {review['summary']}")

    for title, key in [("Support", "support"), ("Caution", "caution"), ("Context", "context"), ("Audit examples", "audit")]:
        print(f"\n{title}")
        rows = review[key]
        if not rows:
            print("  --")
            continue
        for row in rows[:10]:
            if "summary" in row:
                print(f"  [E{row['evidence_id']}] {row['summary']}")
            else:
                cited = ", ".join(f"E{evidence_id}" for evidence_id in row["citations"])
                print(f"  [{cited}] {row['claim']}")

    print("\nReview notes")
    for note in review["review_notes"]:
        print(f"  - {note}")


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.pool)
    conn.row_factory = sqlite3.Row
    try:
        fight_pool_id = resolve_fight_pool_id(conn, args)
        evidence = load_evidence(conn, fight_pool_id)
    finally:
        conn.close()

    deterministic_review = build_review(evidence)
    review = (
        build_llm_review(
            deterministic_review,
            evidence,
            max_evidence=args.max_evidence,
            temperature=args.temperature,
        )
        if args.llm
        else deterministic_review
    )
    if args.json_only:
        print(json.dumps(review, indent=2, sort_keys=True))
    else:
        print_review(review)


if __name__ == "__main__":
    main()
