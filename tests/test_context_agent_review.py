import json

import pytest

from backtest.context_agent_review import build_llm_prompt, build_review, classify, parse_llm_review


def _evidence(evidence_id, evidence_type, payload, summary="summary"):
    return {
        "evidence_id": evidence_id,
        "evidence_role": "context_metric",
        "evidence_type": evidence_type,
        "summary": summary,
        "data_json": json.dumps(payload),
        "source_table": "source",
        "source_key": "key",
    }


def test_classify_supports_positive_pattern_and_elo():
    assert classify(
        _evidence(
            1,
            "pattern_stat",
            {"sample_size": 64, "win_rate": 0.797, "roi": 0.184},
        )
    ) == "support"
    assert classify(_evidence(2, "elo_snapshot", {"pick_elo_diff": 141})) == "support"


def test_build_review_keeps_evidence_citations():
    review = build_review(
        [
            _evidence(
                1,
                "target_context",
                {"date": "2026-01-01", "fighter1": "A", "fighter2": "B", "pick": "A"},
                "target",
            ),
            _evidence(2, "trait_delta", {"deltas": {"cardio_score_diff": 12}}, "trait support"),
        ]
    )

    assert review["not_a_recommendation"] is True
    assert review["target"]["pick"] == "A"
    assert review["support"][0]["evidence_id"] == 2


def test_build_llm_prompt_contains_evidence_ids():
    evidence = [
        _evidence(
            1,
            "target_context",
            {"date": "2026-01-01", "fighter1": "A", "fighter2": "B", "pick": "A"},
            "target",
        ),
        _evidence(2, "trait_delta", {"deltas": {"cardio_score_diff": 12}}, "trait support"),
    ]
    review = build_review(evidence)

    prompt = build_llm_prompt(review, evidence, max_evidence=10)

    assert '"evidence_id": 1' in prompt
    assert '"evidence_id": 2' in prompt
    assert "trait support" in prompt


def test_parse_llm_review_validates_citations():
    target = {"fighter1": "A", "fighter2": "B", "pick": "A"}
    parsed = parse_llm_review(
        json.dumps(
            {
                "summary": "Evidence suggests some contextual support.",
                "support": [{"claim": "Cardio evidence supports the pick contextually.", "citations": [2]}],
                "caution": [],
                "context": [{"claim": "The packet remains evidence-only.", "citations": [1]}],
                "audit": [],
                "review_notes": [
                    "Evidence-only; not a recommendation.",
                    "Every claim is backed by explicit evidence_id citations.",
                ],
            }
        ),
        allowed_ids={1, 2},
        target=target,
    )

    assert parsed["review_mode"] == "llm_evidence_review_v0"
    assert parsed["support"][0]["citations"] == [2]
    assert parsed["target"]["pick"] == "A"


def test_parse_llm_review_rejects_unknown_citation():
    target = {"fighter1": "A", "fighter2": "B", "pick": "A"}
    with pytest.raises(ValueError, match="citation E99"):
        parse_llm_review(
            json.dumps(
                {
                    "summary": "Evidence suggests some contextual support.",
                    "support": [{"claim": "Unsupported citation.", "citations": [99]}],
                    "caution": [],
                    "context": [],
                    "audit": [],
                    "review_notes": [
                        "Evidence-only; not a recommendation.",
                        "Every claim is backed by explicit evidence_id citations.",
                    ],
                }
            ),
            allowed_ids={1, 2},
            target=target,
        )
