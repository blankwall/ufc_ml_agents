"""
AI Analysis Service
-------------------
Calls the z.ai GLM API (via ZAIClient) to produce an independent
fight analysis based purely on fighter stats and history.

Deliberately does NOT receive odds or model predictions — the goal is
an independent signal, not a confirmation of what the model already says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# ── z.ai client from freshflowAI ─────────────────────────────────────────────
ZAI_PATH = Path.home() / "code" / "freshflowAI"
if str(ZAI_PATH) not in sys.path:
    sys.path.insert(0, str(ZAI_PATH))

from chat_z import ZAIClient  # type: ignore


# ── system prompt contract ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert UFC fight analyst. You will be given stats and recent fight history for two fighters.

Your job is to predict the winner and explain your reasoning.

You MUST respond with ONLY valid JSON in exactly this structure — no markdown, no extra text:

{
  "winner": "<exact fighter name>",
  "winner_pct": <integer 51-90>,
  "reasons": [
    "<one sentence reason 1>",
    "<one sentence reason 2>",
    "<one sentence reason 3>"
  ]
}

Rules:
- "winner" must be exactly one of the two fighter names provided
- "winner_pct" is your confidence the winner wins (51–90), the loser probability is 100 minus this
- Each reason must be exactly one sentence focused on a specific stat, stylistic edge, or historical pattern
- Do NOT mention betting odds, market lines, or any external model predictions
- Base your analysis only on the stats and fight history provided"""


# ── prompt builder ────────────────────────────────────────────────────────────

def _fmt_stat(val, fmt=".1f", suffix="") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}{suffix}"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _build_fighter_block(name: str, data: dict) -> str:
    record   = data.get("record", "?-?-?")
    age      = data.get("age", "?")
    height   = _fmt_stat(data.get("height_cm"), ".0f", " cm")
    reach    = _fmt_stat(data.get("reach_inches"), ".1f", "\"")
    stance   = data.get("stance") or "Unknown"
    slpm     = _fmt_stat(data.get("sig_strikes_landed_per_min"))
    str_acc  = _fmt_pct(data.get("striking_accuracy"))
    str_def  = _fmt_pct(data.get("striking_defense"))
    sapm     = _fmt_stat(data.get("sig_strikes_absorbed_per_min"))
    td_avg   = _fmt_stat(data.get("takedown_avg_per_15min"))
    td_acc   = _fmt_pct(data.get("takedown_accuracy"))
    td_def   = _fmt_pct(data.get("takedown_defense"))
    sub_avg  = _fmt_stat(data.get("submission_avg_per_15min"))

    recent = data.get("recent_fights", [])
    fights_str = ""
    for f in recent[:3]:
        result   = f.get("result", "?")
        opponent = f.get("opponent", "?")
        event    = f.get("event", "?")[:35]
        close    = f.get("close_odds") or ""
        odds_str = f" ({close})" if close else ""
        fights_str += f"\n    {result}  vs {opponent} — {event}{odds_str}"
    if not fights_str:
        fights_str = "\n    No recent fights in DB"

    return f"""
{name} ({record})
  Age: {age} | Height: {height} | Reach: {reach} | Stance: {stance}
  Striking: {slpm} sig/min landed, {str_acc} accuracy, {sapm} absorbed, {str_def} defense
  Grappling: {td_avg} TDs/15min, {td_acc} TD accuracy, {td_def} TD defense, {sub_avg} subs/15min
  Last 3 fights:{fights_str}"""


def build_prompt(f1_name: str, f1_data: dict, f2_name: str, f2_data: dict) -> str:
    return (
        f"Analyse this UFC matchup:\n"
        f"{_build_fighter_block(f1_name, f1_data)}\n"
        f"{_build_fighter_block(f2_name, f2_data)}\n\n"
        f"Who wins and why? Respond with JSON only."
    )


# ── AI call ───────────────────────────────────────────────────────────────────

def _extract_text(response: dict) -> str:
    """Pull text out of Anthropic-format response content blocks."""
    content = response.get("content", [])
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def analyze_matchup(
    f1_name: str, f1_data: dict,
    f2_name: str, f2_data: dict,
) -> dict:
    """
    Call the LLM and return parsed structured analysis.
    Returns a dict with: winner, winner_pct, loser, loser_pct, reasons, raw (on error).
    """
    client = ZAIClient()
    prompt = build_prompt(f1_name, f1_data, f2_name, f2_data)

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system=SYSTEM_PROMPT,
        max_tokens=512*4,
        temperature=0.3,   # low temp → consistent structured output
    )

    raw_text = _extract_text(response)

    # Strip markdown fences if model wraps in ```json ... ```
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return {
            "error": "AI returned invalid JSON",
            "raw": raw_text,
        }

    winner     = parsed.get("winner", "")
    winner_pct = int(parsed.get("winner_pct", 60))
    winner_pct = max(51, min(90, winner_pct))   # clamp to contract range
    loser      = f2_name if winner == f1_name else f1_name
    loser_pct  = 100 - winner_pct
    reasons    = parsed.get("reasons", [])

    # Validate winner is one of the two fighters
    if winner not in (f1_name, f2_name):
        # Try partial match
        if f1_name.split()[-1].lower() in winner.lower():
            winner = f1_name
        elif f2_name.split()[-1].lower() in winner.lower():
            winner = f2_name
        else:
            winner = f1_name  # fallback

    return {
        "winner":     winner,
        "winner_pct": winner_pct,
        "loser":      loser,
        "loser_pct":  loser_pct,
        "reasons":    reasons[:3],
        "error":      None,
    }
