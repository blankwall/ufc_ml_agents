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

SYSTEM_PROMPT = """You are an expert UFC statistics analyst. You will be given career stats and recent fight history for two fighters.

Your job is to surface statistical patterns and tendencies — NOT to predict who wins. MMA outcomes are highly uncertain; your role is to describe what the numbers suggest, not to make a forecast.

You MUST respond with ONLY valid JSON in exactly this structure — no markdown, no extra text:

{
  "lean": "<exact fighter name who holds the clearer statistical edge>",
  "lean_strength": "<exactly one of: slight, moderate, clear>",
  "observations": [
    "<one sentence statistical observation 1>",
    "<one sentence statistical observation 2>",
    "<one sentence statistical observation 3>"
  ]
}

Rules:
- "lean" must be exactly one of the two fighter names
- "lean_strength" reflects how one-sided the statistical picture is:
    "slight"   = numbers are close, marginal edges only
    "moderate" = noticeable advantages in key areas
    "clear"    = significant edges across multiple dimensions
- Each observation must describe a specific statistical pattern using hedged language
  ("tends to", "shows", "suggests", "historically", "on average", "compared to")
- Do NOT use prediction language: "will win", "is likely to", "should beat", "will dominate"
- Do NOT mention betting odds, model outputs, or any external predictions
- Where numbers are very close, say so explicitly — do not manufacture edges
- Base analysis only on the stats and fight history provided

- IMPORTANT: YOU MUST THINK DEEPLY ABOUT THE STATS AND HISTORY PROVIDED. DO NOT MAKE ASSUMPTIONS.


"""


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
        f"Analyse the statistical picture for this UFC matchup:\n"
        f"{_build_fighter_block(f1_name, f1_data)}\n"
        f"{_build_fighter_block(f2_name, f2_data)}\n\n"
        f"What do the stats and history suggest? Respond with JSON only."
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

    lean          = parsed.get("lean", "")
    lean_strength = parsed.get("lean_strength", "slight").lower()
    if lean_strength not in ("slight", "moderate", "clear"):
        lean_strength = "slight"

    observations = parsed.get("observations") or parsed.get("reasons", [])

    # Validate lean is one of the two fighters; partial-match fallback
    if lean not in (f1_name, f2_name):
        if f1_name.split()[-1].lower() in lean.lower():
            lean = f1_name
        elif f2_name.split()[-1].lower() in lean.lower():
            lean = f2_name
        else:
            lean = f1_name  # last resort

    other = f2_name if lean == f1_name else f1_name

    return {
        "lean":          lean,
        "lean_strength": lean_strength,
        "other":         other,
        "observations":  observations[:3],
        "error":         None,
        # Legacy aliases so existing code that references winner/reasons won't hard-crash
        "winner":        lean,
        "reasons":       observations[:3],
    }
