# UFC Card Scan Skill

## Purpose

Use this skill when the user asks to scan every fight on a UFC/MMA card, especially from the remote FastAPI events page.

## First step: parse the card

Use the helper script to get only fights that match the user's review band:

```bash
.venv/bin/python scripts/filter_remote_card.py --date 2026-05-30
```

Defaults:

- remote URL: `http://107.175.94.166:8002/api/events`
- odds band: `-300` through `+300`
- fight-count rule: both fighters must have `fight_count > 2`

Useful options:

```bash
.venv/bin/python scripts/filter_remote_card.py \
  --url http://107.175.94.166:8002/api/events \
  --date 2026-05-30 \
  --min-fights 2 \
  --odds-low -300 \
  --odds-high 300 \
  --json
```

## MCP review loop

For each eligible fight, use UFC MCP tools for fight evidence:

1. `init_fight_analysis`
2. `get_deterministic_signal_filter`
3. `query_fragility_cases` when a fight resembles a known failure mode
4. Add `find_trait_matchup_examples`, `find_similar_market_fights`, or `find_similar_elo_gap_fights` only when a fight is flagged as needing care

Do not deep-dive every fight by default. The card-scan output should be quick and simple.

## Output format

For each eligible fight, report one compact line:

```text
Fighter A vs Fighter B — PICK/SIDE | price | quick thesis | flag
```

Use flags:

- `OK` — model, market, and context are not raising special concern
- `CARE` — model side is price-thin, ELO conflicts, fragility pattern appears, or manual bet already exists
- `PASS` — not bettable under model-gated rules or current line is too expensive
- `ALREADY BET` — user said this was already bet/analyzed

End with a short list of fights requiring more care.
