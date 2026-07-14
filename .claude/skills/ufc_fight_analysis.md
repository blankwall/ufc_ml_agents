# UFC Fight Analysis Skill

## Purpose

Use this skill for one-off UFC matchup reviews and betting-config gap analysis. The role is **model-gated calibration**: decide whether contextual, ELO, trait, and historical evidence suggests the model probability for its own pick should be adjusted up or down.

## Invocation

The Copilot CLI exposes `/skills` and `/agent`, but the documented command list does not include repo-defined custom slash commands such as `/fight`.

Recommended prompts:

```text
analyze fight Alex Perez vs Su Mudaerji -150/+118 using UFC MCP only
```

or select the existing agent:

```text
/agent ufc-fight-context-analyst
```

If a future CLI/plugin supports repo custom slash commands, `/fight fighter1 fighter2 odds1 odds2` should map to this skill.

## Hard rule

Never recommend a fighter with less than **48% model probability**. Treat the old "model pick only" rule as "model-supported side only": the higher-probability model pick always qualifies, and an alternate/underdog side can also qualify when the model gives that side at least 48%. If the contextual case favors a fighter below 48%, frame it as:

- pass on Fighter A,
- fade the current price,
- wait for a better model-side line, or
- flag the matchup as a config/model weakness.

Do not say "bet Fighter B" unless Fighter B has at least 48% model probability.

## Required MCP workflow

Use UFC MCP tools as the primary interface for fight analysis.

1. `init_fight_analysis`
   - Pass fighter names, fight date if known, and both American odds when provided.
   - Treat model probabilities for both sides, normalized market probabilities, raw edge, fighter resolution, and snapshots as source of truth.
2. `get_deterministic_signal_filter` or `get_elo_market_signal`
   - Capture ELO support/caution, cardio flags, model-vs-ELO gap, market-vs-ELO gap, and boost/risk tier.
3. `find_similar_market_fights`
   - Compare price, implied probability, model confidence, and edge.
4. `find_similar_elo_gap_fights`
   - Compare oriented ELO gap, model confidence, and edge.
5. `find_trait_matchup_examples`
   - Identify whether trait deltas support an upward/downward model adjustment.
6. `get_historical_pattern_summary`
   - Use aggregate sample size, win rate, ROI, and warnings as calibration evidence.
7. `get_fighter_elo_history`
   - Use for both fighters when current ELO or trajectory is central to the case.

## Reasoning standard

Always separate:

- **Raw model edge:** model probability minus normalized market probability.
- **Context-adjusted thesis:** whether evidence not fully represented in training plausibly raises or lowers a 48%+ model-supported side's true probability.

Negative raw edge is not an automatic no-bet. It means the answer must explain what evidence could lift the model pick above the break-even line and why that evidence is strong enough. If the adjustment thesis is thin, say pass or threshold only.

## Config-gap analysis

When the user asks how this should affect the app or betting setup, inspect:

- `config/betting_config.json`
- `fastapi_app/routers/predict.py`
- `fastapi_app/static/js/events.js`
- `backtest/bucket_analysis.py`
- `backtest/optimize_config.py`

Look for missing or underweighted context lanes:

- model pick with negative raw edge but strong trait/ELO support,
- ELO-against model pick that should lower sizing or require better price,
- cardio/trait offsets that reopen skipped fights,
- favorite tax and favorite-confidence threshold issues,
- underdog edge threshold issues,
- odds-cap and bet-sizing bucket issues.

## Output template

Use concise sections:

1. **Decision**
2. **Model and market**
3. **Context adjustment thesis**
4. **ELO / trait / historical evidence**
5. **Config implication**
6. **Line threshold**
