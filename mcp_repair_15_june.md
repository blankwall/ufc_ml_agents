# MCP repair plan - June 15

## Goal

Close the gap between directional MCP reviews and top-confidence pricing calls for future fights. Dynamic synthetic mode is useful for side identification and support framing, but it needs stronger pricing logic, coverage labels, and matchup-native evidence before it should be treated like a complete context-pool packet.

## Phase 1 - Pricing decision clarity

- [x] Add explicit MCP bet-decision tool.
  - Use `config/betting_config.json`.
  - Use the same app-side bet evaluator where possible.
  - Return `bet`, `no_bet`, or `wait`.
  - Include skip code/reason, decision source, edge, confidence, market probability, and config thresholds.
- [x] Add line sensitivity output.
  - Return model fair price.
  - Return break-even market probability.
  - Return the price/probability where the current pick becomes bettable.
  - Identify when no line can fix the decision because confidence, data, or WMMA rules fail.
- [x] Add tests for the MCP decision wrapper and line-threshold math.

## Phase 2 - Dynamic packet trust score

- [x] Add coverage/confidence score to dynamic packets.
  - Exact context-pool row coverage is informational only for future fights, not a dynamic-packet warning.
  - Real market coverage.
  - ELO coverage.
  - Trait coverage.
  - Opponent-quality coverage.
  - Historical comps coverage.
- [x] Label true historical evidence vs synthetic reconstruction.
- [x] Materialize fuller future-fight context rows on demand.
  - Populate context-pool-like fields where possible.
  - Leave unavailable fields null with coverage notes.

## Phase 3 - Evidence quality upgrades

- [x] Improve opponent-quality enrichment.
  - Strength of schedule.
  - Best wins.
  - Recent opponent level.
  - Current-vs-peak decline curves.
- [x] Add richer matchup risk flags.
  - Layoff.
  - Chin/damage trend.
  - Five-round uncertainty.
  - Cardio uncertainty.
  - Small-sample inflation.
- [x] Improve nearest-example retrieval.
  - Style/trait similarity.
  - Market shape.
  - Model confidence.
  - ELO gap.
  - Favorite/underdog profile.

## Phase 4 - Future context sidecar persistence

- [x] Add generated future-context sidecar storage.
  - Persist dynamic future-fight rows to `data/enrichment/future_context.sqlite`.
  - Key rows by fighter/date/odds signature so repeated MCP calls reuse the same materialization identity.
  - Keep historical `context_pool.sqlite` immutable.
- [x] Attach future-context materialization metadata to dynamic packets.
  - Include sidecar key, path, stored-packet status, request signature, and refresh policy.
  - Mark packet `materialized_context_row` as persisted once stored.
- [x] Attach future-context materialization metadata to standalone helper dynamic fallbacks.
  - Cover trait deltas, style flags, historical patterns, and nearest examples through the shared fallback path.
- [x] Add tests for sidecar persistence and dynamic packet/helper metadata.

## Completion rule

Each phase is only done when the MCP output exposes the new fields, tests cover the behavior where practical, and the branch is committed/pushed.
