# MCP improvement plan - June 15

## Problem

The current MCP context flow still leans too much on `context_pool` target rows. That is the wrong primary abstraction for live work because most agent analysis is for future fights, and future fights usually do not have exact rows in the historical context pool.

The second major gap is market context. If MCP analysis defaults to a 50/50 market when real odds exist elsewhere in the app, the agent gets distorted evidence for:

- edge
- market-vs-ELO interpretation
- expensive-favorite and live-dog logic
- market-shape comparisons
- whether a historical pattern is relevant at the current line

## Design principle

For future fights, MCP should answer:

> Given this matchup, this fight date, and this current line, what does the model plus historical evidence say?

It should not start with:

> Can I find this exact fight in `context_pool`?

`context_pool` should become a historical evidence library, not the source of truth for the target fight.

## Target architecture

1. `init_fight_analysis` is the root future-fight entrypoint.
   - Resolve fighters and aliases.
   - Resolve fight date and feature cutoff.
   - Resolve odds from user input first, then persisted app odds.
   - Run the model.
   - Return explicit market provenance.

2. Dynamic target packets replace exact-row requirements.
   - Build target identity from fighter/date/market/model state.
   - Use historical pools only for nearest examples, bucket stats, and pattern evidence.
   - Do not fail a future-fight packet because no `fight_pool_id` exists.

3. Real odds are first-class.
   - If user odds are supplied, use them.
   - If not supplied, search persisted app odds.
   - If odds are missing, continue with a neutral-line model analysis but mark pricing evidence as degraded.

4. Packet output should separate evidence families.
   - `target`: fighters, date, request, resolution.
   - `market`: odds, implied probabilities, provenance, last update.
   - `model`: probabilities, pick, edge.
   - `elo`: current ELO gap and price relationship.
   - `fighters`: compact snapshots.
   - `historical_examples`: market/ELO/trait comps.
   - `evidence_chain`: support and concern items with provenance.

## Implementation steps

1. Add a dynamic packet builder for future fights.
2. Make `get_context_packet` fall back to the dynamic packet when no exact historical target row exists or when fighter/date inputs describe a future fight.
3. Update pricing-sensitive MCP tools to use `init_fight_analysis` and app odds lookup before defaulting to 50/50.
4. Add warnings and labels for missing-market analyses:
   - `market_missing`
   - `neutral_line_edge`
   - `pricing_context_degraded`
5. Add tests for:
   - supplied odds
   - persisted app odds lookup
   - missing odds fallback
   - future fight with no context_pool row
   - Paddy Pimblett vs Benoit Saint Denis parity with event-card odds/date

## Expected result

An agent should be able to analyze a future fight without a context-pool row and still get decision-grade context when odds exist. If odds do not exist, the packet should clearly say the model view is neutral-line only and that pricing conclusions are degraded.
