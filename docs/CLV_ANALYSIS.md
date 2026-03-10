# CLV (Closing Line Value) Analysis

Validates the model against **opening** vs **closing** lines. If the model is closer to the closing line than the opening line was, it suggests the model is pricing like the sharp market.

## Why CLV matters

- **Results are noisy** — you can go 0–3 while your prices were correct.
- **Closing lines are the most efficient prices** in sports betting.
- If `|Model − Close| < |Open − Close|` on average, the model is **predicting market movement**.

## Pipeline

1. **Fetch open/close odds** (from bestfightodds.com graphs):

   ```bash
   python fetch_odds_graphs.py ufc-3971
   # Writes /tmp/ufc-3971_odds.json (or use -o data/odds/graphs/ufc-3971_odds.json)
   ```

2. **Run CLV analysis** (runs model on each fight, compares to open/close):

   ```bash
   python analysis/clv_analysis.py /tmp/ufc-3971_odds.json
   python analysis/clv_analysis.py /tmp/ufc-3971_odds.json --out data/clv_results.csv
   ```

3. **Multiple events** (e.g. a season):

   ```bash
   python analysis/clv_analysis.py /tmp/ufc-3970_odds.json /tmp/ufc-3971_odds.json
   python analysis/clv_analysis.py --dir /tmp --pattern "ufc-*_odds.json" --out data/clv_season.csv
   ```

## Metrics

| Metric | Meaning |
|--------|--------|
| **Model MAE** | `mean(\|model_prob − close_prob\|)`. Model’s error vs the closing line. |
| **Open MAE** | `mean(\|open_prob − close_prob\|)`. Opening line’s error vs the closing line. |
| **Model MAE < Open MAE** | Model predicts the closing line better than the open → **strong evidence of edge**. |
| **Model closer to close** | Count where `\|Model − Close\| < \|Open − Close\|` per fight. Higher % = model tracks sharp movement. |
| **Model predicted movement** | Model prob is **between** open and close (model agreed with direction of line move). |
| **clv_f1** | Close − Open for fighter 1 (positive = line moved toward f1). |

The **next metric** to watch: **Mean Absolute Error**.  
If your model error is less than open error **consistently**, the model is pricing closer to the (sharp) closing line than the open was — which is strong evidence of edge.

## Interpreting results

- **Model agrees with close but not open** → Model predicts market movement (good).
- **Model agrees with open but not close** → Model weaker than closing market.
- **Model disagrees with both** → Model likely wrong on that fight.

## Betting rule of thumb

**If you consistently beat closing lines (positive CLV on your bets), you will make money long-term** even when short-term results are bad.

## Future: line-movement model

A separate model could predict **where the line will move** (e.g. using narrative, KO rate, popularity). Combined with the outcome model:

```
Fight stats model → fair odds
Market model      → predicts line movement
Betting engine   → bet when both align (value + CLV)
```
