# UFC Fight Prediction System 🥊

A machine learning research project for predicting UFC fight outcomes using web scraping, feature engineering, and XGBoost models.

## Project Overview

This system explores predictive modeling for UFC fights by:
1. **Data Collection**: Scraping comprehensive fighter stats and fight history from UFCStats.com
2. **Feature Engineering**: Creating 299 predictive features from historical fight data
3. **ML Models**: Training XGBoost models with point-in-time validation
4. **Model Evaluation**: Validating performance on held-out test sets
5. **Backtesting**: Analyzing model performance on historical fights

## Project Structure

```
ufc_analysis_v2/
├── scrapers/              # Web scraping modules for UFCStats.com
├── database/              # SQLite database schema and management
├── features/              # Feature engineering pipeline (299 features)
├── models/                # XGBoost model implementation
├── evaluation/            # Model evaluation and reporting
├── backtesting/           # Backtesting framework
├── scripts/               # Analysis and utility scripts
├── quick_scripts/         # One-off scripts and utilities
├── data/                  # Data storage (raw, processed, predictions)
└── config/                # Configuration files
```

## Installation

This project uses `uv` for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage

### Phase 1: Data Collection

```bash
# Scrape all fighters
python scrapers/fighter_scraper.py --mode all

# Scrape specific event
python scrapers/event_scraper.py --event-id UFC-319
```

### Phase 2: Populate Database

**Important**: Due to naming collisions in the events data (where events always have fighter 1 as the winner and IDs weren't mapping properly), we use a simplified database populator that focuses on fights themselves:

```bash
python quick_scripts/populate_db_simple.py
```

### Phase 3: Create Feature Set

```bash
python -m features.feature_pipeline --create --feature-set full
```

This generates `data/processed/training_data.csv` with 299 engineered features.

### Phase 4: Train Models

#### Train model with 2025 data (for predictions)

```bash
python -m models.xgboost_model \
  --train \
  --evaluate \
  --check-calibration \
  --save-plots \
  --export-schema \
  --data-path data/processed/training_data.csv \
  --n-estimators 200 \
  --max-depth 4 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --model-name xgboost_model_with_2025
```

#### Train model without 2025 data (for evaluation)

```bash
python -m models.xgboost_model \
  --train \
  --evaluate \
  --check-calibration \
  --save-plots \
  --export-schema \
  --holdout-from-year 2025 \
  --data-path data/processed/training_data.csv \
  --n-estimators 200 \
  --max-depth 4 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.8
```

**Model naming convention:**
- `xgboost_model`: Does not contain 2025 data (prevents data leakage during evaluation)
- `xgboost_model_with_2025`: Used for predicting future events

### Phase 5: Generate Predictions

#### Predict a single fight

```bash
python xgboost_predict.py \
  --fighter-1 "CAIO BORRALHO" \
  --fighter-2 "Reinier De Ridder" \
  --model xgboost_model_with_2025
```

**Note**: Ordering is important. All predictions are based on fighter 1's win probability. You may need to swap the fighters if you want to predict the other way.

### Phase 6: Evaluate Model

Evaluate the model on 2025+ data:

```bash
python -m evaluation.evaluate_model \
  --data-path data/processed/training_data.csv \
  --odds-path ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports4 \
  --odds-date-tolerance-days 5 \
  --model-name xgboost_model \
  --strict-point-in-time
```

**Evaluation options:**
- `--strict-point-in-time`: Zero out known leaky base striking/grappling columns during evaluation (recommended)

### Phase 7: Backtest Model

#### Backtest on evaluation data

```bash
python scripts/backtest_betting_strategy.py \
  --eval-data reports_strict/eval_data_20251216_093142.csv \
  --strategy best_ev \
  --min-p 0.5 \
  --flat-stake 1 \
  --symmetric
```

#### Backtest on specific card

```bash
python scripts/backtest_manual_card.py \
  --input data/predictions/royval_kape_20251213_results.csv \
  --strategy winner \
  --model-name xgboost_model_with_2025 \
  --min-ev 0.01 \
  --flat-stake 1 \
  --output-csv data/predictions/royval_kape_20251213_results_with_model.csv
```

## Does it work?

Here's an example of the system running end-to-end on a real UFC card (UFC Fight Night: Royval vs. Kape, December 13, 2025):

```bash
python scripts/backtest_manual_card.py \
  --input data/predictions/royval_kape_20251213_results.csv \
  --strategy winner \
  --model-name xgboost_model_with_2025 \
  --min-ev 0.01 \
  --flat-stake 1 \
  --output-csv data/predictions/royval_kape_20251213_results_with_model.csv
```

**Output:**

```
2025-12-20 07:20:29.331 | INFO     | Loading XGBoost model 'xgboost_model_with_2025' and feature pipeline...
2025-12-20 07:20:29.344 | SUCCESS  | Loaded model from models/saved/xgboost_model_with_2025.json
2025-12-20 07:20:29.345 | SUCCESS  | Loaded feature pipeline from models/saved
2025-12-20 07:20:29.357 | INFO     | Database initialized: sqlite:///data/ufc_database.db
2025-12-20 07:20:29.476 | SUCCESS  | Prepared 299 features

================================================================================
MANUAL CARD BACKTEST | model=xgboost_model_with_2025 | strategy=winner
================================================================================
rows:      5
bets:      5

  fighter_1_name   fighter_2_name  fighter_1_odds  fighter_2_odds  model_p_f1_pct  model_p_f2_pct     ev_f1     ev_f2         bet_name  bet_side    bet_ev   profit
  Brandon Royval       Manel Kape             250            -300            21.4            78.6 -0.251000  0.048000       Manel Kape fighter_2  0.048000 0.333333
   Giga Chikadze   Kevin Vallejos             225            -265            45.9            54.1  0.491750 -0.254849   Kevin Vallejos fighter_2 -0.254849 0.377358
Melquizael Costa Morgan Charriere            -108            -112            64.8            35.2  0.248000 -0.333714 Melquizael Costa fighter_1  0.248000 0.925926
  Steven Asplund      Sean Sharaf            -205             175            71.7            28.3  0.066756 -0.221750   Steven Asplund fighter_1  0.066756 0.487805
    Luana Santos   Melissa Croden            -140             120            59.4            40.6  0.018286 -0.106800     Luana Santos fighter_1  0.018286 0.714286

Realized:
  staked:  5.000u
  profit:  2.839u
  ROI:     56.774%

Wrote: data/predictions/royval_kape_20251213_results_with_model.csv
```

The system successfully:
1. Loaded the trained XGBoost model and feature pipeline
2. Generated 299 features for each of the 5 fights
3. Produced win probability predictions for each fighter
4. Calculated expected value metrics
5. Generated a detailed output CSV with all predictions and results

### 2025 Season Performance Analysis

The model was evaluated on all 2025 UFC events (17 events, 151 fights) using strict point-in-time validation to prevent data leakage. Here are the key results:

**Overall Performance:**
- **Overall Accuracy**: 48.3% (73/151 correct predictions)
- **Events Analyzed**: 17
- **Holdout Period**: 2025+

**Underdog Performance (Key Finding):**

The model demonstrates strong performance in identifying underdog opportunities:
- **Underdog Picks (Model)**: 19 out of 65 underdog predictions (29% of underdog fights)
- **Underdog Wins (Actual Upsets)**: 19 out of 51 total upsets (37% of all upsets)

This suggests the model is particularly effective at identifying fights where underdogs have a better chance than the market suggests. The model picked underdogs in 29% of cases where they were underdogs, and those picks captured 37% of all actual upsets that occurred.

**Interactive Evaluation Report:**

A detailed HTML report is generated for each evaluation run, showing:
- Event-by-event breakdown with color-coded results
- Fight-by-fight predictions with confidence levels
- Comparison of model predictions vs. market odds
- Visual probability bars and calibration metrics

View the latest report: `reports_strict/model_evaluation_*.html` (generated during evaluation)

## Model Architecture

### Feature Pipeline

The system generates 299 features including:

- **Fighter career statistics**: Strikes, takedowns, submissions, defense rates
- **Rolling averages**: Last 3, 5, 10 fights
- **Opponent-adjusted metrics**: Quality-adjusted performance measures
- **Matchup-specific features**: Reach advantage, age difference, weight class
- **Momentum indicators**: Win streaks, finish rates, recent form

### ML Models

- **XGBoost**: Tree-based gradient boosting model optimized for structured tabular features

## Data Leakage Considerations

This project includes careful handling of temporal data leakage:

**What's leaking**: "Current" UFCStats-derived career averages stored on the fighters table (and in `features/striking.py` it can also iterate all linked fights without an as-of-date cut). These values can implicitly include future fights, so the model gets a more mature/accurate estimate of the fighter than would be available at prediction time.

**Why it matters**: With 0–2 prior fights, a small amount of "future-informed stabilization" (e.g., true-ish striking accuracy, takedown defense) can dominate the feature vector and swamp the uncertainty you'd actually have pre-fight.

**Mitigation**: Use `--strict-point-in-time` flag during evaluation to zero out known leaky base striking/grappling columns. This provides a more realistic assessment of model performance.

## Data Sources

- **UFCStats.com**: Fighter stats, fight history, event data

## Performance Metrics

- **Accuracy**: Overall prediction accuracy
- **Log Loss**: Probabilistic prediction quality
- **Calibration**: Prediction reliability at different confidence levels
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Contributing

This is a research project. Key areas for improvement:
- Additional data sources
- Novel feature engineering
- Advanced model architectures
- Better temporal validation strategies

## License

MIT License - See LICENSE file for details
