# UFC Prediction Web Interface

A simple Streamlit web application for running UFC fight predictions.

## Features

1. **Batch Predictions (CSV)**
   - Upload CSV files or paste CSV data
   - Run predictions on multiple fights at once
   - View results with model probabilities, market odds, and edges
   - Download results as CSV

2. **Fighter Comparison**
   - Compare two fighters directly
   - Get detailed prediction breakdown
   - View feature analysis and model reasoning

## Requirements

- Python 3.12+
- All dependencies from `pyproject.toml`
- Streamlit (already in dependencies)

## Running the App

### From the main project directory:

```bash
# Activate virtual environment (if using one)
source .venv/bin/activate  # or your venv activation command

# Run Streamlit
streamlit run predict_site/app.py
```

### From the bundle:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Streamlit
streamlit run predict_site/app.py
```

The app will open in your browser at `http://localhost:8501`

## CSV Format

For batch predictions, your CSV should have these columns:

- `event`: Event name (e.g., "UFC 325")
- `fight_date`: Optional date
- `fighter_1_name`: First fighter name
- `fighter_2_name`: Second fighter name
- `fighter_1_odds`: American odds (e.g., -172, 250)
- `fighter_2_odds`: American odds (e.g., 147, -340)
- `is_title_fight`: 0 or 1

Example:
```csv
event,fight_date,fighter_1_name,fighter_2_name,fighter_1_odds,fighter_2_odds,is_title_fight
UFC 325,,Tai Tuivasa,Tallison Teixeira,250,-340,0
```

## Deployment

To deploy this with the bundle:

1. Copy the `predict_site` folder into the bundle
2. Update the bundle's bundler script to include `predict_site` if needed
3. Run the app on the remote machine using Streamlit

For production deployment, consider using:
- Streamlit Cloud
- Docker containerization
- Reverse proxy (nginx) with Streamlit



