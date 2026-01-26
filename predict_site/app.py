#!/usr/bin/env python3
"""
UFC Prediction Web Interface

A simple Streamlit app for:
1. Uploading/pasting CSV files and running batch predictions
2. Comparing two fighters directly
"""

import sys
import warnings
import logging
from pathlib import Path

# Add parent directory to path to import project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress Streamlit ScriptRunContext warning when running in bare mode
warnings.filterwarnings(
    "ignore",
    message=".*missing ScriptRunContext.*",
    category=UserWarning,
    module="streamlit"
)

# Suppress Streamlit logger warnings about missing ScriptRunContext
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st
from io import StringIO
from typing import Optional
from loguru import logger

# Import project modules (after adding PROJECT_ROOT to path)
from evaluation.export_predictions_to_excel import add_model_predictions
from xgboost_predict import xgboost_predict

# Configure page
st.set_page_config(
    page_title="UFC Fight Predictions",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🥊 UFC Fight Predictions")
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Select Model",
    options=["baseline_jan_11_2026_age_feature_add_striking_landed", "baseline_jan_9_2026_age_feature_add_striking", "baseline_jan_9_2026_age_feature_add", "baseline_jan_9_2026_age", "xgboost_model_with_2025", "xgboost_model"],
    index=0,
    help="Choose which trained model to use for predictions"
)

use_symmetric = st.sidebar.checkbox(
    "Use Symmetric Mode",
    value=True,
    help="Average predictions from both fighter orders for more stable results"
)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Batch Predictions (CSV)", "⚔️ Fighter Comparison", "📈 Model Evaluation", "🔍 Fighter Search"])

# Tab 1: Batch Predictions
with tab1:
    st.header("Batch Predictions from CSV")
    st.markdown(
        """
        Upload a CSV file or paste CSV data with the following columns:
        - `event`: Event name (e.g., "UFC 325")
        - `fight_date`: Optional date
        - `fighter_1_name`: First fighter name
        - `fighter_2_name`: Second fighter name
        - `fighter_1_odds`: American odds (e.g., -172, 250)
        - `fighter_2_odds`: American odds (e.g., 147, -340)
        - `is_title_fight`: 0 or 1
        """
    )
    
    # Example CSV files with dates
    from datetime import datetime
    example_files_data = [
        {"name": "UFC 326", "path": "data/predictions/upcoming_fights_ufc326.csv", "date": datetime(2026, 3, 7)},
        {"name": "UFC Stricklan", "path": "data/predictions/upcoming_fights_strickland.csv", "date": datetime(2026, 2, 21)},
        {"name": "UFC 325", "path": "data/predictions/upcoming_fights_ufc325.csv", "date": datetime(2026, 1, 31)},
        {"name": "UFC 324", "path": "data/predictions/upcoming_fights_ufc324.csv", "date": datetime(2026, 1, 24)},
        {"name": "Fight Night: Royval vs. Kape", "path": "data/predictions/upcoming_fights_fight_night_royval_kape.csv", "date": datetime(2025, 12, 13)},
        {"name": "UFC 323", "path": "data/predictions/upcoming_fights_ufc323.csv", "date": datetime(2025, 12, 6)},
    ]
    
    # Sort by date (ascending - earliest first)
    example_files_data.sort(key=lambda x: x["date"])
    
    # Create display dictionary with dates
    example_files = {}
    for item in example_files_data:
        date_str = item["date"].strftime("%b %d, %Y")
        display_name = f"{item['name']} ({date_str})"
        example_files[display_name] = item["path"]
    
    # Initialize session state for selected example
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = None
    
    st.markdown("### 📋 Example Files")
    example_cols = st.columns(len(example_files))
    
    for idx, (display_name, path) in enumerate(example_files.items()):
        with example_cols[idx]:
            if st.button(f"📄 {display_name}", key=f"example_{idx}", use_container_width=True):
                st.session_state.selected_example = path
                st.rerun()
    
    # CSV input methods
    input_method = st.radio(
        "Input Method",
        ["Use Example File", "Upload CSV File", "Paste CSV Data"],
        horizontal=True
    )
    
    csv_data = None
    
    if input_method == "Use Example File" or st.session_state.selected_example:
        # Use selected example or let user choose
        if st.session_state.selected_example:
            example_path = st.session_state.selected_example
        else:
            example_path = st.selectbox(
                "Select Example File",
                options=list(example_files.values()),
                format_func=lambda x: [k for k, v in example_files.items() if v == x][0]
            )
            st.session_state.selected_example = example_path
        
        example_file = Path(PROJECT_ROOT / example_path)
        if example_file.exists():
            csv_data = example_file.read_text()
            # Get display name (remove date part for cleaner display)
            display_name = [k for k, v in example_files.items() if v == example_path][0]
            # Extract just the event name without date
            event_name = display_name.split(" (")[0] if " (" in display_name else display_name
            st.success(f"✅ Loaded: {event_name}")
            with st.expander("Preview Example File", expanded=False):
                st.dataframe(pd.read_csv(StringIO(csv_data)), use_container_width=True)
        else:
            st.warning(f"Example file not found: {example_path}")
            st.info("Make sure the file is committed to your repository for Streamlit Cloud deployment.")
    
    # Clear selection if switching to other input methods
    if input_method != "Use Example File":
        st.session_state.selected_example = None
    
    elif input_method == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with fight data"
        )
        if uploaded_file is not None:
            csv_data = uploaded_file.read().decode('utf-8')
    
    else:  # Paste CSV Data
        csv_text = st.text_area(
            "Paste CSV data here",
            height=200,
            help="Paste CSV data including header row"
        )
        if csv_text:
            csv_data = csv_text
    
    if csv_data:
        try:
            # Parse CSV
            df = pd.read_csv(StringIO(csv_data))
            
            # Validate required columns
            required_cols = [
                "fighter_1_name", "fighter_2_name",
                "fighter_1_odds", "fighter_2_odds"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Preprocess: Add fighter IDs for known ambiguous fighters
                # This handles cases where name matching might pick the wrong fighter
                def resolve_ambiguous_fighter(name: str, context: str = "") -> Optional[int]:
                    """
                    Return fighter ID for known ambiguous fighters.
                    context can be used for additional disambiguation (e.g., opponent name, event)
                    """
                    name_lower = name.lower().strip()
                    
                    # Jean Silva 'Lord' (ID: 3707) - the active UFC fighter
                    # This is the one we want for UFC 324 (vs Arnold Allen)
                    if "jean silva" in name_lower or name_lower == "jean silva":
                        # Check if opponent is Arnold Allen (UFC 324 context)
                        if "arnold allen" in context.lower() or "arnold" in context.lower():
                            return 3707  # Jean Silva 'Lord'
                        # Default to the active one (Lord) for now
                        return 3707
                    
                    return None
                
                # Add fighter IDs if not already present
                if "fighter_1_id" not in df.columns:
                    df["fighter_1_id"] = None
                if "fighter_2_id" not in df.columns:
                    df["fighter_2_id"] = None
                
                # Apply special handling for ambiguous fighters
                for idx, row in df.iterrows():
                    f1_name = str(row.get("fighter_1_name", "")).strip()
                    f2_name = str(row.get("fighter_2_name", "")).strip()
                    
                    # Create context from opponent and event
                    context = f"{f1_name} {f2_name} {row.get('event', '')}"
                    
                    # Check fighter 1
                    if pd.isna(df.at[idx, "fighter_1_id"]) or df.at[idx, "fighter_1_id"] is None:
                        resolved_id = resolve_ambiguous_fighter(f1_name, context)
                        if resolved_id is not None:
                            df.at[idx, "fighter_1_id"] = resolved_id
                    
                    # Check fighter 2
                    if pd.isna(df.at[idx, "fighter_2_id"]) or df.at[idx, "fighter_2_id"] is None:
                        resolved_id = resolve_ambiguous_fighter(f2_name, context)
                        if resolved_id is not None:
                            df.at[idx, "fighter_2_id"] = resolved_id
                
                st.success(f"✅ Loaded {len(df)} fights")
                
                # Show preview
                with st.expander("Preview Data", expanded=False):
                    st.dataframe(df, use_container_width=True)
                
                # Run predictions
                if st.button("🚀 Run Predictions", type="primary"):
                    with st.spinner("Running predictions... This may take a minute."):
                        try:
                            # Add model predictions
                            df_results = add_model_predictions(
                                df,
                                model_name=model_name,
                                symmetric=use_symmetric
                            )
                            
                            st.success("✅ Predictions complete!")
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            # Calculate predicted winner (fighter with highest model probability) and best edge columns if not present
                            # (preview_upcoming_fights creates these, but add_model_predictions might not)
                            if "predicted_winner" not in df_results.columns:
                                df_results["predicted_winner"] = df_results.apply(
                                    lambda row: row["fighter_1_name"] if row.get("model_p_f1_pct", 0) > row.get("model_p_f2_pct", 0) 
                                    else row["fighter_2_name"], axis=1
                                )
                            # Also check for legacy "best_fighter" column name and rename it
                            if "best_fighter" in df_results.columns and "predicted_winner" not in df_results.columns:
                                df_results["predicted_winner"] = df_results["best_fighter"]
                            
                            if "best_model_prob_pct" not in df_results.columns:
                                df_results["best_model_prob_pct"] = df_results.apply(
                                    lambda row: max(row.get("model_p_f1_pct", 0), row.get("model_p_f2_pct", 0)), axis=1
                                )
                            
                            if "best_market_prob_pct" not in df_results.columns:
                                df_results["best_market_prob_pct"] = df_results.apply(
                                    lambda row: max(row.get("implied_p_f1_pct", 0), row.get("implied_p_f2_pct", 0)), axis=1
                                )
                            
                            if "best_edge_pct" not in df_results.columns:
                                df_results["best_edge_pct"] = df_results.apply(
                                    lambda row: max(row.get("edge_f1_pct", 0), row.get("edge_f2_pct", 0)), axis=1
                                )
                            
                            # Calculate market probability and edge for the predicted winner specifically
                            # (the fighter with the highest model probability, not the market favorite)
                            if "predicted_winner_market_prob_pct" not in df_results.columns:
                                df_results["predicted_winner_market_prob_pct"] = df_results.apply(
                                    lambda row: row.get("implied_p_f1_pct", 0) if row.get("model_p_f1_pct", 0) > row.get("model_p_f2_pct", 0)
                                    else row.get("implied_p_f2_pct", 0), axis=1
                                )
                            
                            if "predicted_winner_edge_pct" not in df_results.columns:
                                df_results["predicted_winner_edge_pct"] = df_results.apply(
                                    lambda row: row.get("edge_f1_pct", 0) if row.get("model_p_f1_pct", 0) > row.get("model_p_f2_pct", 0)
                                    else row.get("edge_f2_pct", 0), axis=1
                                )
                            
                            # Calculate model confidence metrics
                            # model_confidence = max(model_prob_f1, model_prob_f2)
                            df_results["model_confidence"] = df_results.apply(
                                lambda row: max(row.get("model_p_f1_pct", 0), row.get("model_p_f2_pct", 0)), axis=1
                            )
                            
                            # confidence_rank = rank by confidence (1 = highest confidence)
                            df_results["confidence_rank"] = df_results["model_confidence"].rank(ascending=False, method="min").astype(int)
                            
                            # is_top_25 = True if in top 25% by confidence
                            # Calculate 75th percentile threshold (top 25% means above 75th percentile)
                            if len(df_results) > 0:
                                threshold_75th = df_results["model_confidence"].quantile(0.75)
                                df_results["is_top_25"] = df_results["model_confidence"] >= threshold_75th
                            else:
                                df_results["is_top_25"] = False
                            
                            # Calculate potential_underdog flag
                            # Underdogs only if: Model ≥ 55% AND Market ≤ 45%
                            # This identifies cases where model is confident in an underdog
                            def check_potential_underdog(row):
                                model_prob = row.get("model_confidence", 0)
                                # Get market probability for the predicted winner
                                market_prob = row.get("predicted_winner_market_prob_pct", 0)
                                # If predicted_winner_market_prob_pct doesn't exist, calculate it
                                if market_prob == 0 or pd.isna(market_prob):
                                    if row.get("model_p_f1_pct", 0) > row.get("model_p_f2_pct", 0):
                                        market_prob = row.get("implied_p_f1_pct", 0)
                                    else:
                                        market_prob = row.get("implied_p_f2_pct", 0)
                                
                                # Check conditions: Model ≥ 55% AND Market ≤ 45%
                                return (model_prob >= 55.0) and (market_prob <= 45.0)
                            
                            df_results["potential_underdog"] = df_results.apply(check_potential_underdog, axis=1)
                            
                            # Create styled summary table
                            # Prefer columns that show values for the predicted winner specifically
                            summary_cols = [
                                "event", "fight_date",
                                "fighter_1_name", "fighter_2_name",
                                "predicted_winner",
                                "model_confidence", "confidence_rank", "is_top_25", "potential_underdog",
                                "predicted_winner_market_prob_pct", "predicted_winner_edge_pct",
                                "risk_notes"
                            ]
                            # Fall back to best_* columns if predicted_winner_* columns don't exist
                            available_summary_cols = []
                            for col in summary_cols:
                                if col in df_results.columns:
                                    available_summary_cols.append(col)
                                elif col == "predicted_winner_market_prob_pct" and "best_market_prob_pct" in df_results.columns:
                                    available_summary_cols.append("best_market_prob_pct")
                                elif col == "predicted_winner_edge_pct" and "best_edge_pct" in df_results.columns:
                                    available_summary_cols.append("best_edge_pct")
                            
                            # Display with highlighting
                            df_display = df_results[available_summary_cols].copy()
                            
                            # Ensure best_edge_pct exists in df_results for sorting (even if not displayed)
                            if "best_edge_pct" not in df_results.columns:
                                df_results["best_edge_pct"] = df_results.apply(
                                    lambda row: max(row.get("edge_f1_pct", 0), row.get("edge_f2_pct", 0)), axis=1
                                )
                            
                            # Create a styled dataframe with highlighted predicted_winner
                            def style_row(row):
                                styles = [''] * len(row)
                                if 'predicted_winner' in df_display.columns:
                                    predicted_winner_idx = df_display.columns.get_loc('predicted_winner')
                                    styles[predicted_winner_idx] = 'background-color: #fff3cd; font-weight: bold; color: #856404; font-size: 1.1em;'
                                # Also check for legacy column name
                                elif 'best_fighter' in df_display.columns:
                                    best_fighter_idx = df_display.columns.get_loc('best_fighter')
                                    styles[best_fighter_idx] = 'background-color: #fff3cd; font-weight: bold; color: #856404; font-size: 1.1em;'
                                # Also highlight best_edge_pct if positive
                                if 'best_edge_pct' in df_display.columns:
                                    edge_idx = df_display.columns.get_loc('best_edge_pct')
                                    edge_val = row.iloc[edge_idx] if edge_idx < len(row) else 0
                                    if edge_val > 10:
                                        styles[edge_idx] = 'background-color: #d4edda; font-weight: bold; color: #155724;'
                                    elif edge_val > 0:
                                        styles[edge_idx] = 'background-color: #fff3cd; font-weight: bold;'
                                return styles
                            
                            # Apply styling
                            styled_df = df_display.style.apply(style_row, axis=1)
                            
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=400
                            )
                            
                            # Also show a cleaner summary table
                            st.markdown("---")
                            st.subheader("📊 Summary by Edge")
                            
                            # Sort by edge - use best_edge_pct from df_results (for sorting), or calculate it
                            # We need to sort from df_results to have access to all columns, then filter to display columns
                            if "best_edge_pct" in df_results.columns:
                                df_sorted = df_results.sort_values("best_edge_pct", ascending=False)
                            elif "edge_f1_pct" in df_results.columns and "edge_f2_pct" in df_results.columns:
                                df_results["_temp_best_edge"] = df_results.apply(
                                    lambda row: max(row.get("edge_f1_pct", 0), row.get("edge_f2_pct", 0)), axis=1
                                )
                                df_sorted = df_results.sort_values("_temp_best_edge", ascending=False)
                            else:
                                df_sorted = df_results
                            
                            # Now filter to display columns for the summary display
                            df_sorted_display = df_sorted[available_summary_cols].copy() if available_summary_cols else df_sorted
                            
                            # Create a more readable summary
                            # df_sorted is df_results sorted by edge, so it has all columns
                            for idx, row in df_sorted.iterrows():
                                fighter_1 = row.get("fighter_1_name", "N/A")
                                fighter_2 = row.get("fighter_2_name", "N/A")
                                
                                # Determine which fighter is the predicted winner based on model probabilities
                                model_p_f1 = row.get("model_p_f1_pct", 0)
                                model_p_f2 = row.get("model_p_f2_pct", 0)
                                
                                if model_p_f1 > model_p_f2:
                                    predicted_winner = fighter_1
                                    model_prob = model_p_f1
                                    predicted_winner_market_prob = row.get("implied_p_f1_pct", 0)
                                else:
                                    predicted_winner = fighter_2
                                    model_prob = model_p_f2
                                    predicted_winner_market_prob = row.get("implied_p_f2_pct", 0)
                                
                                # Fallback to predicted_winner column if model probabilities aren't available
                                if not predicted_winner or predicted_winner == "N/A":
                                    predicted_winner = row.get("predicted_winner", row.get("best_fighter", "N/A"))
                                    model_prob = row.get("best_model_prob_pct", 0)
                                    predicted_winner_market_prob = row.get("predicted_winner_market_prob_pct", row.get("best_market_prob_pct", 0))
                                
                                # Calculate edge directly from model and market probabilities to ensure accuracy
                                # Edge = model_probability - market_probability (for the predicted winner)
                                predicted_winner_edge = model_prob - predicted_winner_market_prob
                                risk_notes = row.get("risk_notes", "")
                                
                                # Determine if this is a strong edge
                                is_strong_edge = predicted_winner_edge > 10
                                border_color = "#28a745" if is_strong_edge else "#6c757d"
                                bg_color = "#d4edda" if is_strong_edge else "#f8f9fa"
                                
                                risk_html = ""
                                if risk_notes:
                                    risk_html = f'<p style="margin: 5px 0; color: #856404;"><strong>⚠️ Note:</strong> {risk_notes}</p>'
                                
                                st.markdown(f"""
                                <div style="padding: 15px; margin: 10px 0; border: 2px solid {border_color}; border-radius: 8px; background-color: {bg_color};">
                                    <h4 style="margin: 0 0 10px 0; color: {border_color};">
                                        {fighter_1} vs {fighter_2}
                                    </h4>
                                    <p style="margin: 5px 0;">
                                        <strong>🏆 Predicted Winner:</strong> 
                                        <span style="background-color: #fff3cd; padding: 3px 8px; border-radius: 4px; font-weight: bold;">
                                            {predicted_winner}
                                        </span>
                                    </p>
                                    <div style="display: flex; gap: 20px; margin-top: 10px;">
                                        <div>
                                            <strong>Model Probability:</strong> {model_prob:.1f}%
                                        </div>
                                        <div>
                                            <strong>Market Probability:</strong> {predicted_winner_market_prob:.1f}%
                                        </div>
                                        <div>
                                            <strong>Edge:</strong> <span style="color: {'#28a745' if predicted_winner_edge > 0 else '#dc3545'}; font-weight: bold;">{predicted_winner_edge:+.1f}%</span>
                                        </div>
                                    </div>
                                    {risk_html}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Download button
                            csv_output = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_output,
                                file_name=f"predictions_{model_name}.csv",
                                mime="text/csv"
                            )
                        
                        except Exception as e:
                            st.error(f"Error running predictions: {str(e)}")
                            logger.exception("Prediction error")
        
        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")
            logger.exception("CSV parsing error")

# Tab 2: Fighter Comparison
with tab2:
    st.header("Compare Two Fighters")
    st.markdown("Search for fighters by name. Matching fighters will appear as you type.")
    
    # Initialize database connection for fighter search
    @st.cache_resource
    def get_db():
        from database.db_manager import DatabaseManager
        return DatabaseManager()
    
    db = get_db()
    
    def search_fighters(query: str, limit: int = 20):
        """Search for fighters matching the query."""
        if not query or len(query) < 2:
            return []
        
        from database.schema import Fighter, Fight
        from sqlalchemy import or_
        
        session = db.get_session()
        try:
            # Search by name (case-insensitive)
            fighters = session.query(Fighter).filter(
                Fighter.name.ilike(f"%{query}%")
            ).limit(limit).all()
            
            # Get fight counts for each fighter
            results = []
            for fighter in fighters:
                fight_count = session.query(Fight).filter(
                    or_(Fight.fighter_1_id == fighter.id, Fight.fighter_2_id == fighter.id)
                ).count()
                
                record = f"{fighter.wins or 0}-{fighter.losses or 0}-{fighter.draws or 0}"
                results.append({
                    'id': fighter.id,
                    'name': fighter.name,
                    'nickname': fighter.nickname or '',
                    'record': record,
                    'ufcstats_id': fighter.fighter_id,
                    'age': fighter.age,
                    'fight_count': fight_count
                })
            
            # Sort by fight count (most active first), then by name
            results.sort(key=lambda x: (-x['fight_count'], x['name']))
            return results
        finally:
            session.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fighter 1")
        fighter_1_query = st.text_input(
            "Search Fighter 1",
            value="",
            key="fighter_1_search",
            help="Type to search for a fighter (e.g., 'Arn' for Arnold)"
        )
        
        fighter_1_id = None
        fighter_1_name = None
        
        if fighter_1_query:
            matches = search_fighters(fighter_1_query)
            
            if matches:
                # Create display strings for selectbox
                fighter_options = []
                for f in matches:
                    display = f"{f['name']}"
                    if f['nickname']:
                        display += f" '{f['nickname']}'"
                    display += f" (ID: {f['id']}, Record: {f['record']}"
                    if f['fight_count'] > 0:
                        display += f", {f['fight_count']} fights"
                    display += ")"
                    fighter_options.append((f['id'], display, f['name']))
                
                if len(matches) == 1:
                    # Auto-select if only one match
                    selected = fighter_options[0]
                    fighter_1_id = selected[0]
                    fighter_1_name = selected[2]
                    st.info(f"✅ Found: {selected[1]}")
                else:
                    # Show selectbox for multiple matches
                    selected_display = st.selectbox(
                        "Select Fighter 1",
                        options=[opt[1] for opt in fighter_options],
                        key="fighter_1_select",
                        help=f"Found {len(matches)} matches. Select the correct fighter."
                    )
                    
                    # Find selected fighter
                    for opt in fighter_options:
                        if opt[1] == selected_display:
                            fighter_1_id = opt[0]
                            fighter_1_name = opt[2]
                            break
            else:
                st.warning(f"No fighters found matching '{fighter_1_query}'")
        
        # Manual ID override
        manual_id_1 = st.number_input(
            "Or enter Fighter 1 ID manually",
            min_value=0,
            value=0,
            key="fighter_1_manual_id",
            help="Override with specific fighter ID"
        )
        if manual_id_1 > 0:
            fighter_1_id = int(manual_id_1)
            # Look up name
            from database.schema import Fighter
            session = db.get_session()
            try:
                f = session.query(Fighter).filter(Fighter.id == fighter_1_id).first()
                if f:
                    fighter_1_name = f.name
                    st.success(f"✅ Fighter: {f.name} (Record: {f.wins}-{f.losses}-{f.draws})")
                else:
                    st.error(f"Fighter ID {fighter_1_id} not found")
            finally:
                session.close()
    
    with col2:
        st.subheader("Fighter 2")
        fighter_2_query = st.text_input(
            "Search Fighter 2",
            value="",
            key="fighter_2_search",
            help="Type to search for a fighter (e.g., 'Arn' for Arnold)"
        )
        
        fighter_2_id = None
        fighter_2_name = None
        
        if fighter_2_query:
            matches = search_fighters(fighter_2_query)
            
            if matches:
                # Create display strings for selectbox
                fighter_options = []
                for f in matches:
                    display = f"{f['name']}"
                    if f['nickname']:
                        display += f" '{f['nickname']}'"
                    display += f" (ID: {f['id']}, Record: {f['record']}"
                    if f['fight_count'] > 0:
                        display += f", {f['fight_count']} fights"
                    display += ")"
                    fighter_options.append((f['id'], display, f['name']))
                
                if len(matches) == 1:
                    # Auto-select if only one match
                    selected = fighter_options[0]
                    fighter_2_id = selected[0]
                    fighter_2_name = selected[2]
                    st.info(f"✅ Found: {selected[1]}")
                else:
                    # Show selectbox for multiple matches
                    selected_display = st.selectbox(
                        "Select Fighter 2",
                        options=[opt[1] for opt in fighter_options],
                        key="fighter_2_select",
                        help=f"Found {len(matches)} matches. Select the correct fighter."
                    )
                    
                    # Find selected fighter
                    for opt in fighter_options:
                        if opt[1] == selected_display:
                            fighter_2_id = opt[0]
                            fighter_2_name = opt[2]
                            break
            else:
                st.warning(f"No fighters found matching '{fighter_2_query}'")
        
        # Manual ID override
        manual_id_2 = st.number_input(
            "Or enter Fighter 2 ID manually",
            min_value=0,
            value=0,
            key="fighter_2_manual_id",
            help="Override with specific fighter ID"
        )
        if manual_id_2 > 0:
            fighter_2_id = int(manual_id_2)
            # Look up name
            from database.schema import Fighter
            session = db.get_session()
            try:
                f = session.query(Fighter).filter(Fighter.id == fighter_2_id).first()
                if f:
                    fighter_2_name = f.name
                    st.success(f"✅ Fighter: {f.name} (Record: {f.wins}-{f.losses}-{f.draws})")
                else:
                    st.error(f"Fighter ID {fighter_2_id} not found")
            finally:
                session.close()
    
    is_title_fight = st.checkbox("Title Fight (5 rounds)", value=False)
    
    # Use the selected names or fall back to queries if names weren't selected
    fighter_1 = fighter_1_name if fighter_1_name else (fighter_1_query if fighter_1_query else "")
    fighter_2 = fighter_2_name if fighter_2_name else (fighter_2_query if fighter_2_query else "")
    
    if st.button("🥊 Predict Fight", type="primary"):
        if not fighter_1 or not fighter_2:
            st.warning("Please search and select both fighters")
        elif not fighter_1_id and not fighter_1_name:
            st.warning("Please select Fighter 1 from the search results")
        elif not fighter_2_id and not fighter_2_name:
            st.warning("Please select Fighter 2 from the search results")
        else:
            with st.spinner("Running prediction... This may take a moment."):
                try:
                    # Capture output from xgboost_predict
                    import io
                    import re
                    from contextlib import redirect_stdout, redirect_stderr
                    
                    output_buffer = io.StringIO()
                    error_buffer = io.StringIO()
                    
                    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                        xgboost_predict(
                            fighter_1_name=fighter_1,
                            fighter_2_name=fighter_2,
                            title_fight=is_title_fight,
                            quiet=False,
                            model_name=model_name,
                            fighter_1_id=fighter_1_id,
                            fighter_2_id=fighter_2_id,
                            allow_ambiguous=True,
                            symmetric=use_symmetric,
                        )
                    
                    output = output_buffer.getvalue()
                    errors = error_buffer.getvalue()
                    
                    if errors:
                        st.warning(f"Warnings: {errors}")
                    
                    # Parse the output to extract key information
                    # Extract prediction percentages
                    prob_pattern = r'(\w+(?:\s+\w+)*):\s+(\d+\.\d+)% chance to win'
                    probabilities = re.findall(prob_pattern, output)
                    
                    # Extract predicted winner
                    winner_pattern = r'⭐ Predicted Winner:\s+(.+)'
                    winner_match = re.search(winner_pattern, output)
                    predicted_winner = winner_match.group(1).strip() if winner_match else None
                    
                    # Extract fight type
                    fight_type_pattern = r'Fight Type:\s+(.+)'
                    fight_type_match = re.search(fight_type_pattern, output)
                    fight_type = fight_type_match.group(1).strip() if fight_type_match else "Unknown"
                    
                    # Display main prediction in a nice format
                    st.markdown("---")
                    st.markdown("### 🥊 Prediction Results")
                    
                    # Create columns for fighter probabilities
                    col1, col2 = st.columns(2)
                    
                    f1_prob = None
                    f2_prob = None
                    f1_name_display = fighter_1
                    f2_name_display = fighter_2
                    
                    for name, prob in probabilities:
                        prob_float = float(prob)
                        if name.strip() == fighter_1 or name.strip() in fighter_1:
                            f1_prob = prob_float
                            f1_name_display = name.strip()
                        elif name.strip() == fighter_2 or name.strip() in fighter_2:
                            f2_prob = prob_float
                            f2_name_display = name.strip()
                    
                    # Display fighter 1
                    with col1:
                        is_winner_1 = predicted_winner and (f1_name_display in predicted_winner or predicted_winner in f1_name_display)
                        winner_badge = " 🏆 WINNER" if is_winner_1 else ""
                        color = "#28a745" if is_winner_1 else "#6c757d"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px; background-color: {'#d4edda' if is_winner_1 else '#f8f9fa'};">
                            <h2 style="color: {color}; margin-bottom: 10px;">{f1_name_display}{winner_badge}</h2>
                            <h1 style="color: {color}; font-size: 48px; margin: 10px 0;">{f1_prob:.1f}%</h1>
                            <p style="color: #666;">Win Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display fighter 2
                    with col2:
                        is_winner_2 = predicted_winner and (f2_name_display in predicted_winner or predicted_winner in f2_name_display)
                        winner_badge = " 🏆 WINNER" if is_winner_2 else ""
                        color = "#28a745" if is_winner_2 else "#6c757d"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px; background-color: {'#d4edda' if is_winner_2 else '#f8f9fa'};">
                            <h2 style="color: {color}; margin-bottom: 10px;">{f2_name_display}{winner_badge}</h2>
                            <h1 style="color: {color}; font-size: 48px; margin: 10px 0;">{f2_prob:.1f}%</h1>
                            <p style="color: #666;">Win Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Fight type
                    st.markdown(f"**Fight Type:** {fight_type}")
                    
                    # Extract top features
                    top_features_pattern = r'Top \d+ Most Important Features in Model:\s*\n((?:\s+\d+\.\s+[^\n]+\n?)+)'
                    top_features_match = re.search(top_features_pattern, output)
                    
                    if top_features_match:
                        st.markdown("---")
                        st.markdown("### 📊 Top Features")
                        features_text = top_features_match.group(1)
                        # Clean up the features list
                        features_list = [line.strip() for line in features_text.split('\n') if line.strip() and '.' in line]
                        for feature in features_list[:5]:  # Show top 5
                            # Remove the number prefix
                            feature_clean = re.sub(r'^\s*\d+\.\s*', '', feature)
                            st.markdown(f"- **{feature_clean}**")
                    
                    # Extract feature contribution analysis
                    contribution_pattern = r'\[FEATURE CONTRIBUTION ANALYSIS\](.*?)(?=\n\n|\Z)'
                    contribution_match = re.search(contribution_pattern, output, re.DOTALL)
                    
                    if contribution_match:
                        st.markdown("---")
                        st.markdown("### 🔍 Feature Analysis")
                        contribution_text = contribution_match.group(1)
                        
                        # Extract features favoring each fighter - use the actual names from output
                        # Look for "Top 3 favoring [name]:" pattern
                        f1_favors_pattern = rf'Top 3 favoring {re.escape(f1_name_display)}:\s*\n((?:.*\n)*?)(?=\n\s*Top 3 favoring|\n\s*\[|\Z)'
                        f1_favors_match = re.search(f1_favors_pattern, contribution_text, re.DOTALL)
                        
                        f2_favors_pattern = rf'Top 3 favoring {re.escape(f2_name_display)}:\s*\n((?:.*\n)*?)(?=\n\s*\[|\Z)'
                        f2_favors_match = re.search(f2_favors_pattern, contribution_text, re.DOTALL)
                        
                        col_f1, col_f2 = st.columns(2)
                        
                        with col_f1:
                            st.markdown(f"**Favoring {f1_name_display}:**")
                            if f1_favors_match:
                                favors_text = f1_favors_match.group(1)
                                favors_lines = [line.strip() for line in favors_text.split('\n') if line.strip() and ':' in line]
                                for line in favors_lines[:3]:
                                    # Format: feature_name: value (description)
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        feature = parts[0].strip()
                                        rest = ':'.join(parts[1:]).strip()
                                        st.markdown(f"  • **{feature}**: {rest}")
                        
                        with col_f2:
                            st.markdown(f"**Favoring {f2_name_display}:**")
                            if f2_favors_match:
                                favors_text = f2_favors_match.group(1)
                                favors_lines = [line.strip() for line in favors_text.split('\n') if line.strip() and ':' in line]
                                for line in favors_lines[:3]:
                                    # Format: feature_name: value (description)
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        feature = parts[0].strip()
                                        rest = ':'.join(parts[1:]).strip()
                                        st.markdown(f"  • **{feature}**: {rest}")
                    
                    # Full output in expandable section
                    with st.expander("📋 Full Prediction Details (Debug Output)", expanded=False):
                        st.code(output, language=None)
                
                except Exception as e:
                    st.error(f"Error running prediction: {str(e)}")
                    logger.exception("Fighter comparison error")

# Tab 3: Model Evaluation
with tab3:
    st.header("📈 Model Evaluation Report")
    st.markdown("Generate and view model evaluation metrics and performance analysis for the selected model.")
    
    # Check if feature pipeline files exist for the selected model
    models_dir = PROJECT_ROOT / "models" / "saved"
    scaler_path = models_dir / f"{model_name}_feature_scaler.pkl"
    features_path = models_dir / f"{model_name}_feature_names.pkl"
    model_path = models_dir / f"{model_name}.json"
    
    if model_path.exists():
        if not (scaler_path.exists() and features_path.exists()):
            st.warning(
                f"⚠️ **Feature pipeline files missing for model '{model_name}'**\n\n"
                f"The model file exists, but the feature pipeline files are missing:\n"
                f"- `{scaler_path.name}`\n"
                f"- `{features_path.name}`\n\n"
                f"**This will cause errors when generating the evaluation report.**\n\n"
                f"**Solution:** Ensure these files are committed to your repository. "
                f"They should be saved automatically when you train the model. "
                f"If they're missing, you'll need to retrain the model or rebuild the feature pipeline."
            )
    else:
        st.error(f"Model file not found: {model_path}")
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        min_year = st.number_input(
            "Minimum Year",
            min_value=2020,
            max_value=2030,
            value=2025,
            help="Minimum event year to include in evaluation"
        )
    with col2:
        odds_date_tolerance = st.number_input(
            "Odds Date Tolerance (days)",
            min_value=0,
            max_value=10,
            value=5,
            help="Allow matching odds rows even if date is off by N days"
        )
    
    # Check if we have cached results for this model
    cache_key = f"eval_report_{model_name}_{min_year}"
    html_content = None
    
    # Check for cached HTML in session state
    if cache_key in st.session_state:
        st.info(f"📋 Using cached evaluation report for model: **{model_name}**")
        html_content = st.session_state[cache_key]
    
    # Button to generate/regenerate report
    if st.button("🚀 Generate Evaluation Report", type="primary"):
        with st.spinner("Generating evaluation report... This may take a few minutes."):
            try:
                import tempfile
                import subprocess
                from datetime import datetime
                from evaluation.generate_html_report import generate_html_report
                
                # Create temporary directory for outputs
                temp_dir = Path(tempfile.mkdtemp())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Prepare command to run evaluate_model
                cmd = [
                    sys.executable,
                    "-m",
                    "evaluation.evaluate_model",
                    "--model-name", model_name,
                    "--data-path", str(PROJECT_ROOT / "data" / "processed" / "training_data.csv"),
                    "--odds-path", str(PROJECT_ROOT / "ufc_2025_odds.csv"),
                    "--min-year", str(int(min_year)),
                    "--output-dir", str(temp_dir),
                    "--odds-date-tolerance-days", str(int(odds_date_tolerance)),
                ]
                
                if use_symmetric:
                    cmd.append("--symmetric")
                else:
                    cmd.append("--no-symmetric")
                
                # Run evaluation as subprocess
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT)
                )
                
                # Check for errors
                if result.returncode != 0:
                    st.error(f"Evaluation failed with return code {result.returncode}")
                    st.code(result.stderr)
                    logger.error(f"Evaluation stderr: {result.stderr}")
                else:
                    # Find the generated eval data CSV
                    eval_data_files = list(temp_dir.glob("eval_data_*.csv"))
                    if not eval_data_files:
                        st.error("Evaluation completed but no data file was generated.")
                        st.code(result.stdout)
                        logger.warning(f"Evaluation stdout: {result.stdout}")
                    else:
                        # Use the most recent eval data file
                        eval_data_path = sorted(eval_data_files)[-1]
                        
                        # Generate HTML report
                        html_path = temp_dir / f"model_evaluation_{timestamp}.html"
                        generate_html_report(
                            eval_data_path=eval_data_path,
                            output_path=html_path,
                            min_year=int(min_year)
                        )
                        
                        # Read the generated HTML
                        if html_path.exists():
                            html_content = html_path.read_text()
                            # Cache in session state
                            st.session_state[cache_key] = html_content
                            st.success(f"✅ Evaluation report generated successfully for model: **{model_name}**")
                        else:
                            st.error("HTML report file was not created.")
                            
            except Exception as e:
                st.error(f"Error generating evaluation report: {str(e)}")
                logger.exception("Evaluation report generation error")
                import traceback
                st.code(traceback.format_exc())
    
    # Re-check session state after potential generation (in case report was just generated)
    if cache_key in st.session_state and html_content is None:
        html_content = st.session_state[cache_key]
    
    # Display the report if available
    if html_content:
        st.markdown("---")
        st.subheader(f"Evaluation Report: {model_name}")
        
        # Display HTML
        st.components.v1.html(html_content, height=800, scrolling=True)
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=html_content,
            file_name=f"model_evaluation_{model_name}_{min_year}.html",
            mime="text/html"
        )
    else:
        st.info("👆 Click 'Generate Evaluation Report' to create a report for the selected model.")
        
        # Show fallback to static file if it exists
        evaluation_file = PROJECT_ROOT / "reports_strict" / "model_evaluation_latest.html"
        if evaluation_file.exists():
            st.markdown("---")
            st.markdown("### 📄 Static Report (Fallback)")
            st.info(f"Found a static report at: {evaluation_file}")
            st.warning("This is a pre-generated report and may not match the selected model.")
            
            if st.button("📖 View Static Report"):
                try:
                    static_html = evaluation_file.read_text()
                    st.components.v1.html(static_html, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error loading static report: {str(e)}")

# Tab 4: Fighter Search
with tab4:
    st.header("🔍 Fighter Search & Statistics")
    st.markdown("Search for a fighter to view their statistics and complete fight history.")
    
    # Comparison mode toggle
    comparison_mode = st.checkbox(
        "🔀 Enable Comparison Mode",
        value=False,
        key="comparison_mode_tab4",
        help="Enable to compare two fighters side-by-side"
    )
    
    # Initialize database connection
    db = get_db()
    
    def search_fighters_for_tab(query: str, limit: int = 20):
        """Search for fighters matching the query."""
        if not query or len(query) < 2:
            return []
        
        from database.schema import Fighter, Fight
        from sqlalchemy import or_
        
        session = db.get_session()
        try:
            # Search by name (case-insensitive)
            fighters = session.query(Fighter).filter(
                Fighter.name.ilike(f"%{query}%")
            ).limit(limit).all()
            
            # Get fight counts for each fighter
            results = []
            for fighter in fighters:
                fight_count = session.query(Fight).filter(
                    or_(Fight.fighter_1_id == fighter.id, Fight.fighter_2_id == fighter.id)
                ).count()
                
                record = f"{fighter.wins or 0}-{fighter.losses or 0}-{fighter.draws or 0}"
                results.append({
                    'id': fighter.id,
                    'name': fighter.name,
                    'nickname': fighter.nickname or '',
                    'record': record,
                    'ufcstats_id': fighter.fighter_id,
                    'age': fighter.age,
                    'fight_count': fight_count
                })
            
            # Sort by fight count (most active first), then by name
            results.sort(key=lambda x: (-x['fight_count'], x['name']))
            return results
        finally:
            session.close()
    
    def display_fighter_info(fighter_id: int, fighter_name: str, col=None):
        """Display fighter information in a column or full width."""
        from database.schema import Fighter, Fight, Event
        from sqlalchemy import or_
        
        session = db.get_session()
        try:
            fighter = session.query(Fighter).filter(Fighter.id == fighter_id).first()
            
            if not fighter:
                if col:
                    col.error("Fighter not found.")
                else:
                    st.error("Fighter not found.")
                return
            
            # Fighter Statistics Section
            if col:
                col.markdown(f"### 🥊 {fighter.name}")
            else:
                st.markdown("---")
                st.subheader(f"🥊 {fighter.name}")
            
            if fighter.nickname:
                if col:
                    col.markdown(f"*'{fighter.nickname}'*")
                else:
                    st.markdown(f"*'{fighter.nickname}'*")
            
            # Basic Info
            if col:
                info_cols = col.columns(4)
            else:
                info_cols = st.columns(4)
            
            with info_cols[0]:
                st.metric("Record", f"{fighter.wins or 0}-{fighter.losses or 0}-{fighter.draws or 0}")
            
            with info_cols[1]:
                total_fights = (fighter.wins or 0) + (fighter.losses or 0) + (fighter.draws or 0)
                st.metric("Total Fights", total_fights)
            
            with info_cols[2]:
                win_rate = (fighter.wins or 0) / total_fights * 100 if total_fights > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with info_cols[3]:
                if fighter.age:
                    st.metric("Age", fighter.age)
            
            # Physical Attributes
            if col:
                col.markdown("#### Physical Attributes")
                phys_cols = col.columns(4)
            else:
                st.markdown("### Physical Attributes")
                phys_cols = st.columns(4)
            
            with phys_cols[0]:
                if fighter.height_cm:
                    height_ft = fighter.height_cm / 30.48
                    st.metric("Height", f"{height_ft:.1f} ft ({fighter.height_cm:.0f} cm)")
            
            with phys_cols[1]:
                if fighter.weight_lbs:
                    st.metric("Weight", f"{fighter.weight_lbs:.1f} lbs")
            
            with phys_cols[2]:
                if fighter.reach_inches:
                    st.metric("Reach", f"{fighter.reach_inches:.1f} in")
            
            with phys_cols[3]:
                if fighter.stance:
                    st.metric("Stance", fighter.stance)
            
            # Career Statistics
            if col:
                col.markdown("#### Career Statistics")
                stats_cols = col.columns(4)
            else:
                st.markdown("### Career Statistics")
                stats_cols = st.columns(4)
            
            with stats_cols[0]:
                if fighter.sig_strikes_landed_per_min:
                    st.metric("Sig Strikes/Min", f"{fighter.sig_strikes_landed_per_min:.2f}")
            
            with stats_cols[1]:
                if fighter.striking_accuracy:
                    st.metric("Striking Accuracy", f"{fighter.striking_accuracy:.1f}%")
            
            with stats_cols[2]:
                if fighter.takedown_avg_per_15min:
                    st.metric("Takedowns/15min", f"{fighter.takedown_avg_per_15min:.2f}")
            
            with stats_cols[3]:
                if fighter.takedown_accuracy:
                    st.metric("TD Accuracy", f"{fighter.takedown_accuracy:.1f}%")
            
            # Fight History
            if col:
                col.markdown("---")
                col.subheader("📋 Fight History")
            else:
                st.markdown("---")
                st.subheader("📋 Fight History")
            
            # Get all fights for this fighter
            fights = session.query(Fight).join(Event).filter(
                or_(Fight.fighter_1_id == fighter.id, Fight.fighter_2_id == fighter.id)
            ).order_by(Event.date).all()
            
            if fights:
                # Create fight history dataframe
                fight_data = []
                for fight in fights:
                    # Determine opponent
                    if fight.fighter_1_id == fighter.id:
                        opponent = fight.fighter_2
                        was_fighter_1 = True
                    else:
                        opponent = fight.fighter_1
                        was_fighter_1 = False
                    
                    # Determine result
                    if fight.result == 'draw':
                        result = "Draw"
                        result_icon = "🤝"
                    elif fight.result == 'no_contest':
                        result = "No Contest"
                        result_icon = "❌"
                    elif fight.winner_id == fighter.id:
                        result = "Win"
                        result_icon = "✅"
                    elif fight.winner_id == opponent.id if opponent else None:
                        result = "Loss"
                        result_icon = "❌"
                    else:
                        result = "Unknown"
                        result_icon = "❓"
                    
                    # Store date as-is (will convert to datetime later for sorting)
                    event_date = None
                    if fight.event and fight.event.date:
                        event_date = fight.event.date
                    
                    fight_data.append({
                        "Date": event_date,
                        "Event": fight.event.name if fight.event else "N/A",
                        "Opponent": opponent.name if opponent else "Unknown",
                        "Weight Class": fight.weight_class or "N/A",
                        "Result": f"{result_icon} {result}",
                        "Method": fight.method or "N/A",
                        "Round": fight.round_finished if fight.round_finished else "N/A",
                    })
                
                fight_df = pd.DataFrame(fight_data)
                
                # Convert Date column to datetime for proper sorting
                if "Date" in fight_df.columns:
                    # Convert to datetime, handling both string and datetime objects
                    fight_df["Date"] = pd.to_datetime(fight_df["Date"], errors='coerce')
                    # Sort by date (chronological order - oldest first)
                    fight_df = fight_df.sort_values("Date", na_position='last')
                    # Keep as datetime so Streamlit can sort it properly
                    # Streamlit will display datetime objects nicely
                
                if col:
                    col.dataframe(fight_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(fight_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                if col:
                    col.markdown("#### Fight History Summary")
                    summary_cols = col.columns(3)
                else:
                    st.markdown("### Fight History Summary")
                    summary_cols = st.columns(3)
                
                wins = sum(1 for f in fights if f.winner_id == fighter.id)
                losses = sum(1 for f in fights if f.winner_id and f.winner_id != fighter.id and (f.fighter_1_id == fighter.id or f.fighter_2_id == fighter.id))
                draws = sum(1 for f in fights if f.result == 'draw')
                
                with summary_cols[0]:
                    st.metric("Wins in Database", wins)
                
                with summary_cols[1]:
                    st.metric("Losses in Database", losses)
                
                with summary_cols[2]:
                    st.metric("Draws/NC", draws)
            else:
                if col:
                    col.info("No fight history found in database for this fighter.")
                else:
                    st.info("No fight history found in database for this fighter.")
        finally:
            session.close()
    
    # Search interface - single or dual depending on comparison mode
    if comparison_mode:
        col1_search, col2_search = st.columns(2)
        
        with col1_search:
            st.subheader("Fighter 1")
            fighter_1_query = st.text_input(
                "Search Fighter 1",
                value="",
                key="fighter_1_search_tab4",
                help="Type to search for the first fighter"
            )
        
        with col2_search:
            st.subheader("Fighter 2")
            fighter_2_query = st.text_input(
                "Search Fighter 2",
                value="",
                key="fighter_2_search_tab4",
                help="Type to search for the second fighter"
            )
        
        selected_fighter_1_id = None
        selected_fighter_1_name = None
        selected_fighter_2_id = None
        selected_fighter_2_name = None
        
        # Fighter 1 selection
        if fighter_1_query:
            matches = search_fighters_for_tab(fighter_1_query)
            if matches:
                fighter_options = []
                for f in matches:
                    display = f"{f['name']}"
                    if f['nickname']:
                        display += f" '{f['nickname']}'"
                    display += f" (ID: {f['id']}, Record: {f['record']}"
                    if f['fight_count'] > 0:
                        display += f", {f['fight_count']} fights"
                    display += ")"
                    fighter_options.append((f['id'], display, f['name']))
                
                if len(matches) == 1:
                    selected = fighter_options[0]
                    selected_fighter_1_id = selected[0]
                    selected_fighter_1_name = selected[2]
                else:
                    selected_display = st.selectbox(
                        "Select Fighter 1",
                        options=[opt[1] for opt in fighter_options],
                        key="fighter_1_select_tab4"
                    )
                    for opt in fighter_options:
                        if opt[1] == selected_display:
                            selected_fighter_1_id = opt[0]
                            selected_fighter_1_name = opt[2]
                            break
        
        # Fighter 2 selection
        if fighter_2_query:
            matches = search_fighters_for_tab(fighter_2_query)
            if matches:
                fighter_options = []
                for f in matches:
                    display = f"{f['name']}"
                    if f['nickname']:
                        display += f" '{f['nickname']}'"
                    display += f" (ID: {f['id']}, Record: {f['record']}"
                    if f['fight_count'] > 0:
                        display += f", {f['fight_count']} fights"
                    display += ")"
                    fighter_options.append((f['id'], display, f['name']))
                
                if len(matches) == 1:
                    selected = fighter_options[0]
                    selected_fighter_2_id = selected[0]
                    selected_fighter_2_name = selected[2]
                else:
                    selected_display = st.selectbox(
                        "Select Fighter 2",
                        options=[opt[1] for opt in fighter_options],
                        key="fighter_2_select_tab4"
                    )
                    for opt in fighter_options:
                        if opt[1] == selected_display:
                            selected_fighter_2_id = opt[0]
                            selected_fighter_2_name = opt[2]
                            break
        
        # Display both fighters side-by-side
        if selected_fighter_1_id and selected_fighter_2_id:
            st.markdown("---")
            col1_display, col2_display = st.columns(2)
            with col1_display:
                display_fighter_info(selected_fighter_1_id, selected_fighter_1_name, col=col1_display)
            with col2_display:
                display_fighter_info(selected_fighter_2_id, selected_fighter_2_name, col=col2_display)
            
            # Common Opponents Analysis
            st.markdown("---")
            st.subheader("🔗 Common Opponents & Transitive Analysis")
            
            def analyze_common_opponents(fighter_1_id: int, fighter_2_id: int):
                """Find common opponents and transitive connections."""
                from database.schema import Fighter, Fight, Event
                from sqlalchemy import or_
                from collections import defaultdict
                
                session = db.get_session()
                try:
                    # Get all fights for both fighters
                    f1_fights = session.query(Fight).filter(
                        or_(Fight.fighter_1_id == fighter_1_id, Fight.fighter_2_id == fighter_1_id)
                    ).all()
                    
                    f2_fights = session.query(Fight).filter(
                        or_(Fight.fighter_1_id == fighter_2_id, Fight.fighter_2_id == fighter_2_id)
                    ).all()
                    
                    # Build opponent sets and results
                    f1_opponents = {}  # opponent_id -> (result, fight)
                    f2_opponents = {}  # opponent_id -> (result, fight)
                    
                    for fight in f1_fights:
                        if fight.fighter_1_id == fighter_1_id:
                            opponent_id = fight.fighter_2_id
                        else:
                            opponent_id = fight.fighter_1_id
                        
                        if fight.winner_id == fighter_1_id:
                            result = "Win"
                        elif fight.winner_id == opponent_id:
                            result = "Loss"
                        elif fight.result == 'draw':
                            result = "Draw"
                        else:
                            result = "Unknown"
                        
                        f1_opponents[opponent_id] = (result, fight)
                    
                    for fight in f2_fights:
                        if fight.fighter_1_id == fighter_2_id:
                            opponent_id = fight.fighter_2_id
                        else:
                            opponent_id = fight.fighter_1_id
                        
                        if fight.winner_id == fighter_2_id:
                            result = "Win"
                        elif fight.winner_id == opponent_id:
                            result = "Loss"
                        elif fight.result == 'draw':
                            result = "Draw"
                        else:
                            result = "Unknown"
                        
                        f2_opponents[opponent_id] = (result, fight)
                    
                    # Find common opponents
                    common_opponent_ids = set(f1_opponents.keys()) & set(f2_opponents.keys())
                    
                    # Build fight graph for transitive analysis
                    # Get all fights to build a win/loss graph
                    all_fights = session.query(Fight).filter(
                        Fight.winner_id.isnot(None)
                    ).all()
                    
                    # Build adjacency list: fighter_id -> {opponents they beat}
                    win_graph = defaultdict(set)
                    loss_graph = defaultdict(set)
                    
                    for fight in all_fights:
                        if fight.winner_id:
                            if fight.fighter_1_id == fight.winner_id:
                                loser_id = fight.fighter_2_id
                            else:
                                loser_id = fight.fighter_1_id
                            
                            win_graph[fight.winner_id].add(loser_id)
                            loss_graph[loser_id].add(fight.winner_id)
                    
                    # Find transitive connections (up to 3 degrees of separation)
                    # We want to find paths like: Fighter1 beat A, A beat B, B beat Fighter2
                    def find_transitive_paths(start_id: int, target_id: int, max_depth: int = 3):
                        """Find paths from start to target through fight history."""
                        paths = []
                        visited = set()
                        
                        def dfs(current_id: int, path: list, depth: int):
                            if depth > max_depth or current_id in visited:
                                return
                            
                            visited.add(current_id)
                            
                            if current_id == target_id and len(path) > 0:
                                paths.append(path.copy())
                                visited.remove(current_id)
                                return
                            
                            # Follow wins: if current fighter beat someone, that someone might have connections
                            for beaten_id in win_graph.get(current_id, []):
                                if beaten_id not in visited:
                                    path.append(('beat', beaten_id))
                                    dfs(beaten_id, path, depth + 1)
                                    path.pop()
                            
                            # Follow losses: if current fighter lost to someone, that someone might have connections
                            for lost_to_id in loss_graph.get(current_id, []):
                                if lost_to_id not in visited:
                                    path.append(('lost_to', lost_to_id))
                                    dfs(lost_to_id, path, depth + 1)
                                    path.pop()
                            
                            visited.remove(current_id)
                        
                        dfs(start_id, [], 0)
                        # Sort by path length (shorter paths first)
                        paths.sort(key=len)
                        return paths[:10]  # Return top 10 shortest paths
                    
                    transitive_paths = find_transitive_paths(fighter_1_id, fighter_2_id)
                    
                    return common_opponent_ids, f1_opponents, f2_opponents, transitive_paths, session
                except Exception as e:
                    logger.exception(f"Error analyzing common opponents: {e}")
                    return set(), {}, {}, [], session
            
            common_opp_ids, f1_opps, f2_opps, transitive_paths, session = analyze_common_opponents(
                selected_fighter_1_id, selected_fighter_2_id
            )
            
            try:
                from database.schema import Fighter
                
                # Display common opponents
                if common_opp_ids:
                    st.markdown("### 📊 Direct Common Opponents")
                    
                    common_opp_data = []
                    for opp_id in common_opp_ids:
                        opp = session.query(Fighter).filter(Fighter.id == opp_id).first()
                        if opp:
                            f1_result, f1_fight = f1_opps[opp_id]
                            f2_result, f2_fight = f2_opps[opp_id]
                            
                            # Get fight dates
                            f1_date = "N/A"
                            f2_date = "N/A"
                            if f1_fight and f1_fight.event:
                                f1_date = f1_fight.event.date if isinstance(f1_fight.event.date, str) else f1_fight.event.date.strftime("%Y-%m-%d") if hasattr(f1_fight.event.date, 'strftime') else str(f1_fight.event.date)
                            if f2_fight and f2_fight.event:
                                f2_date = f2_fight.event.date if isinstance(f2_fight.event.date, str) else f2_fight.event.date.strftime("%Y-%m-%d") if hasattr(f2_fight.event.date, 'strftime') else str(f2_fight.event.date)
                            
                            # Determine advantage
                            if f1_result == "Win" and f2_result == "Loss":
                                advantage = f"{selected_fighter_1_name} (beat opponent, {selected_fighter_2_name} lost)"
                            elif f1_result == "Loss" and f2_result == "Win":
                                advantage = f"{selected_fighter_2_name} (beat opponent, {selected_fighter_1_name} lost)"
                            elif f1_result == "Win" and f2_result == "Win":
                                advantage = "Both won"
                            elif f1_result == "Loss" and f2_result == "Loss":
                                advantage = "Both lost"
                            else:
                                advantage = "Inconclusive"
                            
                            common_opp_data.append({
                                "Common Opponent": opp.name,
                                f"{selected_fighter_1_name} Result": f1_result,
                                f"{selected_fighter_1_name} Date": f1_date,
                                f"{selected_fighter_2_name} Result": f2_result,
                                f"{selected_fighter_2_name} Date": f2_date,
                                "Analysis": advantage
                            })
                    
                    if common_opp_data:
                        common_opp_df = pd.DataFrame(common_opp_data)
                        st.dataframe(common_opp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No direct common opponents found in fight history.")
                
                # Display transitive connections
                if transitive_paths:
                    st.markdown("### 🔄 Transitive Connections")
                    st.markdown("These show how the fighters are connected through their fight history (e.g., Fighter 1 beat A, A beat B, B beat Fighter 2).")
                    
                    transitive_data = []
                    for path in transitive_paths[:10]:  # Limit to top 10 paths
                        path_str_parts = [selected_fighter_1_name]
                        for i, (relation, fighter_id) in enumerate(path):
                            fighter = session.query(Fighter).filter(Fighter.id == fighter_id).first()
                            if fighter:
                                if relation == 'beat':
                                    path_str_parts.append(f"beat {fighter.name}")
                                else:
                                    path_str_parts.append(f"lost to {fighter.name}")
                        
                        path_str_parts.append(selected_fighter_2_name)
                        path_str = " → ".join(path_str_parts)
                        
                        transitive_data.append({
                            "Connection Path": path_str,
                            "Path Length": len(path) + 1
                        })
                    
                    if transitive_data:
                        transitive_df = pd.DataFrame(transitive_data)
                        st.dataframe(transitive_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No transitive connections found (fighters are not connected through common opponents in fight history).")
                    
            finally:
                session.close()
        elif selected_fighter_1_id:
            st.markdown("---")
            display_fighter_info(selected_fighter_1_id, selected_fighter_1_name)
        elif selected_fighter_2_id:
            st.markdown("---")
            display_fighter_info(selected_fighter_2_id, selected_fighter_2_name)
    else:
        # Single fighter search (original behavior)
        fighter_query = st.text_input(
            "Search for a fighter",
            value="",
            key="fighter_search_tab4",
            help="Type to search for a fighter (e.g., 'Jon Jones', 'Silva', etc.)"
        )
        
        selected_fighter_id = None
        selected_fighter_name = None
        
        if fighter_query:
            matches = search_fighters_for_tab(fighter_query)
            
            if matches:
                # Create display strings for selectbox
                fighter_options = []
                for f in matches:
                    display = f"{f['name']}"
                    if f['nickname']:
                        display += f" '{f['nickname']}'"
                    display += f" (ID: {f['id']}, Record: {f['record']}"
                    if f['fight_count'] > 0:
                        display += f", {f['fight_count']} fights"
                    display += ")"
                    fighter_options.append((f['id'], display, f['name']))
                
                if len(matches) == 1:
                    # Auto-select if only one match
                    selected = fighter_options[0]
                    selected_fighter_id = selected[0]
                    selected_fighter_name = selected[2]
                else:
                    # Show selectbox for multiple matches
                    selected_display = st.selectbox(
                        "Select Fighter",
                        options=[opt[1] for opt in fighter_options],
                        key="fighter_select_tab4",
                        help=f"Found {len(matches)} matches. Select the fighter to view."
                    )
                    
                    # Find selected fighter
                    for opt in fighter_options:
                        if opt[1] == selected_display:
                            selected_fighter_id = opt[0]
                            selected_fighter_name = opt[2]
                            break
        
        # Display fighter information
        if selected_fighter_id:
            display_fighter_info(selected_fighter_id, selected_fighter_name)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        UFC Analysis v2 - Prediction Interface
    </div>
    """,
    unsafe_allow_html=True
)

# Handle command-line arguments for running with python instead of streamlit run
if __name__ == "__main__":
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description="Run UFC Prediction Streamlit App")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run Streamlit on (default: localhost)"
    )
    
    args = parser.parse_args()
    
    # Launch Streamlit via subprocess
    app_path = str(Path(__file__).absolute())
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    
    # Run Streamlit
    sys.exit(subprocess.run(cmd).returncode)

