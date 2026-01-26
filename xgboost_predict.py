#!/usr/bin/env python3
"""
XGBoost Quick Predict - Fast predictions using XGBoost model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database.db_manager import DatabaseManager
from database.schema import Fighter, Fight, Event
from features.matchup_features import MatchupFeatureExtractor
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel
import pandas as pd
from loguru import logger
import argparse
from sqlalchemy import or_, desc
from datetime import datetime


def _print_fighter_candidates(candidates, session=None):
    """Pretty-print ambiguous fighter matches to help disambiguate."""
    print("\n[AMBIGUOUS FIGHTER MATCH]")
    print("Multiple fighters matched your query. Use --fighter-*-id or --fighter-*-ufcstats-id to disambiguate.\n")
    for f in candidates:
        # Optionally show how many fights we have for this fighter (helps pick the active UFC fighter)
        fights_count = None
        try:
            if session is not None:
                from database.schema import Fight
                fights_count = session.query(Fight).filter(
                    or_(Fight.fighter_1_id == f.id, Fight.fighter_2_id == f.id)
                ).count()
        except Exception:
            fights_count = None

        record = f"{f.wins or 0}-{f.losses or 0}-{f.draws or 0}"
        fights_suffix = f", fights_in_db={fights_count}" if fights_count is not None else ""
        print(
            f"- id={f.id}, ufcstats_id={f.fighter_id}, name='{f.name}', nickname='{f.nickname or ''}', "
            f"dob='{f.date_of_birth or ''}', age={f.age}, record={record}{fights_suffix}"
        )
    print("")


def _resolve_fighter(
    session,
    name: str,
    *,
    db_id: int | None = None,
    ufcstats_id: str | None = None,
    allow_ambiguous: bool = False,
):
    """
    Resolve a fighter from CLI inputs.

    Priority:
    1) db_id (fighters.id)
    2) ufcstats_id (fighters.fighter_id)
    3) fuzzy name match (fighters.name ILIKE %name%)
       - if multiple matches: error (unless allow_ambiguous=True)
    """
    if db_id is not None:
        fighter = session.query(Fighter).filter(Fighter.id == db_id).first()
        return fighter, []

    if ufcstats_id is not None:
        fighter = session.query(Fighter).filter(Fighter.fighter_id == ufcstats_id).first()
        return fighter, []

    candidates = session.query(Fighter).filter(Fighter.name.ilike(f"%{name}%")).all()
    if len(candidates) == 0:
        return None, []
    if len(candidates) == 1:
        return candidates[0], []

    # Multiple matches: either fail loudly (default) or pick a "best" candidate with a warning.
    if not allow_ambiguous:
        return None, candidates

    # Heuristic: prefer the candidate with the most fights in our DB (then lowest id).
    from database.schema import Fight
    scored = []
    for f in candidates:
        cnt = session.query(Fight).filter(or_(Fight.fighter_1_id == f.id, Fight.fighter_2_id == f.id)).count()
        scored.append((cnt, f.id, f))
    scored.sort(key=lambda t: (-t[0], t[1]))
    best = scored[0][2]
    return best, candidates


def _show_fighter_recent_fights(session, fighter_id: int, fighter_name: str, num_fights: int):
    """Display the last X fights for a fighter with win/loss information."""
    # Query fights where this fighter participated
    # Fetch more than needed in case some dates can't be parsed
    all_fights = (
        session.query(Fight, Event)
        .join(Event, Fight.event_id == Event.id)
        .filter(
            or_(Fight.fighter_1_id == fighter_id, Fight.fighter_2_id == fighter_id)
        )
        .all()
    )
    
    if not all_fights:
        print(f"  No fights found in database for {fighter_name}")
        return
    
    # Parse dates and sort chronologically (most recent first)
    def parse_date(date_str):
        """Try to parse various date formats."""
        if not date_str:
            return datetime.min
        # Try common date formats
        formats = [
            "%B %d, %Y",  # "September 28, 2024"
            "%Y-%m-%d",   # "2024-09-28"
            "%m/%d/%Y",    # "09/28/2024"
            "%d/%m/%Y",    # "28/09/2024"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        # If no format works, return min date (will sort last)
        return datetime.min
    
    # Sort by parsed date (most recent first)
    fights_with_dates = []
    for fight, event in all_fights:
        event_date = parse_date(event.date) if event.date else datetime.min
        fights_with_dates.append((event_date, fight, event))
    
    # Sort by date descending (most recent first) and take top N
    fights_with_dates.sort(key=lambda x: x[0], reverse=True)
    fights = [(fight, event) for _, fight, event in fights_with_dates[:num_fights]]
    
    print(f"\n  Last {len(fights)} fight(s) for {fighter_name}:")
    print("  " + "-" * 80)
    
    for fight, event in fights:
        # Determine opponent
        if fight.fighter_1_id == fighter_id:
            opponent = fight.fighter_2
            was_fighter_1 = True
        else:
            opponent = fight.fighter_1
            was_fighter_1 = False
        
        opponent_name = opponent.name if opponent else "Unknown"
        
        # Determine result
        if fight.winner_id is None:
            result = "Draw/NC"
            result_symbol = "="
        elif fight.winner_id == fighter_id:
            result = "WIN"
            result_symbol = "✓"
        else:
            result = "LOSS"
            result_symbol = "✗"
        
        # Get method and round if available
        method_str = ""
        if fight.method:
            method_str = f" ({fight.method}"
            if fight.round_finished:
                method_str += f", R{fight.round_finished}"
            method_str += ")"
        
        # Format event date
        event_date_str = event.date if event.date else "Unknown"
        
        # Format event name if available (truncate if too long)
        event_name_str = ""
        if event.name:
            event_name_short = event.name[:30] + "..." if len(event.name) > 30 else event.name
            event_name_str = f" @ {event_name_short}"
        
        print(f"  {result_symbol} {result:4s} vs {opponent_name:30s} | {event_date_str:12s}{method_str}{event_name_str}")
    
    print("  " + "-" * 80)


def xgboost_predict(
    fighter_1_name: str,
    fighter_2_name: str,
    title_fight: bool = False,
    quiet: bool = False,
    model_name: str = "xgboost_model",
    fighter_1_id: int | None = None,
    fighter_2_id: int | None = None,
    fighter_1_ufcstats_id: str | None = None,
    fighter_2_ufcstats_id: str | None = None,
    allow_ambiguous: bool = False,
    symmetric: bool = True,
    debug_fights: int | None = None,
):
    """Make a prediction using XGBoost model"""
    
    # Load model
    xgb_model = XGBoostModel()
    xgb_model.load_model(model_name)
    
    if not quiet:
        logger.info(f"Using model: {model_name}")
    
    # Load feature pipeline
    pipeline = FeaturePipeline(initialize_db=False)
    pipeline.load_pipeline(model_name=model_name)
    
    # Find fighters
    db = DatabaseManager()
    session = db.get_session()
    
    fighter_1, f1_candidates = _resolve_fighter(
        session,
        fighter_1_name,
        db_id=fighter_1_id,
        ufcstats_id=fighter_1_ufcstats_id,
        allow_ambiguous=allow_ambiguous,
    )
    fighter_2, f2_candidates = _resolve_fighter(
        session,
        fighter_2_name,
        db_id=fighter_2_id,
        ufcstats_id=fighter_2_ufcstats_id,
        allow_ambiguous=allow_ambiguous,
    )
    
    if not fighter_1:
        if f1_candidates:
            _print_fighter_candidates(f1_candidates, session=session)
            logger.error(
                f"Ambiguous fighter match for '{fighter_1_name}'. "
                f"Re-run with --fighter-1-id or --fighter-1-ufcstats-id (or pass --allow-ambiguous)."
            )
        else:
            logger.error(f"Fighter not found: {fighter_1_name}")
        session.close()
        return
    if not fighter_2:
        if f2_candidates:
            _print_fighter_candidates(f2_candidates, session=session)
            logger.error(
                f"Ambiguous fighter match for '{fighter_2_name}'. "
                f"Re-run with --fighter-2-id or --fighter-2-ufcstats-id (or pass --allow-ambiguous)."
            )
        else:
            logger.error(f"Fighter not found: {fighter_2_name}")
        session.close()
        return
    
    if not quiet:
        logger.info(f"Matched: {fighter_1.name} vs {fighter_2.name}")
    
    # Show debug information if requested
    if debug_fights is not None and debug_fights > 0:
        print("\n" + "=" * 80)
        print(f"[DEBUG] Recent Fight History (last {debug_fights} fights)")
        print("=" * 80)
        _show_fighter_recent_fights(session, fighter_1.id, fighter_1.name, debug_fights)
        _show_fighter_recent_fights(session, fighter_2.id, fighter_2.name, debug_fights)
        print("=" * 80 + "\n")
    
    # Extract features and make predictions
    extractor = MatchupFeatureExtractor(session)
    
    if symmetric:
        # Compute predictions for both fighter orders and average them
        # This makes the prediction order-invariant
        features_1 = extractor.extract_matchup_features(fighter_1.id, fighter_2.id)
        features_1['is_title_fight'] = 1 if title_fight else 0
        
        features_2 = extractor.extract_matchup_features(fighter_2.id, fighter_1.id)
        features_2['is_title_fight'] = 1 if title_fight else 0
        
        # Prepare both feature sets
        X_df_1 = pd.DataFrame([features_1])
        X_scaled_1, _ = pipeline.prepare_features(X_df_1, fit_scaler=False)
        
        X_df_2 = pd.DataFrame([features_2])
        X_scaled_2, _ = pipeline.prepare_features(X_df_2, fit_scaler=False)
        
        # Predict for both orders
        proba_1 = xgb_model.predict(X_scaled_1, use_calibrated=False)
        proba_2 = xgb_model.predict(X_scaled_2, use_calibrated=False)
        
        # p_1 = P(fighter_1 wins | fighter_1, fighter_2)
        # p_2 = P(fighter_2 wins | fighter_2, fighter_1)
        # Symmetric probability: 0.5 * (p_1 + (1 - p_2))
        p_f1_raw = float(proba_1[0])
        p_f2_raw = float(proba_2[0])
        p_f1 = 0.5 * (p_f1_raw + (1.0 - p_f2_raw))
        p_f2 = 1.0 - p_f1
        
        # Use features_1 for display (fighter_1 as f1)
        features = features_1
        
        if not quiet:
            logger.info(f"Using symmetric mode: p({fighter_1.name}|{fighter_1.name},{fighter_2.name})={p_f1_raw:.3f}, "
                       f"p({fighter_2.name}|{fighter_2.name},{fighter_1.name})={p_f2_raw:.3f}, "
                       f"symmetric={p_f1:.3f}")
    else:
        # Non-symmetric: use raw prediction from single order
        features = extractor.extract_matchup_features(fighter_1.id, fighter_2.id)
        features['is_title_fight'] = 1 if title_fight else 0
        
        # Prepare features
        X_df = pd.DataFrame([features])
        X_scaled, _ = pipeline.prepare_features(X_df, fit_scaler=False)
        
        # Predict
        proba = xgb_model.predict(X_scaled, use_calibrated=False)
        p_f1 = float(proba[0])
        p_f2 = 1.0 - p_f1
    
    debug_keys = []

    if not quiet:
        # DEBUG: inspect some key features for this matchup
        debug_keys = [
        "f1_wins", "f1_losses", "f2_wins", "f2_losses",
        "f1_win_rate_last_3", "f2_win_rate_last_3",
        "f1_win_rate_last_5", "f2_win_rate_last_5",
        "f1_age", "f2_age",
        "round_3_win_rate_diff",
        "takedown_matchup",
        "striking_differential",
        "recent_form_diff",
        "f1_recent_finish_losses_last_2", "f2_recent_finish_losses_last_2",
        "f1_recent_finish_loss_last_fight", "f2_recent_finish_loss_last_fight",
        "f1_recent_finish_loss_ratio_last_2", "f2_recent_finish_loss_ratio_last_2",
        "f1_recent_sig_strike_diff_last_3", "f2_recent_sig_strike_diff_last_3",
        "f1_recent_knockdown_diff_last_3", "f2_recent_knockdown_diff_last_3",
        # Opponent quality features
        "f1_avg_opponent_win_rate", "f2_avg_opponent_win_rate",
        "f1_avg_beaten_opponent_win_rate", "f2_avg_beaten_opponent_win_rate",
        "f1_avg_lost_to_opponent_win_rate", "f2_avg_lost_to_opponent_win_rate",
        "f1_opponent_quality_score", "f2_opponent_quality_score",
        "f1_avg_opponent_total_fights", "f2_avg_opponent_total_fights",
        "avg_opponent_win_rate_diff", "opponent_quality_score_diff",
        # Time decayed features
        "f1_time_decayed_win_rate", "f2_time_decayed_win_rate",
        "time_decayed_win_rate_diff",
        # Striking features
        "f1_sig_strikes_landed_per_min_lifetime", "f2_sig_strikes_landed_per_min_lifetime",
        "f1_striking_accuracy_lifetime", "f2_striking_accuracy_lifetime",
        "striking_output_diff", "striking_accuracy_diff",
    ]
        print("\n[DEBUG] Key features for this matchup:")
        for k in debug_keys:
            if k in features:
                print(f"  {k}: {features[k]}")
        print("")
        
        # Debug: Show all opponent quality related features
        print("[DEBUG] All opponent/loss related features:")
        opponent_features = [k for k in features.keys() if 'opponent' in k.lower() or 'lost' in k.lower()]
        for k in sorted(opponent_features):
            print(f"  {k}: {features[k]}")
        print("")

    
    prediction = 1 if p_f1 > 0.5 else 0
    
    # Always print the main prediction (even in quiet mode)
    fight_type = "TITLE FIGHT (5 rounds)" if title_fight else "Non-title (3 rounds)"
    print("\n" + "="*60)
    print(f"PREDICTION: {fighter_1.name} vs {fighter_2.name}")
    print(f"Fight Type: {fight_type}")
    print("="*60)
    print(f"{fighter_1.name}: {p_f1*100:.1f}% chance to win")
    print(f"{fighter_2.name}: {p_f2*100:.1f}% chance to win")
    print("")
    if prediction == 1:
        print(f"⭐ Predicted Winner: {fighter_1.name}")
    else:
        print(f"⭐ Predicted Winner: {fighter_2.name}")
    print("="*60)
    
    if quiet:
        # In quiet mode, only print the percentages and exit
        print("")  # Just a newline
        return
    
    print("")
    
    # Show key factors
    importance_df = xgb_model.get_feature_importance(top_n=5)
    print("Top 5 Most Important Features in Model:")
    for idx, row in importance_df.iterrows():
        print(f"  {idx+1}. {row['feature']}")
    print("")
    
    # Show actual values for top features
    print("[ANALYSIS] Top Feature Values:")
    top_features = importance_df['feature'].tolist()
    for feat in top_features:
        if feat in features:
            print(f"  {feat}: {features[feat]}")
    print("")
    
    # Feature contribution analysis
    print("[FEATURE CONTRIBUTION ANALYSIS]")
    print(f"Features favoring {fighter_1.name} (f1):")
    f1_favors = []
    f2_favors = []
    
    # Check differential features (positive = f1 advantage)
    diff_features = [f for f in top_features if '_diff' in f]
    for feat in diff_features:
        if feat in features:
            val = features[feat]
            if val > 0:
                f1_favors.append((feat, val, "f1 advantage"))
            elif val < 0:
                f2_favors.append((feat, val, "f2 advantage"))
    
    # Check individual features
    for feat in top_features:
        if feat.startswith('f1_') and feat in features:
            f1_favors.append((feat, features[feat], "f1 value"))
        elif feat.startswith('f2_') and feat in features:
            f2_favors.append((feat, features[feat], "f2 value"))
    
    # Sort by absolute value
    f1_favors.sort(key=lambda x: abs(x[1]), reverse=True)
    f2_favors.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"  Top 3 favoring {fighter_1.name}:")
    for feat, val, desc in f1_favors[:3]:
        print(f"    {feat}: {val:.4f} ({desc})")
    
    print(f"\n  Top 3 favoring {fighter_2.name}:")
    for feat, val, desc in f2_favors[:3]:
        # Special explanation for age_difference
        if feat == 'age_difference':
            f1_age_val = features.get('f1_age', 'N/A')
            f2_age_val = features.get('f2_age', 'N/A')
            if val < 0:
                explanation = f"f2 advantage (f2 is {abs(val):.1f} years older: {f2_age_val} vs {f1_age_val})"
            elif val > 0:
                explanation = f"f1 advantage (f1 is {val:.1f} years older: {f1_age_val} vs {f2_age_val})"
            else:
                explanation = "f2 advantage (same age)"
            print(f"    {feat}: {val:.4f} ({explanation})")
        else:
            print(f"    {feat}: {val:.4f} ({desc})")
    print("")
    
    # Age difference explanation
    if 'age_difference' in top_features and 'age_difference' in features:
        age_diff = features['age_difference']
        f1_age = features.get('f1_age', None)
        f2_age = features.get('f2_age', None)
        if f1_age is not None and f2_age is not None:
            print("[AGE_DIFFERENCE EXPLANATION]")
            print(f"  Calculation: age_difference = f1_age - f2_age = {f1_age:.1f} - {f2_age:.1f} = {age_diff:.1f}")
            if age_diff < 0:
                print(f"  Interpretation: {fighter_1.name} (f1) is {abs(age_diff):.1f} years YOUNGER than {fighter_2.name} (f2)")
                print(f"  Model learned: Negative age_difference often favors the OLDER fighter (f2) because:")
                print(f"    - Older fighters typically have more experience")
                print(f"    - Experience can offset age-related decline (especially in 20s-30s)")
                print(f"    - The model considers other factors: opponent quality, recent form, etc.")
            elif age_diff > 0:
                print(f"  Interpretation: {fighter_1.name} (f1) is {age_diff:.1f} years OLDER than {fighter_2.name} (f2)")
                print(f"  Model learned: Positive age_difference may favor f1 if age brings experience")
                print(f"    - But age interaction features penalize older fighters who are declining/inactive")
            else:
                print(f"  Interpretation: Both fighters are the same age")
            print(f"  Note: The model has NO monotone constraint on age_difference, so it learned")
            print(f"        a nuanced relationship from training data (not simply 'younger = better')")
            print(f"  Note: Some features (striking/grappling stats) were temporarily excluded due to")
            print(f"        point-in-time leakage concerns, but will be restored once fixed.")
            print("")
    
    # Contradiction analysis
    if p_f1 > 0.75:
        print("[CONTRADICTION DETECTED]")
        print(f"  Model heavily favors {fighter_1.name} despite:")
        if 'avg_opponent_win_rate_diff' in features and features['avg_opponent_win_rate_diff'] < 0:
            print(f"    - {fighter_2.name} faced better opponents (avg_opponent_win_rate_diff: {features['avg_opponent_win_rate_diff']:.4f})")
        if 'avg_beaten_opponent_win_rate_diff' in features and features['avg_beaten_opponent_win_rate_diff'] < 0:
            print(f"    - {fighter_2.name} beat better opponents (avg_beaten_opponent_win_rate_diff: {features['avg_beaten_opponent_win_rate_diff']:.4f})")
        # Recency: prefer the direct days-since feature when available (more interpretable than age×days)
        if "f1_days_since_last_fight" in features and "f2_days_since_last_fight" in features:
            f1_days = float(features["f1_days_since_last_fight"])
            f2_days = float(features["f2_days_since_last_fight"])
            if f1_days != f2_days:
                more_recent = fighter_1.name if f1_days < f2_days else fighter_2.name
                print(
                    f"    - {more_recent} fought more recently "
                    f"(f1_days_since_last_fight={f1_days:.0f}, f2_days_since_last_fight={f2_days:.0f})"
                )
        elif 'age_x_days_since_last_fight_diff' in features and features['age_x_days_since_last_fight_diff'] != 0:
            # Fallback: age×days is harder to interpret because age differs, but sign still matters.
            more_recent = fighter_1.name if features['age_x_days_since_last_fight_diff'] < 0 else fighter_2.name
            print(
                f"    - {more_recent} fought more recently "
                f"(age_x_days_since_last_fight_diff: {features['age_x_days_since_last_fight_diff']:.0f})"
            )
        
        # Show individual opponent quality scores
        print("\n  Opponent Quality Breakdown:")
        if 'f1_opponent_quality_score' in features:
            print(f"    {fighter_1.name} opponent_quality_score: {features['f1_opponent_quality_score']:.4f}")
        if 'f2_opponent_quality_score' in features:
            print(f"    {fighter_2.name} opponent_quality_score: {features['f2_opponent_quality_score']:.4f}")
        if 'f1_avg_lost_to_opponent_win_rate' in features:
            print(f"    {fighter_1.name} avg_lost_to_opponent_win_rate: {features['f1_avg_lost_to_opponent_win_rate']:.4f}")
        if 'f2_avg_lost_to_opponent_win_rate' in features:
            f2_lost_wr = features['f2_avg_lost_to_opponent_win_rate']
            print(f"    {fighter_2.name} avg_lost_to_opponent_win_rate: {f2_lost_wr:.4f}")
            if f2_lost_wr > 0.75:
                print(f"      ⚠️  CRITICAL: {fighter_2.name} lost to ELITE opponents ({f2_lost_wr*100:.1f}% win rate)")
                print(f"      This means their losses were to top-tier fighters, not weak competition")
                print(f"      But opponent_quality_score penalizes them for this!")
        
        # Opponent quality score calculation explanation (matches features/opponent_quality.py)
        if 'f2_avg_beaten_opponent_win_rate' in features and 'f2_avg_lost_to_opponent_win_rate' in features:
            beaten_wr = float(features['f2_avg_beaten_opponent_win_rate'])
            lost_wr = float(features['f2_avg_lost_to_opponent_win_rate'])
            naive = beaten_wr - lost_wr
            adjusted = beaten_wr - (1.0 - lost_wr)
            print(f"\n  Opponent Quality Score Calculation (info):")
            print(f"    {fighter_2.name} beat opponents with {beaten_wr*100:.1f}% win rate")
            print(f"    {fighter_2.name} lost to opponents with {lost_wr*100:.1f}% win rate")
            print(f"    naive_raw = beaten - lost = {beaten_wr:.4f} - {lost_wr:.4f} = {naive:.4f}")
            print(f"    adjusted_raw = beaten - (1 - lost) = {beaten_wr:.4f} - (1 - {lost_wr:.4f}) = {adjusted:.4f}")
            print(f"    (Adjusted formula reduces penalty for losses to elite opponents.)")
        print("")
        print("  This suggests:")
        print("    - Recent form (time_decayed_win_rate) is dominating the prediction")
        print("    - Model may not be properly weighting opponent quality")
        print(f"    - Consider: {fighter_2.name}'s losses were to elite fighters")
        print(f"    - {fighter_1.name}'s wins may be against lower-tier competition")
        print(f"    - The 'avg_lost_to_opponent_win_rate' should be HIGH for {fighter_2.name} if they lost to elite fighters")
        print("")
    
    # Warning if prediction seems extreme
    if p_f1 > 0.85 or p_f1 < 0.15:
        print("[WARNING] Extreme prediction detected. Possible reasons:")
        print("  - Heavy weighting of recent form (last 3-5 fights)")
        print("  - Opponent quality may not fully account for elite-level losses")
        print("  - Model may be overfitting to recent win rates")
        print("  - Consider checking opponent quality features above")
        print("")
    
    session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost UFC Fight Prediction')
    parser.add_argument('--fighter-1', type=str, required=True, help='First fighter name')
    parser.add_argument('--fighter-2', type=str, required=True, help='Second fighter name')
    parser.add_argument('--fighter-1-id', type=int, default=None, help='Disambiguate fighter 1 by DB id (fighters.id)')
    parser.add_argument('--fighter-2-id', type=int, default=None, help='Disambiguate fighter 2 by DB id (fighters.id)')
    parser.add_argument('--fighter-1-ufcstats-id', type=str, default=None, help='Disambiguate fighter 1 by UFCStats id (fighters.fighter_id)')
    parser.add_argument('--fighter-2-ufcstats-id', type=str, default=None, help='Disambiguate fighter 2 by UFCStats id (fighters.fighter_id)')
    parser.add_argument('--title-fight', action='store_true', help='Is this a title fight?')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode: only show prediction percentages')
    parser.add_argument('--model-name', '--model', dest='model_name', type=str, default='xgboost_model',
                        help='Model name to use (default: xgboost_model, e.g., xgboost_model_with_2025)')
    parser.add_argument('--allow-ambiguous', action='store_true',
                        help='Allow ambiguous name matches by picking a best guess (prints candidates).')
    parser.add_argument('--symmetric', action='store_true', default=True,
                        help='Use symmetric probabilities by averaging both fighter orders (DEFAULT: True). '
                             'Makes prediction order-invariant (flipping fighters gives same result).')
    parser.add_argument('--no-symmetric', dest='symmetric', action='store_false',
                        help='Disable symmetric mode (use raw prediction from single fighter order).')
    parser.add_argument('--debug', type=int, default=None, metavar='N',
                        help='Show last N fights in database for each fighter (e.g., --debug 3 shows last 3 fights)')
    
    args = parser.parse_args()
    
    xgboost_predict(
        args.fighter_1,
        args.fighter_2,
        args.title_fight,
        quiet=args.quiet,
        model_name=args.model_name,
        fighter_1_id=args.fighter_1_id,
        fighter_2_id=args.fighter_2_id,
        fighter_1_ufcstats_id=args.fighter_1_ufcstats_id,
        fighter_2_ufcstats_id=args.fighter_2_ufcstats_id,
        allow_ambiguous=args.allow_ambiguous,
        symmetric=args.symmetric,
        debug_fights=args.debug,
    )

