"""
Compare general model vs underdog model on holdout underdog fights.
File: models/validate_underdog.py

Run: python models/validate_underdog.py
"""

import sys
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_CSV  = "reports_mar_4_v2/eval_data_20260304_165004.csv"
MODEL_DIR = Path("models/saved")
THRESHOLD = 0.40


def main():
    df = pd.read_csv(EVAL_CSV)

    # Deduplicate double-sampled rows: keep only target=1 rows (f1 is actual winner)
    df = df[df["target"] == 1].copy()

    # Filter to fights where f1 (the actual winner) was the underdog
    df_under = df[df["market_prob_f1"] < THRESHOLD].copy()
    print(f"Underdog fights in holdout: {len(df_under)}")

    # ── General model baseline ────────────────────────────────────────────────
    gen_col  = "model_prob_f1_symmetric"
    gen_pred = (df_under[gen_col] > 0.5).astype(int)
    gen_acc  = accuracy_score(df_under["target"], gen_pred)
    gen_det  = gen_pred.sum() / len(df_under)
    print(f"\n=== GENERAL MODEL (mar_4_v2) ===")
    print(f"  Accuracy on underdog fights: {gen_acc*100:.1f}%")
    print(f"  Upset detection rate:        {gen_det*100:.1f}%  ← baseline 34.2%")

    # ── Underdog model ────────────────────────────────────────────────────────
    ud_model_path = MODEL_DIR / "underdog_v1.json"
    if not ud_model_path.exists():
        print("\nUnderdog model not found. Run: python models/train_underdog_model.py")
        return

    ud_model    = xgb.XGBClassifier()
    ud_model.load_model(ud_model_path)
    ud_scaler   = joblib.load(MODEL_DIR / "underdog_v1_feature_scaler.pkl")
    ud_features = joblib.load(MODEL_DIR / "underdog_v1_feature_names.pkl")

    X_ud = df_under.reindex(columns=ud_features, fill_value=0).fillna(0)
    X_ud_scaled = pd.DataFrame(
        ud_scaler.transform(X_ud), columns=ud_features, index=X_ud.index
    )

    ud_prob = ud_model.predict_proba(X_ud_scaled)[:, 1]
    ud_pred = (ud_prob > 0.5).astype(int)
    ud_acc  = accuracy_score(df_under["target"], ud_pred)
    ud_det  = ud_pred.sum() / len(df_under)
    print(f"\n=== UNDERDOG MODEL (underdog_v1) ===")
    print(f"  Accuracy on underdog fights: {ud_acc*100:.1f}%")
    print(f"  Upset detection rate:        {ud_det*100:.1f}%  ← target >= 40%")

    # ── Blended ───────────────────────────────────────────────────────────────
    BLEND         = 0.65
    blended_prob  = BLEND * ud_prob + (1 - BLEND) * df_under[gen_col].values
    blended_pred  = (blended_prob > 0.5).astype(int)
    blended_acc   = accuracy_score(df_under["target"], blended_pred)
    blended_det   = blended_pred.sum() / len(df_under)
    print(f"\n=== BLENDED (65% underdog + 35% general) ===")
    print(f"  Accuracy:            {blended_acc*100:.1f}%")
    print(f"  Upset detection:     {blended_det*100:.1f}%")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    if ud_det >= 0.40:
        print("  ✓ GATE PASSED — deploy underdog_v1")
    else:
        print(f"  ✗ GATE FAILED — upset_detection={ud_det:.3f} < 0.40")
        print("  Increase --upsample or decrease --max-depth and retrain.")


if __name__ == "__main__":
    main()
