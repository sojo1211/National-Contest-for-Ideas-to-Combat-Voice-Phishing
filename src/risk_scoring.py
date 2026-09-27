import joblib
import pandas as pd
import numpy as np
from src.config import (
    PROCESSED_FEATURES_FILE,
    LABELS_FILE,
    BEST_MODEL_FILE,
    RISK_SCORES_FILE,
    FEATURE_COLS
)

DISCLAIMER = (
    "※ 본 Risk Score는 보이스피싱 또는 자금세탁 여부를 확정하지 않음.\n"
    "※ 온체인 거래행동 기반 AML 위험 신호 및 추가 분석 우선순위를 나타내는 내부 지표임."
)

def compute_risk_scores():
    model_data = joblib.load(BEST_MODEL_FILE)
    model = model_data["model"]
    model_name = model_data["model_name"]
    
    df_features = pd.read_csv(PROCESSED_FEATURES_FILE)
    df_labels = pd.read_csv(LABELS_FILE)
    
    df_merged = pd.merge(df_features, df_labels, on="wallet", how="inner")
    
    # Filter target wallets (unlabeled / Label='?') or compute all
    df_target = df_merged[df_merged["label"].isin(["?", "?"])].copy()
    if len(df_target) == 0:
        # If no explicit '?' fallback to all wallets for demonstration
        df_target = df_merged.copy()
        
    X_target = df_target[FEATURE_COLS]
    probabilities = model.predict_proba(X_target)[:, 1]
    
    df_target["positive_probability"] = np.round(probabilities, 4)
    df_target["risk_score"] = np.round(probabilities * 100, 1)
    
    # Define investigation priority threshold (e.g. Risk Score >= 70)
    df_target["priority_status"] = df_target["risk_score"].apply(
        lambda s: "[추가 분석 우선 대상]" if s >= 70 else "[일반 관찰 대상]"
    )
    
    output_cols = [
        "wallet",
        "risk_score",
        "positive_probability",
        "priority_status",
        "fan_in",
        "fan_out",
        "transaction_count",
        "holding_time_median",
        "pass_through_ratio",
        "small_tx_ratio",
        "amount_cv"
    ]
    
    df_results = df_target[output_cols].sort_values(by="risk_score", ascending=False).reset_index(drop=True)
    df_results.to_csv(RISK_SCORES_FILE, index=False, encoding="utf-8")
    
    print(f"\n--- Risk Scoring Complete ---")
    print(f"Model used: {model_name}")
    print(f"Total target wallets scored: {len(df_results)}")
    print(f"High risk wallets (>=70): {len(df_results[df_results['risk_score'] >= 70])}")
    print(f"Saved risk scoring results to {RISK_SCORES_FILE}\n")
    print(DISCLAIMER)
    
    return df_results

if __name__ == "__main__":
    compute_risk_scores()
