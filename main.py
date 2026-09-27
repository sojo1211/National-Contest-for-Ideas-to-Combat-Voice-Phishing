import sys
import pandas as pd
from src.config import (
    RAW_TX_FILE,
    PROCESSED_FEATURES_FILE,
    LABELS_FILE,
    MODEL_COMPARISON_FILE,
    RISK_SCORES_FILE,
    FEATURE_IMPORTANCE_FILE
)
from src.trongrid_collector import collect_real_dataset
from src.preprocess import preprocess_raw_transactions
from src.feature_engineering import extract_wallet_features
from src.label_builder import build_proxy_labels
from src.train import train_and_compare_models
from src.risk_scoring import compute_risk_scores
from src.explain import explain_model_features

def main():
    print("=========================================================================")
    print("    Safe-Trade AI: 100% REAL TRON USDT On-chain AML Risk Scoring System  ")
    print("=========================================================================\n")
    
    # Step 1: Collect 100% Real TRON Mainnet USDT Transfers & OFAC SDN List
    print(">>> Phase 1: Real On-chain Data Collection (TronScan API & Treasury OFAC)")
    df_raw, pos_wallets, neg_wallets, target_wallets = collect_real_dataset()
    
    # Step 2: Data Preprocessing
    print("\n>>> Phase 2: Data Preprocessing")
    df_clean = preprocess_raw_transactions(RAW_TX_FILE)
    
    # Step 3: Extract Wallet Behavioral Features
    print("\n>>> Phase 3: Extracting Wallet Behavioral Features")
    all_eval_wallets = list(set(pos_wallets + neg_wallets + target_wallets))
    df_features = extract_wallet_features(df_clean, target_wallets=all_eval_wallets)
    
    # Step 4: Build Proxy Labels
    print("\n>>> Phase 4: Building Proxy Labels (OFAC SDN = 1, Exchange = 0, Target = ?)")
    build_proxy_labels(pos_wallets, neg_wallets, target_wallets)
    
    # Step 5: Train and Compare ML Models
    print("\n>>> Phase 5: Machine Learning Training & Model Comparison")
    df_comparison, best_model = train_and_compare_models()
    
    # Step 6: Risk Scoring on Target Wallets
    print("\n>>> Phase 6: Risk Scoring on Target Wallets")
    df_risk = compute_risk_scores()
    
    # Step 7: Feature Importance & Explainability
    print("\n>>> Phase 7: Model Explainability & Feature Importance")
    df_explain = explain_model_features()
    
    print("=========================================================================")
    print("   Safe-Trade AI Pipeline Completed Successfully with 100% REAL Data!   ")
    print("=========================================================================")

if __name__ == "__main__":
    main()
