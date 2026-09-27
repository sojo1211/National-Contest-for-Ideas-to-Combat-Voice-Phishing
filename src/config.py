import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DATA_DIR = DATA_DIR / "labels"
BACKGROUND_DATA_DIR = DATA_DIR / "background"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DATA_DIR, BACKGROUND_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_TX_FILE = RAW_DATA_DIR / "tron_usdt_transactions.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "wallet_features.csv"
LABELS_FILE = LABELS_DATA_DIR / "proxy_labels.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
RISK_SCORES_FILE = RESULTS_DIR / "risk_scores.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "feature_importance.csv"

# Feature list
FEATURE_COLS = [
    "fan_in",
    "fan_out",
    "transaction_count",
    "amount_mean",
    "amount_median",
    "amount_std",
    "amount_cv",
    "small_tx_ratio",
    "holding_time_mean",
    "holding_time_median",
    "pass_through_ratio"
]
