import joblib
import pandas as pd
import numpy as np
from src.config import BEST_MODEL_FILE, FEATURE_IMPORTANCE_FILE, FEATURE_COLS

def explain_model_features():
    model_data = joblib.load(BEST_MODEL_FILE)
    model = model_data["model"]
    model_name = model_data["model_name"]
    features = model_data["features"]
    
    importances = None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        # Fallback uniform importance if model doesn't expose coefficients
        importances = np.ones(len(features)) / len(features)
        
    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": np.round(importances, 4)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    # Calculate relative percentage
    tot = df_imp["Importance"].sum()
    df_imp["Relative_Weight_%"] = np.round((df_imp["Importance"] / (tot if tot > 0 else 1.0)) * 100, 2)
    
    df_imp.to_csv(FEATURE_IMPORTANCE_FILE, index=False, encoding="utf-8")
    
    print(f"\n--- Feature Importance / Model Explanation ({model_name}) ---")
    for idx, row in df_imp.iterrows():
        print(f"{idx+1}. {row['Feature']}: {row['Importance']} ({row['Relative_Weight_%']}%)")
        
    print(f"Saved feature importance matrix to {FEATURE_IMPORTANCE_FILE}\n")
    return df_imp

if __name__ == "__main__":
    explain_model_features()
