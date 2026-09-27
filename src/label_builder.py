import pandas as pd
from src.config import LABELS_FILE

def build_proxy_labels(pos_wallets, neg_wallets, target_wallets):
    records = []
    
    for w in pos_wallets:
        records.append({
            "wallet": w,
            "label": 1,
            "description": "OFAC SDN TRON Proxy (Positive)"
        })
        
    for w in neg_wallets:
        records.append({
            "wallet": w,
            "label": 0,
            "description": "Verified Exchange / Issuer Proxy (Negative)"
        })
        
    for w in target_wallets:
        records.append({
            "wallet": w,
            "label": "?",
            "description": "Unlabeled Target Wallet (Detection Subject)"
        })
        
    df_labels = pd.DataFrame(records)
    df_labels.to_csv(LABELS_FILE, index=False, encoding="utf-8")
    print(f"[Label Builder] Saved {len(df_labels)} wallet labels into {LABELS_FILE}")
    return df_labels

if __name__ == "__main__":
    pass
