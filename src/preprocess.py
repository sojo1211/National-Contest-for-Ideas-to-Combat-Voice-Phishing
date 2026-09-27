import pandas as pd
from src.config import RAW_TX_FILE

def preprocess_raw_transactions(raw_filepath=RAW_TX_FILE):
    df = pd.read_csv(raw_filepath)
    
    # 1. Ensure string clean up on addresses
    df["from"] = df["from"].astype(str).str.strip()
    df["to"] = df["to"].astype(str).str.strip()
    
    # 2. Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 3. Handle amounts
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df = df[df["amount"] > 0]
    
    # 4. Remove exact duplicates
    df = df.drop_duplicates(subset=["transaction_id"])
    
    # 5. Sort by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    print(f"[Preprocess] Preprocessed {len(df)} transactions.")
    return df

if __name__ == "__main__":
    preprocess_raw_transactions()
