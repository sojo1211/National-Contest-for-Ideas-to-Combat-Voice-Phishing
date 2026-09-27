import pandas as pd
import numpy as np
from src.config import PROCESSED_FEATURES_FILE

def extract_wallet_features(df_tx, target_wallets=None):
    if target_wallets is None:
        target_wallets = set(df_tx["from"].tolist() + df_tx["to"].tolist())

    wallet_records = []
    
    for wallet in target_wallets:
        in_txs = df_tx[df_tx["to"] == wallet].sort_values("timestamp")
        out_txs = df_tx[df_tx["from"] == wallet].sort_values("timestamp")
        
        all_txs = pd.concat([in_txs, out_txs])
        if len(all_txs) == 0:
            continue
            
        fan_in = in_txs["from"].nunique()
        fan_out = out_txs["to"].nunique()
        tx_count = len(all_txs)
        
        amounts = all_txs["amount"].values
        amt_mean = float(np.mean(amounts))
        amt_median = float(np.median(amounts))
        amt_std = float(np.std(amounts)) if len(amounts) > 1 else 0.0
        amt_cv = (amt_std / amt_mean) if amt_mean > 0 else 0.0
        
        # Small tx ratio (e.g., transactions <= 3000 USDT)
        small_tx_count = np.sum(amounts <= 3000)
        small_tx_ratio = float(small_tx_count / len(amounts))
        
        # Pass-through Ratio
        total_in_amt = in_txs["amount"].sum()
        total_out_amt = out_txs["amount"].sum()
        pass_through_ratio = (total_out_amt / total_in_amt) if total_in_amt > 0 else 0.0
        pass_through_ratio = float(min(1.0, max(0.0, pass_through_ratio)))
        
        # Holding Time (minutes)
        holding_times = []
        if len(in_txs) > 0 and len(out_txs) > 0:
            out_times = out_txs["timestamp"].values
            for in_time in in_txs["timestamp"].values:
                # Find the first outgoing transaction after this incoming transaction
                subsequent_outs = out_times[out_times > in_time]
                if len(subsequent_outs) > 0:
                    diff_mins = (subsequent_outs[0] - in_time) / np.timedelta64(1, 'm')
                    holding_times.append(diff_mins)
        
        if len(holding_times) > 0:
            ht_mean = float(np.mean(holding_times))
            ht_median = float(np.median(holding_times))
        else:
            # Default fallback holding time if no matched pair
            ht_mean = 1440.0 # 24 hours in minutes
            ht_median = 1440.0
            
        wallet_records.append({
            "wallet": wallet,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "transaction_count": tx_count,
            "amount_mean": round(amt_mean, 2),
            "amount_median": round(amt_median, 2),
            "amount_std": round(amt_std, 2),
            "amount_cv": round(amt_cv, 4),
            "small_tx_ratio": round(small_tx_ratio, 4),
            "holding_time_mean": round(ht_mean, 2),
            "holding_time_median": round(ht_median, 2),
            "pass_through_ratio": round(pass_through_ratio, 4)
        })

    df_features = pd.DataFrame(wallet_records)
    df_features.to_csv(PROCESSED_FEATURES_FILE, index=False, encoding="utf-8")
    print(f"[Feature Engineering] Extracted features for {len(df_features)} wallets into {PROCESSED_FEATURES_FILE}")
    return df_features

if __name__ == "__main__":
    from src.preprocess import preprocess_raw_transactions
    df_tx = preprocess_raw_transactions()
    extract_wallet_features(df_tx)
