import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# === Path Configuration ===
DATA_DIR = "./data"
OUTPUT_DIR = "./inference"
INPUT_FILE = "final_preprocessed_df.parquet"
OUTPUT_FILE = "ranking_df.parquet"

# [Key] Random Seed Constant for Reproducibility
RANDOM_SEED = 42

def main():
    # 1. Set random seed (Ensures consistent results for every run)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"[Error] Input file not found: {input_path}")
        return

    print(f"Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Sort by time and split Test Set
    df = df.sort_values("event_time").reset_index(drop=True)
    n = len(df)
    test_df = df.iloc[int(n * 0.9):].reset_index(drop=True)

    # Build Product Meta Lookup
    print("Building product meta lookup...")
    product_meta = (
        df.groupby("product_id_idx")[[
            "category_id_idx", "category_code_idx", "brand_idx", "price_scaled"
        ]].agg("first")
    )
    product_meta_dict = product_meta.to_dict("index")
    all_products_np = np.array(list(product_meta_dict.keys()))

    # Pre-index seen items per session
    print("Indexing seen items per session...")
    seen_map = test_df.groupby("user_session_idx")["product_id_idx"].apply(set).to_dict()

    # Extract Positive Rows
    print("Extracting positive samples...")
    pos_df = test_df[test_df["target"] == 1].groupby("user_session_idx").tail(1)
    pos_rows = pos_df.to_dict("records")

    ranking_rows = []
    print(f"Generating Ranking Samples (Fixed Seed={RANDOM_SEED}) for {len(pos_rows)} sessions...")

    for pos_row in tqdm(pos_rows, ncols=80):
        sid = pos_row["user_session_idx"]
        seen = seen_map.get(sid, set())

        neg_samples = []
        while len(neg_samples) < 50:
            # Since the seed is fixed, this choice result will be in the same order every time.
            cands = np.random.choice(all_products_np, size=60)
            for c in cands:
                if c not in seen and c not in neg_samples:
                    neg_samples.append(c)
                    if len(neg_samples) == 50: break

        # Positive Sample
        pos_row["ranking_label"] = 1
        ranking_rows.append(pos_row)

        # Negative Samples
        for neg_pid in neg_samples:
            neg_row = pos_row.copy()
            meta = product_meta_dict[neg_pid]
            neg_row.update(meta)
            neg_row["product_id_idx"] = neg_pid
            neg_row["ranking_label"] = 0
            ranking_rows.append(neg_row)

    # Save Result
    print("Saving to parquet...")
    result_df = pd.DataFrame(ranking_rows)
    result_df.to_parquet(output_path, index=False)
    print(f"\n[Success] Dataset generation complete! (Seed fixed)")

if __name__ == "__main__":
    main()
