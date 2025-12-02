import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# === 경로 설정 ===
DATA_DIR = "./data"
OUTPUT_DIR = "./inference"
INPUT_FILE = "final_preprocessed_df.parquet"
OUTPUT_FILE = "ranking_df.parquet"

# [핵심] 랜덤 시드 고정 상수
RANDOM_SEED = 42

def main():
    # 1. 시드 고정 (이 코드가 있으면 언제 실행해도 결과가 같습니다)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"[Error] 입력 파일이 없습니다: {input_path}")
        return

    print(f"Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)

    # 시간순 정렬 및 Test Set 분리
    df = df.sort_values("event_time").reset_index(drop=True)
    n = len(df)
    test_df = df.iloc[int(n * 0.9):].reset_index(drop=True)

    # Product Meta 정보 생성
    print("Building product meta lookup...")
    product_meta = (
        df.groupby("product_id_idx")[[
            "category_id_idx", "category_code_idx", "brand_idx", "price_scaled"
        ]].agg("first")
    )
    product_meta_dict = product_meta.to_dict("index")
    all_products_np = np.array(list(product_meta_dict.keys()))

    # Seen Item 미리 집계
    print("Indexing seen items per session...")
    seen_map = test_df.groupby("user_session_idx")["product_id_idx"].apply(set).to_dict()

    # Positive Rows 추출
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
            # 시드가 고정되어 있으므로, 이 choice 결과도 매번 동일한 순서로 나옵니다.
            cands = np.random.choice(all_products_np, size=60)
            for c in cands:
                if c not in seen and c not in neg_samples:
                    neg_samples.append(c)
                    if len(neg_samples) == 50: break

        # Positive
        pos_row["ranking_label"] = 1
        ranking_rows.append(pos_row)

        # Negative
        for neg_pid in neg_samples:
            neg_row = pos_row.copy()
            meta = product_meta_dict[neg_pid]
            neg_row.update(meta)
            neg_row["product_id_idx"] = neg_pid
            neg_row["ranking_label"] = 0
            ranking_rows.append(neg_row)

    # 저장
    print("Saving to parquet...")
    result_df = pd.DataFrame(ranking_rows)
    result_df.to_parquet(output_path, index=False)
    print(f"\n[Success] 데이터셋 생성 완료! (Seed 고정됨)")

if __name__ == "__main__":
    main()
