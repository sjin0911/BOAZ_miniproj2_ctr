import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import inspect
from tqdm import tqdm

# ==================================================================================
# [MODEL IMPORT]
# ==================================================================================
try:
    from model.AutoInt import AutoInt
    from model.DeepFM import DeepFM
except ImportError:
    print("[Warn] 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

# ==================================================================================
# 설정
# ==================================================================================
class Config:
    DATA_DIR = "./data"
    CKPT_DIR = "./ckpts"
    OUTPUT_DIR = "./inference"

    # 입출력 파일명
    RANKING_DATA_FILE = "ranking_df.parquet"
    RESULT_FILE = "inference_results.csv"

    MAPPING_FILE = "ID_Index_mappings.pkl"
    CONFIG_FILE = "config.json"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 컬럼 정의 (수정 금지)
DENSE_COLS = ["is_weekend", "price_scaled", "session_rank_scaled", "hour_scaled", "day_of_week_scaled"]
SPARSE_BASE_COLS = ["product_id_idx", "category_id_idx", "category_code_idx", "brand_idx", "user_id_idx", "user_session_idx"]
SEQ_COLS = ["seq_prod_1_idx", "seq_prod_2_idx", "seq_prod_3_idx", "seq_prod_4_idx", "seq_prod_5_idx"]

def load_json(path):
    with open(path, 'r') as f: return json.load(f)
def load_pickle(path):
    with open(path, 'rb') as f: return pickle.load(f)

def prepare_batch_inputs(batch_df, version, device):
    """버전(V1/V2/V3)에 따라 데이터를 모델 입력 형태(Tensor)로 변환"""
    dense_vals = batch_df[DENSE_COLS].values.astype(np.float32)
    sparse_vals = batch_df[SPARSE_BASE_COLS].values.astype(np.int64)
    seq_vals = batch_df[SEQ_COLS].fillna(0).values.astype(np.int64)

    # Xi, Xv 생성 로직
    if version in ["v1", "v3"]:
        Xi = np.concatenate((np.zeros_like(dense_vals, dtype=np.int64), sparse_vals), axis=1)
        Xv = np.concatenate((dense_vals, np.ones_like(sparse_vals, dtype=np.float32)), axis=1)
    elif version == "v2":
        Xi = np.concatenate((np.zeros_like(dense_vals, dtype=np.int64), sparse_vals, seq_vals), axis=1)
        Xv = np.concatenate((dense_vals, np.ones_like(sparse_vals, dtype=np.float32), np.ones_like(seq_vals, dtype=np.float32)), axis=1)
    else:
        # 기본값 V3 처리
        return prepare_batch_inputs(batch_df, "v3", device)

    Xi_tensor = torch.tensor(Xi, dtype=torch.long, device=device).unsqueeze(-1)
    Xv_tensor = torch.tensor(Xv, dtype=torch.float, device=device).unsqueeze(-1)

    # V3일 때만 seq 텐서 반환
    seq_tensor = torch.tensor(seq_vals, dtype=torch.long, device=device) if version == "v3" else None

    return Xi_tensor, Xv_tensor, seq_tensor

def main():
    print(f"=== Auto-Detect Inference Started on {Config.DEVICE} ===")

    # 1. Config 로드 및 자동 감지
    config_path = os.path.join(os.getcwd(), Config.CONFIG_FILE)
    if os.path.exists(config_path):
        train_config = load_json(config_path)
        print(f"Loaded config from {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # (1) 모델 종류 감지 (config.json의 "model_type")
    model_type = train_config.get("model_type", "AutoInt").lower()
    # (2) 데이터 버전 감지 (config.json의 "data_loader")
    version = train_config.get("data_loader", "v3").lower()

    print(f"[Auto-Detect] Target Model: {model_type.upper()} | Data Version: {version.upper()}")

    # 2. 매핑 파일 로드 및 Feature Size 계산
    mappings = load_pickle(os.path.join(Config.DATA_DIR, Config.MAPPING_FILE))

    # +1 제거된 정확한 사이즈 (학습 코드와 일치)
    base_sparse_dims = [len(mappings.get(c.replace("_idx", ""), [])) for c in SPARSE_BASE_COLS]
    product_vocab_size = len(mappings.get("product_id", []))
    dense_dims = [1] * len(DENSE_COLS)

    if version == "v1": feature_sizes = dense_dims + base_sparse_dims
    elif version == "v2": feature_sizes = dense_dims + base_sparse_dims + [product_vocab_size]*len(SEQ_COLS)
    else: feature_sizes = dense_dims + base_sparse_dims # V3

    # 3. 모델 초기화 (동적 선택)
    if "autoint" in model_type:
        ModelClass = AutoInt
    elif "deepfm" in model_type:
        ModelClass = DeepFM
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    # [핵심] Config 키 -> 모델 인자 매핑 로직
    def get_valid_kwargs(cls, cfg):
        # JSON 키 : 모델 __init__ 인자 이름 매핑
        key_map = {
            "autoint_dnn_hidden_units": "dnn_hidden_units",
            "deepfm_hidden_dims": "hidden_dims",  # DeepFM 호환성 확보
            "deepfm_dropout": "dropout",
            "embedding_size": "embedding_size"
        }
        valid_keys = set(inspect.signature(cls.__init__).parameters.keys())
        kwargs = {}
        for k, v in cfg.items():
            target_k = key_map.get(k, k)
            if target_k in valid_keys:
                kwargs[target_k] = v
        return kwargs

    model_kwargs = get_valid_kwargs(ModelClass, train_config)

    # [안전장치] DeepFM hidden_dims 누락 시 강제 주입
    if "DeepFM" in ModelClass.__name__:
        if "hidden_dims" not in model_kwargs:
             if "deepfm_hidden_dims" in train_config:
                 model_kwargs["hidden_dims"] = train_config["deepfm_hidden_dims"]
             else:
                 # 최후의 수단: 에러 로그에서 본 값 사용
                 print("[Warn] Config 누락. [150, 300, 150] 강제 사용")
                 model_kwargs["hidden_dims"] = [150, 300, 150]

    # 공통 필수 인자 업데이트
    model_params = inspect.signature(ModelClass.__init__).parameters
    if "feature_sizes" in model_params:
        model_kwargs['feature_sizes'] = feature_sizes

    # DeepFM은 device 인자가 없는 경우가 많음
    if "device" in model_params:
        model_kwargs['device'] = Config.DEVICE
    else:
        model_kwargs['use_cuda'] = torch.cuda.is_available()

    if version == "v3" and "seq_vocab_size" in model_params:
        model_kwargs['seq_vocab_size'] = product_vocab_size

    # 설정 확인 로그
    if "hidden_dims" in model_kwargs:
        print(f"[Config Check] Loaded hidden_dims: {model_kwargs['hidden_dims']}")
    elif "dnn_hidden_units" in model_kwargs:
        print(f"[Config Check] Loaded dnn_hidden_units: {model_kwargs['dnn_hidden_units']}")

    print(f"Initializing {ModelClass.__name__}...")
    model = ModelClass(**model_kwargs)

    # 4. 가중치 로드
    ckpt_path = None
    target_name = f"best_{model_type}.pth"
    possible_path = os.path.join(Config.CKPT_DIR, target_name)

    if os.path.exists(possible_path):
        ckpt_path = possible_path
    else:
        # 파일명 불일치 시 최신 파일 자동 탐색
        if os.path.exists(Config.CKPT_DIR):
            fs = [f for f in os.listdir(Config.CKPT_DIR) if f.endswith(".pth")]
            if fs: ckpt_path = max([os.path.join(Config.CKPT_DIR, f) for f in fs], key=os.path.getctime)

    if not ckpt_path:
        raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {Config.CKPT_DIR}")

    print(f"Loading weights from {os.path.basename(ckpt_path)}")
    cp = torch.load(ckpt_path, map_location=Config.DEVICE)

    state_dict = cp["model_state"] if isinstance(cp, dict) and "model_state" in cp else cp
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE).eval()

    # 5. 데이터셋 로드
    data_path = os.path.join(Config.OUTPUT_DIR, Config.RANKING_DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터셋 없음: {data_path}\nmake_ranking_df.py를 먼저 실행하세요.")

    ranking_df = pd.read_parquet(data_path)

    # 6. 추론
    print(f"Starting Inference ({len(ranking_df)} samples)...")
    all_scores = []
    BATCH_SIZE = 4096

    forward_params = inspect.signature(model.forward).parameters
    accepts_seq = "seq" in forward_params

    with torch.no_grad():
        for i in range(0, len(ranking_df), BATCH_SIZE):
            batch = ranking_df.iloc[i:i+BATCH_SIZE]
            Xi, Xv, seq = prepare_batch_inputs(batch, version, Config.DEVICE)

            if version == "v3" and accepts_seq:
                out = model(Xi, Xv, seq=seq)
            else:
                out = model(Xi, Xv)

            all_scores.append(torch.sigmoid(out).cpu().numpy().flatten())

    ranking_df["score"] = np.concatenate(all_scores)

    # 7. 평가 (HitRate, MRR)
    ranking_df["rank"] = ranking_df.groupby("user_session_idx")["score"].rank(method="first", ascending=False)
    targets = ranking_df[ranking_df["ranking_label"] == 1]

    hit5 = (targets["rank"] <= 5).mean()
    mrr = (1/targets["rank"]).mean()

    print(f"\n[Result] Model: {model_type.upper()} | Ver: {version.upper()}")
    print(f"Hit@5: {hit5:.4f} | MRR: {mrr:.4f}")

    save_name = f"inference_result_{model_type}_{version}.csv"
    save_path = os.path.join(Config.OUTPUT_DIR, save_name)
    targets.to_csv(save_path, index=False)
    print(f"Saved inference results to {save_path}")

    # ==================================================================================
    # 8. [Top-K 리스트 생성] + ID 복원 (Human Readable)
    # ==================================================================================
    TOP_K = 5
    print(f"\n[Info] Generating Top-{TOP_K} recommendation list with restored names...")

    # 1. 점수순 정렬 후 상위 K개 추출
    topk_df = ranking_df.sort_values(["user_session_idx", "score"], ascending=[True, False]).groupby("user_session_idx").head(TOP_K)

    # 2. 매핑 정보를 이용한 ID 복원 (역매핑)
    # 매핑 파일 구조: { 'brand': {'nike': 1, ...}, 'product_id': {'p100': 5, ...} }

    # (A) Product Name 복원
    if "product_id" in mappings:
        idx2prod = {v: k for k, v in mappings["product_id"].items()}
        topk_df["product_name_real"] = topk_df["product_id_idx"].map(idx2prod).fillna("unknown")

    # (B) Brand Name 복원
    if "brand" in mappings:
        idx2brand = {v: k for k, v in mappings["brand"].items()}
        topk_df["brand_name_real"] = topk_df["brand_idx"].map(idx2brand).fillna("unknown")

    # (C) Category Code 복원
    if "category_code" in mappings:
        idx2cat = {v: k for k, v in mappings["category_code"].items()}
        topk_df["category_name_real"] = topk_df["category_code_idx"].map(idx2cat).fillna("unknown")

    # 3. 저장할 컬럼 선별 (깔끔하게)
    # 존재하는 컬럼만 선택해서 저장
    candidate_cols = [
        "user_session_idx", "rank", "score", "ranking_label",
        "product_name_real", "brand_name_real", "category_name_real", # 복원된 이름
        "product_id_idx" # 원본 인덱스도 참고용으로 유지
    ]
    save_cols = [c for c in candidate_cols if c in topk_df.columns]

    topk_name = f"top{TOP_K}_recommendations_{model_type}_{version}.csv"
    topk_path = os.path.join(Config.OUTPUT_DIR, topk_name)
    topk_df[save_cols].to_csv(topk_path, index=False)

    print(f"[Success] Top-{TOP_K} recommendation file saved to: {topk_path}")

if __name__ == "__main__":
    main()
