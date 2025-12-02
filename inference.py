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
    print("[Warn] Model files not found. Please check the path.")

# ==================================================================================
# Configuration
# ==================================================================================
class Config:
    DATA_DIR = "./data"
    CKPT_DIR = "./ckpts"
    OUTPUT_DIR = "./inference"

    # Input/Output Filenames
    RANKING_DATA_FILE = "ranking_df.parquet"
    RESULT_FILE = "inference_results.csv"

    MAPPING_FILE = "ID_Index_mappings.pkl"
    CONFIG_FILE = "config.json"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Column Definitions (DO NOT MODIFY)
DENSE_COLS = ["is_weekend", "price_scaled", "session_rank_scaled", "hour_scaled", "day_of_week_scaled"]
SPARSE_BASE_COLS = ["product_id_idx", "category_id_idx", "category_code_idx", "brand_idx", "user_id_idx", "user_session_idx"]
SEQ_COLS = ["seq_prod_1_idx", "seq_prod_2_idx", "seq_prod_3_idx", "seq_prod_4_idx", "seq_prod_5_idx"]

def load_json(path):
    with open(path, 'r') as f: return json.load(f)
def load_pickle(path):
    with open(path, 'rb') as f: return pickle.load(f)

def prepare_batch_inputs(batch_df, version, device):
    """Convert data to model input format (Tensor) based on version (V1/V2/V3)"""
    dense_vals = batch_df[DENSE_COLS].values.astype(np.float32)
    sparse_vals = batch_df[SPARSE_BASE_COLS].values.astype(np.int64)
    seq_vals = batch_df[SEQ_COLS].fillna(0).values.astype(np.int64)

    # Logic for generating Xi, Xv
    if version in ["v1", "v3"]:
        Xi = np.concatenate((np.zeros_like(dense_vals, dtype=np.int64), sparse_vals), axis=1)
        Xv = np.concatenate((dense_vals, np.ones_like(sparse_vals, dtype=np.float32)), axis=1)
    elif version == "v2":
        Xi = np.concatenate((np.zeros_like(dense_vals, dtype=np.int64), sparse_vals, seq_vals), axis=1)
        Xv = np.concatenate((dense_vals, np.ones_like(sparse_vals, dtype=np.float32), np.ones_like(seq_vals, dtype=np.float32)), axis=1)
    else:
        # Default handling as V3
        return prepare_batch_inputs(batch_df, "v3", device)

    Xi_tensor = torch.tensor(Xi, dtype=torch.long, device=device).unsqueeze(-1)
    Xv_tensor = torch.tensor(Xv, dtype=torch.float, device=device).unsqueeze(-1)

    # Return seq tensor only for V3
    seq_tensor = torch.tensor(seq_vals, dtype=torch.long, device=device) if version == "v3" else None

    return Xi_tensor, Xv_tensor, seq_tensor

def main():
    print(f"=== Auto-Detect Inference Started on {Config.DEVICE} ===")

    # 1. Load Config and Auto-Detect
    config_path = os.path.join(os.getcwd(), Config.CONFIG_FILE)
    if os.path.exists(config_path):
        train_config = load_json(config_path)
        print(f"Loaded config from {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # (1) Detect Model Type (from "model_type" in config.json)
    model_type = train_config.get("model_type", "AutoInt").lower()
    # (2) Detect Data Version (from "data_loader" in config.json)
    version = train_config.get("data_loader", "v3").lower()

    print(f"[Auto-Detect] Target Model: {model_type.upper()} | Data Version: {version.upper()}")

    # 2. Load Mapping File and Calculate Feature Sizes
    mappings = load_pickle(os.path.join(Config.DATA_DIR, Config.MAPPING_FILE))

    # Calculate exact size without +1 (Matches training code)
    base_sparse_dims = [len(mappings.get(c.replace("_idx", ""), [])) for c in SPARSE_BASE_COLS]
    product_vocab_size = len(mappings.get("product_id", []))
    dense_dims = [1] * len(DENSE_COLS)

    if version == "v1": feature_sizes = dense_dims + base_sparse_dims
    elif version == "v2": feature_sizes = dense_dims + base_sparse_dims + [product_vocab_size]*len(SEQ_COLS)
    else: feature_sizes = dense_dims + base_sparse_dims # V3

    # 3. Initialize Model (Dynamic Selection)
    if "autoint" in model_type:
        ModelClass = AutoInt
    elif "deepfm" in model_type:
        ModelClass = DeepFM
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # [Key] Logic mapping Config keys to Model arguments
    def get_valid_kwargs(cls, cfg):
        # JSON Key : Model __init__ argument name mapping
        key_map = {
            "autoint_dnn_hidden_units": "dnn_hidden_units",
            "deepfm_hidden_dims": "hidden_dims",  # Ensure DeepFM compatibility
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

    # [Safety] Force inject hidden_dims if missing for DeepFM
    if "DeepFM" in ModelClass.__name__:
        if "hidden_dims" not in model_kwargs:
             if "deepfm_hidden_dims" in train_config:
                 model_kwargs["hidden_dims"] = train_config["deepfm_hidden_dims"]
             else:
                 # Last resort: Use default values observed in logs
                 print("[Warn] Config missing. Force using [150, 300, 150]")
                 model_kwargs["hidden_dims"] = [150, 300, 150]

    # Update common required arguments
    model_params = inspect.signature(ModelClass.__init__).parameters
    if "feature_sizes" in model_params:
        model_kwargs['feature_sizes'] = feature_sizes

    # DeepFM often doesn't have a device argument
    if "device" in model_params:
        model_kwargs['device'] = Config.DEVICE
    else:
        model_kwargs['use_cuda'] = torch.cuda.is_available()

    if version == "v3" and "seq_vocab_size" in model_params:
        model_kwargs['seq_vocab_size'] = product_vocab_size

    # Log configuration check
    if "hidden_dims" in model_kwargs:
        print(f"[Config Check] Loaded hidden_dims: {model_kwargs['hidden_dims']}")
    elif "dnn_hidden_units" in model_kwargs:
        print(f"[Config Check] Loaded dnn_hidden_units: {model_kwargs['dnn_hidden_units']}")

    print(f"Initializing {ModelClass.__name__}...")
    model = ModelClass(**model_kwargs)

    # 4. Load Weights
    ckpt_path = None
    target_name = f"best_{model_type}.pth"
    possible_path = os.path.join(Config.CKPT_DIR, target_name)

    if os.path.exists(possible_path):
        ckpt_path = possible_path
    else:
        # Auto-detect latest file if filename mismatch
        if os.path.exists(Config.CKPT_DIR):
            fs = [f for f in os.listdir(Config.CKPT_DIR) if f.endswith(".pth")]
            if fs: ckpt_path = max([os.path.join(Config.CKPT_DIR, f) for f in fs], key=os.path.getctime)

    if not ckpt_path:
        raise FileNotFoundError(f"Model weight file not found: {Config.CKPT_DIR}")

    print(f"Loading weights from {os.path.basename(ckpt_path)}")
    cp = torch.load(ckpt_path, map_location=Config.DEVICE)

    state_dict = cp["model_state"] if isinstance(cp, dict) and "model_state" in cp else cp
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE).eval()

    # 5. Load Dataset
    data_path = os.path.join(Config.OUTPUT_DIR, Config.RANKING_DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}\nPlease run make_ranking_df.py first.")

    ranking_df = pd.read_parquet(data_path)

    # 6. Inference
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

    # 7. Evaluation (HitRate, MRR)
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
    # 8. [Generate Top-K List] + Restore IDs (Human Readable)
    # ==================================================================================
    TOP_K = 5
    print(f"\n[Info] Generating Top-{TOP_K} recommendation list with restored names...")

    # 1. Sort by score and extract Top K
    topk_df = ranking_df.sort_values(["user_session_idx", "score"], ascending=[True, False]).groupby("user_session_idx").head(TOP_K)

    # 2. Restore IDs using mapping info (Reverse Mapping)
    # Mapping file structure: { 'brand': {'nike': 1, ...}, 'product_id': {'p100': 5, ...} }

    # (A) Restore Product Name
    if "product_id" in mappings:
        idx2prod = {v: k for k, v in mappings["product_id"].items()}
        topk_df["product_name_real"] = topk_df["product_id_idx"].map(idx2prod).fillna("unknown")

    # (B) Restore Brand Name
    if "brand" in mappings:
        idx2brand = {v: k for k, v in mappings["brand"].items()}
        topk_df["brand_name_real"] = topk_df["brand_idx"].map(idx2brand).fillna("unknown")

    # (C) Restore Category Code
    if "category_code" in mappings:
        idx2cat = {v: k for k, v in mappings["category_code"].items()}
        topk_df["category_name_real"] = topk_df["category_code_idx"].map(idx2cat).fillna("unknown")

    # 3. Select columns to save (Clean format)
    # Select only existing columns
    candidate_cols = [
        "user_session_idx", "rank", "score", "ranking_label",
        "product_name_real", "brand_name_real", "category_name_real", # Restored names
        "product_id_idx" # Keep original index for reference
    ]
    save_cols = [c for c in candidate_cols if c in topk_df.columns]

    topk_name = f"top{TOP_K}_recommendations_{model_type}_{version}.csv"
    topk_path = os.path.join(Config.OUTPUT_DIR, topk_name)
    topk_df[save_cols].to_csv(topk_path, index=False)

    print(f"[Success] Top-{TOP_K} recommendation file saved to: {topk_path}")

if __name__ == "__main__":
    main()
