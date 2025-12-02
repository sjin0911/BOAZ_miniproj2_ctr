import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random 
from tqdm import tqdm

from util.utils import get_device, build_model
from data.split import split_exact_tvt_from_preprocessed, extract_dims
from data.dataset import (
    DENSE_COLS,
    SPARSE_BASE_COLS,
    SEQ_COLS,
    create_v1_dataloaders,
    create_v2_dataloaders,
    create_v3_dataloaders,
)

def infer_field_names(field_size, data_version, has_seq=False):
    """
    Infer field names based on data version and field size.
    
    Args:
        field_size: Total number of fields in the attention map
        data_version: 'v1', 'v2', or 'v3'
        has_seq: Whether sequence feature is included (for v3)
    """
    base_cols = DENSE_COLS + SPARSE_BASE_COLS  # 11 columns
    
    if data_version == "v1":
        # V1: 5 dense + 6 sparse = 11
        return base_cols
    elif data_version == "v2":
        # V2: 5 dense + 6 sparse + 5 seq = 16
        seq_names = [f"seq_prod_{i}" for i in range(1, 6)]
        return base_cols + seq_names
    elif data_version == "v3":
        # V3: 5 dense + 6 sparse + 1 GRU = 12
        if has_seq:
            return base_cols + ["GRU_Sequence"]
        return base_cols
    else:
        # Fallback: use generic names
        return [f"Feat{i}" for i in range(field_size)]

def visualize_individual_heatmap_random(model, dataset, device, data_version, layer_idx=0, sample_idx=0):
    model.eval()
    data = dataset[sample_idx]
    
    xi = torch.tensor(data["xi"]).unsqueeze(0).to(device, dtype=torch.long)
    xv = torch.tensor(data["xv"]).unsqueeze(0).to(device, dtype=torch.float)
    
    seq = None
    has_seq = False
    if "seq" in data:
        seq = torch.tensor(data["seq"]).unsqueeze(0).to(device, dtype=torch.long)
        has_seq = True
        
    with torch.no_grad():
        output = model(xi, xv, seq=seq, return_attention=True)
        if isinstance(output, tuple):
            _, attn_list = output
        else:
            return False

    if layer_idx >= len(attn_list):
        return False

    attn_map = attn_list[layer_idx][0].cpu().numpy()
    field_names = infer_field_names(attn_map.shape[0], data_version, has_seq=has_seq)
    
    if len(field_names) != attn_map.shape[0]:
        field_names = [str(i) for i in range(attn_map.shape[0])]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        attn_map, 
        xticklabels=field_names, 
        yticklabels=field_names, 
        cmap="Blues", 
        square=True, 
        annot=False,
        linewidths=.5,
        linecolor='gray'
    )
    plt.title(f"[Individual] AutoInt Layer {layer_idx} (Random Sample Idx: {sample_idx})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_filename = f"heatmap_individual_layer{layer_idx}.png"
    plt.savefig(os.path.join(current_file_dir, save_filename))
    plt.close()
    print(f"[SAVED] Individual Map (User {sample_idx}): {save_filename}")
    return True

def visualize_global_heatmap(model, loader, device, data_version, layer_idx=0, max_batches=None):
    model.eval()
    total_attn_map = None
    count = 0
    
    print(f"\n[INFO] Calculating Global Average for Layer {layer_idx} (Full Validation Set)...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Global Avg Processing", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
                
            xi = batch["xi"].to(device)
            xv = batch["xv"].to(device)
            seq = batch["seq"].to(device) if "seq" in batch else None
            
            output = model(xi, xv, seq=seq, return_attention=True)
            if not isinstance(output, tuple): continue
                
            _, attn_list = output
            if layer_idx >= len(attn_list): return False
            
            batch_avg_map = attn_list[layer_idx].mean(dim=0).cpu().numpy()
            
            if total_attn_map is None:
                total_attn_map = batch_avg_map
            else:
                total_attn_map += batch_avg_map
            count += 1
            
    if count == 0: return False
        
    avg_attn_map = total_attn_map / count
    
    has_seq = "seq" in batch
    field_names = infer_field_names(avg_attn_map.shape[0], data_version, has_seq=has_seq)
    
    if len(field_names) != avg_attn_map.shape[0]:
        field_names = [str(i) for i in range(avg_attn_map.shape[0])]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        avg_attn_map, 
        xticklabels=field_names, 
        yticklabels=field_names, 
        cmap="Blues", 
        square=True, 
        annot=False,
        linewidths=.5,
        linecolor='gray'
    )
    plt.title(f"[Global Average] AutoInt Layer {layer_idx} (Avg of ALL {count} batches)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_filename = f"heatmap_global_layer{layer_idx}.png"
    plt.savefig(os.path.join(current_file_dir, save_filename))
    plt.close()
    print(f"[SAVED] Global Map: {save_filename}")
    return True

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    if not os.path.exists(config_path): config_path = "config.json"
    
    with open(config_path, "r") as f:
        config = json.load(f)

    if config["model_type"].lower() != "autoint":
        print(f"[WARNING] Model Type is {config['model_type']}. Visualization expects AutoInt.")

    # Properly handle device from config
    use_cuda_str = config.get('use_cuda', 'cpu')
    if use_cuda_str in ['cuda', 'mps']:
        device = torch.device(use_cuda_str)
    else:
        device = torch.device('cpu')
    
    data_path = config["data_dir"] + '/' + config["data_name"]
    if not os.path.isabs(data_path): data_path = os.path.join(current_dir, data_path)
    train_df, val_df, test_df = split_exact_tvt_from_preprocessed(parquet_path=data_path)
    
    try: base_sparse_dims, v2_sparse_dims = extract_dims(train_df, val_df, test_df)
    except: base_sparse_dims, v2_sparse_dims = extract_dims(train_df)

    loader_ver = config["data_loader"]
    
    if loader_ver == "v1": 
        _, val_loader, _ = create_v1_dataloaders(train_df, val_df, test_df, base_sparse_dims)
    elif loader_ver == "v2": 
        _, val_loader, _ = create_v2_dataloaders(train_df, val_df, test_df, v2_sparse_dims)
    elif loader_ver == "v3": 
        _, val_loader, _ = create_v3_dataloaders(train_df, val_df, test_df, v2_sparse_dims)
    
    seq_vocab_size = None
    if hasattr(val_loader.dataset, "seq_vocab_size"):
         seq_vocab_size = val_loader.dataset.seq_vocab_size
    feature_sizes = val_loader.dataset.field_dims
    model = build_model(config, feature_sizes, device, seq_vocab_size=seq_vocab_size)

    ckpt_path = os.path.join(config["save_dir"], f"best_{config['run_name']}.pth")
    if not os.path.isabs(ckpt_path): ckpt_path = os.path.join(current_dir, ckpt_path)
    
    if os.path.exists(ckpt_path):
        print(f"[INFO] Loading weights: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("[ERROR] Checkpoint not found.")
        return

    total_val_samples = len(val_loader.dataset)
    random_user_idx = random.randint(0, total_val_samples - 1)
    
    print("\n" + "="*50)
    print(f"   Visualizing AutoInt (Color: Blues)")
    print(f"   - Selected Individual: Random User Index {random_user_idx} / {total_val_samples}")
    print(f"   - Global Average: Calculating from FULL Validation Set")
    print("="*50)

    data_version = config["data_loader"]
    
    for i in range(6): 
        success_ind = visualize_individual_heatmap_random(model, val_loader.dataset, device, data_version, layer_idx=i, sample_idx=random_user_idx)
        
        success_glob = False
        if success_ind: 
             success_glob = visualize_global_heatmap(model, val_loader, device, data_version, layer_idx=i, max_batches=None)
        
        if not success_ind:
            break 

    print("\n[INFO] All visualizations finished!")

if __name__ == "__main__":
    main()