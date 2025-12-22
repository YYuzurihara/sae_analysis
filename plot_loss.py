"""
1. lossから最小と最大のlossを抽出してプロットする
2. 抽出した内容をcsvに保存する
"""

import os
import dotenv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def load_tensors(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ディレクトリの存在チェック
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    ce_losses = []
    reconstruction_loss = None
    base_ce_loss = None
    for file in os.listdir(data_dir):
        if "ce_loss_batch" in file:
            ce_loss = torch.load(os.path.join(data_dir, file), map_location="cpu").float() # [b, seq_len, 3]
            ce_losses.append(ce_loss)
        elif "reconstruction_loss" in file:
            reconstruction_loss = torch.load(os.path.join(data_dir, file), map_location="cpu").float() # [1, seq_len]
        elif "base_ce_loss" in file:
            base_ce_loss = torch.load(os.path.join(data_dir, file), map_location="cpu").float() # [1, seq_len]
    return torch.cat(ce_losses, dim=0), reconstruction_loss, base_ce_loss

def get_min_max_loss(
    ce_losses: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ce_losses: [b, seq_len, 3] where 3 = (pos, feature, loss)
    losses = ce_losses[:, :, 2]  # [b, seq_len]
    min_indices = torch.argmin(losses, dim=0)  # [seq_len]
    max_indices = torch.argmax(losses, dim=0)  # [seq_len]
    
    # Gather the full [seq_len, 3] data for each min/max batch
    pos_indices = torch.arange(ce_losses.shape[1]) # [seq_len]
    min_loss = ce_losses[min_indices, pos_indices, :]  # [seq_len, 3]
    max_loss = ce_losses[max_indices, pos_indices, :]  # [seq_len, 3]
    return min_loss, max_loss

def plot_loss(
    min_loss: torch.Tensor,
    max_loss: torch.Tensor,
    base_ce_loss: torch.Tensor,
    save_dir: str,
    layer: int
) -> None:
    # min_loss, max_loss, base_lossの3つをプロット
    plt.figure()
    plt.plot(min_loss[:, 2].view(-1), label="minimum")
    plt.plot(max_loss[:, 2].view(-1), label="maximum")
    plt.plot(base_ce_loss.view(-1), label="baseline")
    plt.legend(loc='upper right')
    plt.xlabel("sequence")
    plt.ylabel("loss")
    plt.title(f"L{layer} minimum vs maximum vs baseline losses")
    plt.savefig(os.path.join(save_dir, f"L{layer}_losses.png"))

    # min_loss, base_lossの2つをプロット
    plt.figure()
    plt.plot(min_loss[:, 2].view(-1), label="minimum")
    plt.plot(base_ce_loss.view(-1), label="baseline")
    plt.legend(loc='upper right')
    plt.xlabel("sequence")
    plt.ylabel("loss")
    plt.title(f"L{layer} minimum vs baseline losses")
    plt.savefig(os.path.join(save_dir, f"L{layer}_min_base_losses.png"))

def heatmap_loss(
    ce_losses: torch.Tensor,
    base_ce_loss: torch.Tensor,
    save_dir: str,
    layer: int,
    target_pos: int|None = None,
    abl_pos: int|None = None,
    abl_feat: int|None = None
) -> None:
    if target_pos is not None:
        # pos, featを軸にlossをheatmapでプロット
        losses = ce_losses[:, target_pos, :] # [b, 3]
        
        # [b, 3] -> [p, f] にリシェイプ (3 = (abl_pos, abl_feature, loss))
        abl_positions = losses[:, 0].long()  # [b]
        abl_features = losses[:, 1].long()   # [b]
        losses = losses[:, 2]                 # [b]
        
        # posとfeatの数を取得（0-indexedを仮定）
        n_pos = abl_positions.max().item() + 1
        n_feat = abl_features.max().item() + 1
        
        # 2D loss matrix [p, f] を作成
        loss_matrix = torch.zeros(n_pos, n_feat)
        loss_matrix[abl_positions, abl_features] = losses
        
        # loss_matrix[pos, feat] でlossを取得可能
        


def save_loss(
    min_loss: torch.Tensor,
    max_loss: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    base_ce_loss: torch.Tensor,
) -> None:
    pass

if __name__ == "__main__":
    TARGET_LAYER = 16
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR") + f"/L{TARGET_LAYER}"

    os.makedirs(f"images", exist_ok=True)

    ce_losses, reconstruction_loss, base_ce_loss = load_tensors(data_dir)
    min_loss, max_loss = get_min_max_loss(ce_losses)
    plot_loss(min_loss, max_loss, base_ce_loss, "images", TARGET_LAYER)
    # heatmap_loss(ce_losses, base_ce_loss, "images", TARGET_LAYER, target_pos=0)
    # save_loss(min_loss, max_loss, reconstruction_loss, base_ce_loss)