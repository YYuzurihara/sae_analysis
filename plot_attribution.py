from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_attribution(layer: int, dir_path: str) -> pd.DataFrame:
    attribution_path = os.path.join(dir_path, "attribution", f"layer_{layer}.parquet")
    # columns: act_id1, act_id2, diff1, diff2
    # diff1, diff2 shape: [seq_len]
    return pd.read_parquet(attribution_path)

def calc_cos_similarity(attribution: pd.DataFrame) -> pd.DataFrame:
    """
    各rowについてdiff1, diff2を展開し、それぞれのペアに対してcos類似度を計算する
    diff1, diff2: shape = [seq_len] などの1次元配列（list, np.ndarray等）を想定
    返り値: 新しいDataFrameにcos_similarity列を追加して返す
    """
    def compute_cos_sim(row):
        a = np.array(row["diff1"])
        b = np.array(row["diff2"])
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # ゼロベクトルの場合は0を返す
        if norm_a == 0 or norm_b == 0:
            print(f"Warning: Zero vector detected for row {row['act_id1']}, {row['act_id2']}")
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    attribution = attribution.copy()
    attribution["cos_similarity"] = attribution.apply(compute_cos_sim, axis=1)
    return attribution["cos_similarity"]

def plot_attribution_similarity(dir_path: str, save_path: str, num_layers: int = 32):
    """
    全層についてcos類似度を計算し、その分布をヒストグラムでプロットする
    
    Args:
        dir_path: データディレクトリのパス
        save_path: 保存先のパス
        num_layers: 層の数（デフォルト32）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 各層の類似度を格納するリスト
    all_similarities = []
    
    # 全層についてcos類似度を計算
    for layer in range(num_layers):
        try:
            attribution = load_attribution(layer, dir_path)
            cos_similarity = calc_cos_similarity(attribution)
            all_similarities.append(cos_similarity.values)
            print(f"Layer {layer}: {len(cos_similarity)} samples processed")
        except Exception as e:
            print(f"Error: Failed to process layer {layer}: {e}")
            raise
    
    # 8x4のグリッドでプロット作成
    fig, axes = plt.subplots(8, 4, figsize=(12, 18))
    axes = axes.flatten()
    
    # 各層のヒストグラムをプロット
    for layer in range(num_layers):
        ax = axes[layer]
        
        similarities = all_similarities[layer]
        ax.hist(similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Layer {layer}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Attribution Similarity', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)
        
        # x軸の範囲を[-1, 1]に固定
        ax.set_xlim([-1, 1])
        
        # 統計情報を表示
        mean_sim = np.mean(similarities)
        ax.axvline(mean_sim, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean: {mean_sim:.3f}')
        ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    
    # 全32層の類似度分布をプロット
    save_path = "./images/attribution_similarity.png"
    plot_attribution_similarity(data_dir, save_path, num_layers=32)