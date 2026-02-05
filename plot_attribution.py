from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_attribution(
    task_type: str,
    dir_path: str
) -> pd.DataFrame:
    """
    layer_attribution.parquetを読み込む
    """
    attribution_path = os.path.join(dir_path, "attribution", f"attribution_{task_type}_act_id1.parquet")
    # columns: layer1, layer2, act_id1, act_id2, correlation, cosine_similarity, diff1, diff2, task_type
    # diff1, diff2 shape: [seq_len]
    attribution = pd.read_parquet(attribution_path)
    # "diff1", "diff2" の両方が0ベクトル（全要素が0）の行を削除
    def is_both_zero_vec(row):
        d1 = np.array(row["diff1"])
        d2 = np.array(row["diff2"])
        return np.all(d1 == 0) and np.all(d2 == 0)
    mask = ~attribution.apply(is_both_zero_vec, axis=1)
    attribution = attribution[mask].reset_index(drop=True)
    if task_type == "addition":
        # groupbyで(layer1, act_id1, layer2, act_id2)ごとにまとめ、diff1/diff2は行列としてstackして1つのベクトルにまとめる
        grouped = (
            attribution
            .groupby(["layer1", "act_id1", "layer2", "act_id2"], as_index=False)
            .agg({
                "diff1": lambda x: np.concatenate([np.array(v).flatten() for v in x]),
                "diff2": lambda x: np.concatenate([np.array(v).flatten() for v in x]),
                # 他のカラムは最初の値にする（correlation, cosine_similarity, task_type等）
                "correlation": "first",
                "task_type": "first"
            })
        )
        attribution = grouped
    return attribution

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
        if norm_a == 0 and norm_b == 0:
            print(f"Warning: Zero vector detected for row {row['act_id1']}, {row['act_id2']}")
            return np.nan
        elif norm_a == 0 or norm_b == 0:
            return 0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    attribution = attribution.copy()
    attribution["cos_similarity"] = attribution.apply(compute_cos_sim, axis=1).dropna()
    return attribution["cos_similarity"]

def plot_attribution_similarity(dir_path: str, save_path: str, task_type: str):
    """
    task_typeについてcos類似度を計算し、その分布をヒストグラムでプロットする
    
    Args:
        dir_path: データディレクトリのパス
        save_path: 保存先のパス
        task_type: タスクタイプ
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # task_typeについてcos類似度を計算
    attribution = load_attribution(task_type, dir_path)
    cos_similarity = calc_cos_similarity(attribution)
    print(f"Task type {task_type}: {len(cos_similarity)} samples processed")
    
    # ヒストグラムでプロット
    plt.hist(cos_similarity.values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Attribution Similarity for {task_type}')
    plt.xlabel('Attribution Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    attribution_fig_dir = os.path.join("images", "attribution")
    os.makedirs(attribution_fig_dir, exist_ok=True)

    task_type = "addition"
    
    # タスクタイプの類似度分布をプロット
    save_path = os.path.join(attribution_fig_dir, f"attribution_similarity_{task_type}.png")
    plot_attribution_similarity(data_dir, save_path, task_type)