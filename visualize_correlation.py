import os
import dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_correlation(
    layer1: int,
    layer2: int
) -> pd.DataFrame:
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    parquet_name = f"l1_{layer1}_l2_{layer2}.parquet"
    parquet_path = os.path.join(data_dir, "correlations", parquet_name)
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    # column:
    # act_id1, act_id2, correlation
    return pd.read_parquet(parquet_path)

def find_max_correlation(
    layer1: int,
    layer2: int,
    base: str = "act_id1"
) -> pd.Series:
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"

    correlations = load_correlation(layer1, layer2)
    # act_id1ごとに最大相関値を求める
    max_corrs = correlations.groupby(base)['correlation'].max()
    return max_corrs

def count_highly_correlated_features(
    layer1: int,
    layer2: int,
    threshold: float = 0.9,
    base: str = "act_id1"
) -> pd.Series:
    """
    指定した層に対して、特徴ごとに相関値がthreshold以上のものの数をカウントする
    """
    correlations = load_correlation(layer1, layer2)
    cnt_correlated = correlations[correlations['correlation'] >= threshold].groupby(base)['correlation'].count()
    return cnt_correlated


def visualize_correlation(
    save_path: str,
    base: str = "act_id1"
) -> None:
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"

    labels = ["Llama-3.1-8B", "DeepSeek-R1-Distill-Llama-8B"]

    heatmap = np.zeros((32, 32), dtype=float)
    for layer1 in tqdm(range(32)):
        for layer2 in range(32):
            max_corrs = find_max_correlation(layer1, layer2, base)
            max_corr = max_corrs.mean()
            heatmap[layer1, layer2] = max_corr

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="max correlation")
    if base == "act_id1":
        plt.xlabel(labels[1])
        plt.ylabel(labels[0])
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.title(f"Feature correlation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_correlation_dist(
    save_path: str,
    base: str = "act_id1",
    debug: bool = False
) -> None:
    """
    同じ層の活性化値の相関の分布をプロットする
    """
    fig, axes = plt.subplots(8, 4, figsize=(16, 24), constrained_layout=True)
    for layer in tqdm(range(32)):
        max_corrs = find_max_correlation(layer, layer, base=base)

        if debug:
            print(f"max correlation of layer {layer}: {max_corrs.mean()}")
            top_100 = max_corrs.sort_values(ascending=False).head(100)
            print(top_100.values)

        row = layer // 4
        col = layer % 4
        ax = axes[row, col]
        ax.hist(max_corrs, bins=10)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("correlation")
        ax.set_ylabel("frequency")

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

def visualize_count_highly_correlated_features(
    save_path: str,
    base: str = "act_id1"
) -> None:
    """
    指定した層に対して、特徴ごとに相関値がthreshold以上のものの数をカウントしてプロットする
    """
    fig, axes = plt.subplots(8, 4, figsize=(16, 24), constrained_layout=True)
    for layer in tqdm(range(32)):
        cnt_correlated = count_highly_correlated_features(layer, layer, base=base)
        row = layer // 4
        col = layer % 4
        ax = axes[row, col]
        ax.bar(cnt_correlated.values)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("count of highly correlated features")
        ax.set_ylabel("frequency")

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    # visualize_correlation(
    #     save_path="images/correlation/correlation_heatmap_id1.png",
    #     base="act_id1"
    # )
    # visualize_correlation(
    #     save_path="images/correlation/correlation_heatmap_id2.png",
    #     base="act_id2"
    # )

    # visualize_correlation_dist(
    #     save_path="images/correlation/correlation_distribution_id1_all_layers.png",
    #     base="act_id1"
    # )
    # visualize_correlation_dist(
    #     save_path="images/correlation/correlation_distribution_id2_all_layers.png",
    #     base="act_id2"
    # )

    visualize_count_highly_correlated_features(
        save_path="images/correlation/count_highly_correlated_features_id1.png",
        base="act_id1"
    )
    visualize_count_highly_correlated_features(
        save_path="images/correlation/count_highly_correlated_features_id2.png",
        base="act_id2"
    )