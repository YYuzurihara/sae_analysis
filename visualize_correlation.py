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
) -> float:
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"

    correlations = load_correlation(layer1, layer2)
    # act_id1ごとに最大相関値を求める
    max_corrs = correlations.groupby(base)['correlation'].max()
    return max_corrs

def visualize_correlation(
    save_path: str,
    base: str = "act_id1",
    mode: str = "mean", # mean
) -> None:
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"

    heatmap = np.zeros((32, 32), dtype=float)
    for layer1 in tqdm(range(32)):
        for layer2 in range(32):
            max_corrs = find_max_correlation(layer1, layer2, base)
            max_corr = max_corrs.mean()
            heatmap[layer1, layer2] = max_corr

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="max correlation")
    plt.xlabel("DeepSeek-R1-Distill-Llama-8B")
    plt.ylabel("Llama-3.1-8B")
    plt.title(f"Feature correlation ({mode}, base={base})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    visualize_correlation(
        save_path="images/correlation/correlation_heatmap_id1.png",
        base="act_id1"
    )
    visualize_correlation(
        save_path="images/correlation/correlation_heatmap_id2.png",
        base="act_id2"
    )