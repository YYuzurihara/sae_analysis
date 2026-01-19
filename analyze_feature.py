"""
モデル2に対してSAE活性値の相関が高いモデル1の層について,
特に相関の高い特徴を抽出して分析する
"""

import pandas as pd
import dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.typing import ArrayLike
from visualize_correlation import load_correlation
from analyze_correlation import load_activation, get_feature_acts

def find_max_correlation_df(
    layer1: int,
    layer2: int,
    base: str = "act_id1"
) -> pd.DataFrame:
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"

    correlations = load_correlation(layer1, layer2)
    # baseごとに最大相関値を持つ行（＝相関ペア）を残す
    idx = correlations.groupby(base)['correlation'].idxmax()
    max_corrs_df = correlations.loc[idx].reset_index(drop=True)
    # act_id1, act_id2, correlationの全てを含むDataFrameを返す
    return max_corrs_df

def get_high_correlation_features(
    layer1: int,
    layer2: int,
    base: str = "act_id1",
    threshold: float = 0.9
) -> pd.DataFrame:
    """
    モデルBに対してSAE活性値の相関が高いモデルAの層について,
    特に相関の高い特徴を抽出して分析する
    """
    max_corrs_df = find_max_correlation_df(layer1, layer2, base)
    max_corrs_df = max_corrs_df[max_corrs_df['correlation'] > threshold]
    # column: act_id1, act_id2, correlation
    return max_corrs_df

def get_feature_similarities(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
) -> pd.DataFrame:
    """
    特徴活性値を比較する(トークンが1対1で同じものに対応する必要がある)
    
    Return:
    pd.DataFrame with columns: layer, act_id1, act_id2, similarity
    """
    similarities = []
    for layer in tqdm(range(32)):
        max_corrs_df = find_max_correlation_df(layer, layer, base="act_id1")

        for act_id1, act_id2 in zip(max_corrs_df['act_id1'], max_corrs_df['act_id2']):
            feature_acts1 = get_feature_acts(data1, layer, act_id1)
            feature_acts2 = get_feature_acts(data2, layer, act_id2)
            if feature_acts1.size == 0 or feature_acts2.size == 0:
                similarity = 0.0
            else:
                similarity = (feature_acts1 @ feature_acts2) / (
                    (np.linalg.norm(feature_acts1) + 1e-8) * (np.linalg.norm(feature_acts2) + 1e-8)
                )
            similarities.append({
                'layer': layer,
                'act_id1': act_id1,
                'act_id2': act_id2,
                'similarity': similarity
            })
    return pd.DataFrame(similarities)

def visualize_feature_similarities(
    similarities: pd.DataFrame,
    save_path: str
) -> None:
    """
    各層の特徴活性値のcos類似度を棒グラフで可視化する
    """
    num_layers = 32
    n_rows, n_cols = 8, 4
    bins = 100
    hist_range = (0, 1)
    hist_bins = np.linspace(hist_range[0], hist_range[1], bins + 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 30), sharex=True, sharey=True)
    axes = axes.flatten()

    for layer in range(num_layers):
        ax = axes[layer]
        layer_sims = similarities[similarities['layer'] == layer]['similarity'].values
        counts, edges = np.histogram(layer_sims, bins=hist_bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=edges[1] - edges[0], align='center')
        ax.set_title(f'Layer {layer}')
        if layer % n_cols == 0:
            ax.set_ylabel('Frequency')
        if layer >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Similarity')

    fig.suptitle('Feature Similarities Distribution by Layer')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    data_llama = load_activation("LLAMA_DIR")
    data_llama_distill = load_activation("LLAMA_DISTILL_DIR")

    os.makedirs("./images/compare_features", exist_ok=True)

    similarities = get_feature_similarities(data_llama, data_llama_distill)
    visualize_feature_similarities(similarities, "./images/compare_features/feature_similarities.png")