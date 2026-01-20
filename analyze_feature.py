"""
モデル2に対してSAE活性値の相関が高いモデル1の層について,
特に相関の高い特徴を抽出して分析する
"""

from numpy.typing import ArrayLike
import pandas as pd
import dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualize_correlation import load_correlation
from analyze_correlation import load_activation, get_feature_acts

#########################################################################
# モデルに対してSAE活性値をablationしたときのlogitsの変化を計算する
#########################################################################
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import torch
from typing import Callable
from functools import partial
from model_config import llama_scope_lxr_32x, llama_scope_r1_distill
from plot_prob import load_model
from prompt_hanoi import get_answer

def ablation_hook(
    acts: torch.Tensor, # shape: [b, n, d_model]
    hook: HookPoint,
    ablate_feat_ids: torch.Tensor, # shape: [b]
    encode: Callable[[torch.Tensor], torch.Tensor],
    decode: Callable[[torch.Tensor], torch.Tensor],
    pos_start_abl: int=2
) -> torch.Tensor:
    features = encode(acts) # shape: [b, n, enc_dim]

    diff = acts - decode(features)
    
    # ablateする対象の軸(b, n, enc_dim)を取得
    abl_batch_ids = torch.arange(ablate_feat_ids.shape[0], device=acts.device) # b
    features[abl_batch_ids, pos_start_abl:, ablate_feat_ids] = 0

    # 復元
    acts_restored = decode(features) + diff # shape: [b, n, d_model]
    return acts_restored

@torch.no_grad()
def get_logits_diff(
    model: HookedTransformer,
    sae: SAE,
    text: str,
    target_output: str,
    ablate_feat_ids: torch.Tensor # shape: [b]
    ) -> ArrayLike:
    """
    モデルに対してSAE活性値をablationしたときのtarget_tokenにおけるlogitsの変化を計算する
    """

    input_ids = model.to_tokens(text, prepend_bos=True, truncate=False) # (1, n)
    input_ids = input_ids.expand(ablate_feat_ids.shape[0], -1).to(model.cfg.device) # (b, n)
    target_idx = model.to_tokens(target_output, prepend_bos=False, truncate=False).view(-1).to(model.cfg.device) # (m,)
    m = target_idx.size(0)

    logits = model.run_with_hooks(
        input_ids,
        return_type="logits",
        fwd_hooks=[]
    ) # (b, n, vocab_size)

    logits_ablated = model.run_with_hooks(
        input_ids,
        return_type="logits",
        fwd_hooks=[
            (sae.cfg.metadata.hook_name,
                partial(ablation_hook, ablate_feat_ids=ablate_feat_ids, encode=sae.encode, decode=sae.decode)
            )
        ]
    ) # (b, n, vocab_size)

    # logitsからtarget_outputの確率を計算
    logits = logits[:, -m-1:-1, :] # (b, m, vocab_size)
    logits_ablated = logits_ablated[:, -m-1:-1, :] # (b, m, vocab_size)
    logits_change = logits - logits_ablated # (b, m, vocab_size)
    logits_change = logits_change[:, torch.arange(m, device=logits.device), target_idx] # (b, m)

    return logits_change.cpu().float().numpy()

#########################################################################
# end
#########################################################################


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

# 見づらかったので使わない
# def visualize_all_correlations(
#     layer1: int,
#     layer2: int,
#     save_dir: str
# ) -> None:
#     """
#     指定した層の全ての特徴の相関を可視化する
#     """
#     correlations = load_correlation(layer1, layer2)
    
#     # データをピボットしてヒートマップ用の行列を作成
#     heatmap_data = correlations.pivot(index='act_id1', columns='act_id2', values='correlation')
    
#     # ヒートマップを描画
#     plt.figure(figsize=(12, 10))
#     im = plt.imshow(heatmap_data.values, aspect='auto', cmap='viridis', origin='lower')
#     plt.colorbar(im, label='Correlation')
#     plt.xlabel('Deepseek R1 Distill Llama 8B')
#     plt.ylabel('Llama3.1 8B')
#     plt.title(f'Feature Correlation Heatmap (Layer1: {layer1}, Layer2: {layer2})')
    
#     save_path = os.path.join(save_dir, f"heatmap_l1_{layer1}_l2_{layer2}.png")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"Saved heatmap to {save_path}")

def visualize_high_correlation_count(
    save_dir: str,
    threshold: float = 0.9,
    base: str = "act_id1"
) -> None:
    """
    層ごとにbaseに対して高い相関を持つ特徴の数の分布をプロットする
    32層を8x4に分割したグラフを1枚の画像として保存する
    """
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"
    
    save_path = os.path.join(save_dir, f"high_correlation_count_threshold_{threshold}_{base}.png")
    
    fig, axes = plt.subplots(8, 4, figsize=(16, 24), constrained_layout=True)
    
    for layer in tqdm(range(32), desc="Processing layers"):
        correlations = load_correlation(layer, layer)
        # baseごとに相関がthreshold以上の特徴の数をカウント
        high_correlations = correlations[correlations['correlation'] >= threshold]
        count_per_base = high_correlations.groupby(base)['correlation'].count()
        
        row = layer // 4
        col = layer % 4
        ax = axes[row, col]
        
        # ヒストグラムで分布をプロット
        ax.hist(count_per_base.values, bins=20, edgecolor='black', alpha=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel(f"Count (threshold={threshold})")
        ax.set_ylabel("Frequency")
        
        # x軸を整数表示にする
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 統計情報を追加
        mean_count = count_per_base.mean()
        ax.axvline(mean_count, color='red', linestyle='--', linewidth=1, label=f'Mean={mean_count:.1f}')
        ax.legend(fontsize=8)
    
    fig.suptitle(f"Distribution of High Correlation Counts per {base} (threshold={threshold})", fontsize=16)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

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
    特徴活性値のcos類似度を計算する(トークンが1対1で同じものに対応する必要がある)
    
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
                'layer': int(layer),
                'act_id1': int(act_id1),
                'act_id2': int(act_id2),
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

@torch.no_grad()
def get_attribution_to_logits(
    target_output: str,
    text: str,
    similarities: pd.DataFrame,
    save_dir: str,
    threshold: float = 0.9,
    start_layer: int = 0,
    end_layer: int = 32,
    batch_size: int = 2
) -> None:
    """
    特徴活性値のcos類似度が閾値以上のものを抽出する
    """
    sim_high_df = similarities[similarities['similarity'] > threshold]
    print(f"Processing {len(sim_high_df)} features with similarity > {threshold}")
    
    # 層の範囲でフィルタリング
    sim_high_df = sim_high_df[(sim_high_df['layer'] >= start_layer) & (sim_high_df['layer'] < end_layer)]
    print(f"Filtered to layers [{start_layer}, {end_layer}): {len(sim_high_df)} features")
    
    # 層ごとにグループ化
    grouped = sim_high_df.groupby('layer')
    total_layers = len(grouped)
    print(f"Total layers to process: {total_layers}")
    
    model1, sae1 = None, None
    model2, sae2 = None, None
    
    for layer_idx, (layer, group) in enumerate(grouped, 1):
        layer = int(layer)
        print(f"\n[{layer_idx}/{total_layers}] Processing Layer {layer} ({len(group)} features)...")
        
        # モデルをロード
        sae1, sae2 = None, None
        model_config1 = llama_scope_lxr_32x("cpu", layer)
        model_config2 = llama_scope_r1_distill("cpu", layer)
        model1, sae1 = load_model(model_config1, model=model1, sae=sae1)
        model2, sae2 = load_model(model_config2, model=model2, sae=sae2)
        
        # 特徴IDをリスト化
        act_ids1 = group['act_id1'].astype(int).tolist()
        act_ids2 = group['act_id2'].astype(int).tolist()
        
        # model1のバッチ処理
        print(f"  Processing model1 (batch_size={batch_size})...")
        model1.cuda()
        sae1.cuda()
        logits_diffs1 = []
        for i in tqdm(range(0, len(act_ids1), batch_size), desc=f"  Model1 Layer {layer}", leave=False):
            batch_ids = act_ids1[i:i+batch_size]
            batch_tensor = torch.tensor(batch_ids).cuda()
            diff = get_logits_diff(
                model1, sae1, text, target_output, ablate_feat_ids=batch_tensor
            )
            logits_diffs1.append(diff)
        logits_diffs1 = np.concatenate(logits_diffs1, axis=0) # (len(act_ids1), m)
        model1.cpu()
        sae1.cpu()
        
        # model2のバッチ処理
        print(f"  Processing model2 (batch_size={batch_size})...")
        model2.cuda()
        sae2.cuda()
        logits_diffs2 = []
        for i in tqdm(range(0, len(act_ids2), batch_size), desc=f"  Model2 Layer {layer}", leave=False):
            batch_ids = act_ids2[i:i+batch_size]
            batch_tensor = torch.tensor(batch_ids).cuda()
            diff = get_logits_diff(
                model2, sae2, text, target_output, ablate_feat_ids=batch_tensor
            )
            logits_diffs2.append(diff)
        logits_diffs2 = np.concatenate(logits_diffs2, axis=0) # (len(act_ids2), m)
        model2.cpu()
        sae2.cpu()
        
        # 層のattributionを構築
        layer_attribution = []
        for act_id1, act_id2, diff1, diff2 in zip(act_ids1, act_ids2, logits_diffs1, logits_diffs2):
            layer_attribution.append({
                'layer': layer,
                'act_id1': act_id1,
                'act_id2': act_id2,
                'diff1': diff1,
                'diff2': diff2
            })
        
        # 現在の層のデータを保存
        layer_df = pd.DataFrame(layer_attribution)
        layer_df.to_parquet(os.path.join(save_dir, f"layer_{layer}.parquet"))
        print(f"  Saved to layer_{layer}.parquet")
        
    return

if __name__ == "__main__":

    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    # step1: 特徴活性値のcos類似度を計算する
    # data_llama = load_activation("LLAMA_DIR")
    # data_llama_distill = load_activation("LLAMA_DISTILL_DIR")

    # os.makedirs("./images/compare_features", exist_ok=True)
    # os.makedirs(data_dir, exist_ok=True)
    # similarities = get_feature_similarities(data_llama, data_llama_distill)
    # similarities.to_parquet(os.path.join(data_dir, "similarities.parquet"))
    # visualize_feature_similarities(similarities, "./images/compare_features/feature_similarities.png")

    # step2: 全ての層の特徴の相関を可視化する
    # os.makedirs("./images/high_correlation_count", exist_ok=True)
    # visualize_high_correlation_count(
    #     save_dir="./images/high_correlation_count",
    #     threshold=0.9,
    #     base="act_id1"
    # )
    # visualize_high_correlation_count(
    #     save_dir="./images/high_correlation_count",
    #     threshold=0.9,
    #     base="act_id2"
    # )

    # step3: 特徴活性値のcos類似度が閾値以上のものについてlogitsの変化を計算する
    save_dir = os.path.join(data_dir, "attribution")
    os.makedirs(save_dir, exist_ok=True)

    similarities = pd.read_parquet(os.path.join(data_dir, "similarities.parquet"))
    prompt, target_output = get_answer(3)
    
    get_attribution_to_logits(
        target_output,
        prompt,
        similarities,
        save_dir=save_dir,
        threshold=0.9,
        start_layer=0,
        end_layer=32,
        batch_size=2
    )
