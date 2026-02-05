from dotenv import load_dotenv
from numpy.typing import ArrayLike
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import torch
from typing import Callable
from functools import partial
from model_config import llama_scope_lxr_32x, llama_scope_r1_distill
from plot_prob import load_model

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

def filter_features(
    correlations: pd.DataFrame,
    similarities: pd.DataFrame,
    correlation_threshold: float = 0.99,
    similarity_threshold: float = 0.99
    ) -> pd.DataFrame:
    """
    相関とコサイン類似度の両方が閾値以上の特徴ペアを抽出する
    
    Parameters:
    -----------
    correlations : pd.DataFrame
        相関係数データ（列: act_id1, act_id2, correlation, layer1, layer2）
    similarities : pd.DataFrame
        コサイン類似度データ（列: act_id1, act_id2, cosine_similarity, layer1, layer2）
    correlation_threshold : float
        相関係数の閾値
    similarity_threshold : float
        コサイン類似度の閾値
    
    Returns:
    --------
    pd.DataFrame
        両方の条件を満たす特徴ペア（列: act_id1, act_id2, layer1, layer2, correlation, cosine_similarity）
    """
    # 各データフレームを閾値でフィルタリング
    filtered_corr = correlations[correlations['correlation'] > correlation_threshold].copy()
    filtered_sim = similarities[similarities['cosine_similarity'] > similarity_threshold].copy()
    
    # 共通するペアを内部結合で抽出
    # マージキー: act_id1, act_id2, layer1, layer2
    merged = pd.merge(
        filtered_corr[['act_id1', 'act_id2', 'layer1', 'layer2', 'correlation']],
        filtered_sim[['act_id1', 'act_id2', 'layer1', 'layer2', 'cosine_similarity']],
        on=['act_id1', 'act_id2', 'layer1', 'layer2'],
        how='inner'
    )
    
    print(f"Common pairs: {len(merged)} pairs")
    
    return merged

@torch.no_grad()
def get_attribution_to_logits(
    target_output: str,
    text: str,
    filtered_features: pd.DataFrame,
    task_type: str,
    batch_size: int = 2
) -> pd.DataFrame:
    """
    filter_featuresで得られた特徴ペアについて、logit変化量を計算する

    Parameters:
    -----------
    target_output : str
        目標出力文字列
    text : str
        入力テキスト
    filtered_features : pd.DataFrame
        filter_features関数で得られた特徴ペア
        列: act_id1, act_id2, layer1, layer2, correlation, cosine_similarity
    task_type : str
        タスクの種類（例: "hanoi", "addition"）
    batch_size : int
        バッチサイズ
    
    Returns:
    --------
    pd.DataFrame
        全てのattributionデータをまとめたDataFrame
    """
    print(f"Processing {len(filtered_features)} feature pairs")

    # layer1とlayer2のペアごとにグループ化
    grouped = filtered_features.groupby(['layer1', 'layer2'])
    total_groups = len(grouped)
    print(f"Total layer pairs to process: {total_groups}")

    model1, sae1 = None, None
    model2, sae2 = None, None
    
    # 全てのattributionデータを格納するリスト
    all_attributions = []

    for group_idx, ((layer1, layer2), group) in enumerate(grouped, 1):
        layer1, layer2 = int(layer1), int(layer2)
        print(f"\n[{group_idx}/{total_groups}] Processing Layer1={layer1}, Layer2={layer2} ({len(group)} features)...")

        # モデルをロード
        sae1, sae2 = None, None
        model_config1 = llama_scope_lxr_32x("cpu", layer1)
        model_config2 = llama_scope_r1_distill("cpu", layer2)
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
        for i in tqdm(range(0, len(act_ids1), batch_size), desc=f"  Model1 L{layer1}", leave=False):
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
        for i in tqdm(range(0, len(act_ids2), batch_size), desc=f"  Model2 L{layer2}", leave=False):
            batch_ids = act_ids2[i:i+batch_size]
            batch_tensor = torch.tensor(batch_ids).cuda()
            diff = get_logits_diff(
                model2, sae2, text, target_output, ablate_feat_ids=batch_tensor
            )
            logits_diffs2.append(diff)
        logits_diffs2 = np.concatenate(logits_diffs2, axis=0) # (len(act_ids2), m)
        model2.cpu()
        sae2.cpu()

        # 層ペアのattributionを構築
        layer_attribution = []
        for idx, (act_id1, act_id2, diff1, diff2) in enumerate(
            zip(act_ids1, act_ids2, logits_diffs1, logits_diffs2)
        ):
            row = group.iloc[idx]
            layer_attribution.append({
                'layer1': layer1,
                'layer2': layer2,
                'act_id1': act_id1,
                'act_id2': act_id2,
                'correlation': row['correlation'],
                'cosine_similarity': row['cosine_similarity'],
                'diff1': diff1,
                'diff2': diff2,
                'task_type': task_type
            })

        # 現在の層ペアのデータをリストに追加
        all_attributions.extend(layer_attribution)
        print(f"  Processed {len(layer_attribution)} features for Layer1={layer1}, Layer2={layer2}")
    
    # 全てのattributionデータをDataFrameにまとめて返す
    result_df = pd.DataFrame(all_attributions)
    print(f"\nTotal attributions collected: {len(result_df)}")
    return result_df

@torch.no_grad()
def get_attribution_to_logits_addition(
    target_output: str,
    text: str,
    filtered_features: pd.DataFrame,
    task_type: str
) -> pd.DataFrame:
    """
    filter_featuresで得られた特徴ペアについて、logit変化量を計算する

    Parameters:
    -----------
    target_output : str
        目標出力文字列
    text : str
        入力テキスト
    filtered_features : pd.DataFrame
        filter_features関数で得られた特徴ペア
        列: act_id1, act_id2, layer1, layer2, correlation, cosine_similarity
    task_type : str
        タスクの種類（例: "hanoi", "addition"）
    batch_size : int
        バッチサイズ
    
    Returns:
    --------
    pd.DataFrame
        全てのattributionデータをまとめたDataFrame
    """
    print(f"Processing {len(filtered_features)} feature pairs")

    # layer1とlayer2のペアごとにグループ化
    grouped = filtered_features.groupby(['layer1', 'layer2'])
    total_groups = len(grouped)
    print(f"Total layer pairs to process: {total_groups}")

    model1, sae1 = None, None
    model2, sae2 = None, None
    
    # 全てのattributionデータを格納するリスト
    all_attributions = []

    for group_idx, ((layer1, layer2), group) in enumerate(grouped, 1):
        layer1, layer2 = int(layer1), int(layer2)
        print(f"\n[{group_idx}/{total_groups}] Processing Layer1={layer1}, Layer2={layer2} ({len(group)} features)...")

        # モデルをロード
        sae1, sae2 = None, None
        model_config1 = llama_scope_lxr_32x("cpu", layer1)
        model_config2 = llama_scope_r1_distill("cpu", layer2)
        model1, sae1 = load_model(model_config1, model=model1, sae=sae1)
        model2, sae2 = load_model(model_config2, model=model2, sae=sae2)

        # 特徴IDをリスト化
        act_ids1 = group['act_id1'].astype(int).tolist()
        act_ids2 = group['act_id2'].astype(int).tolist()

        # model1のバッチ処理
        model1.cuda()
        sae1.cuda()
        # すべてのact_ids1をまとめて一度にバッチ処理
        batch_tensor = torch.tensor(act_ids1).cuda()
        logits_diffs1 = get_logits_diff(
            model1, sae1, text, target_output, ablate_feat_ids=batch_tensor
        )
        model1.cpu()
        sae1.cpu()

        # model2のバッチ処理
        model2.cuda()
        sae2.cuda()
        batch_tensor = torch.tensor(act_ids2).cuda()
        logits_diffs2 = get_logits_diff(
            model2, sae2, text, target_output, ablate_feat_ids=batch_tensor
        )
        model2.cpu()
        sae2.cpu()

        # 層ペアのattributionを構築
        layer_attribution = []
        for idx, (act_id1, act_id2, diff1, diff2) in enumerate(
            zip(act_ids1, act_ids2, logits_diffs1, logits_diffs2)
        ):
            row = group.iloc[idx]
            layer_attribution.append({
                'layer1': layer1,
                'layer2': layer2,
                'act_id1': act_id1,
                'act_id2': act_id2,
                'correlation': row['correlation'],
                'cosine_similarity': row['cosine_similarity'],
                'diff1': diff1,
                'diff2': diff2,
                'task_type': task_type
            })

        # 現在の層ペアのデータをリストに追加
        all_attributions.extend(layer_attribution)
        print(f"  Processed {len(layer_attribution)} features for Layer1={layer1}, Layer2={layer2}")
    
    # 全てのattributionデータをDataFrameにまとめて返す
    result_df = pd.DataFrame(all_attributions)
    print(f"\nTotal attributions collected: {len(result_df)}")
    return result_df

if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    attribution_dir = os.path.join(data_dir, "attribution")
    os.makedirs(attribution_dir, exist_ok=True)

    task_type = "addition"
    # 相関とコサイン類似度のデータを読み込み
    correlations = pd.read_parquet(f"images/correlation/correlation_{task_type}_act_id1.parquet")
    similarities = pd.read_parquet(f"images/cosine_similarity/cosine_similarity_{task_type}_act_id1.parquet")
    filtered_features = filter_features(correlations, similarities)

    # プロンプトとその解答を取得
    # from prompt_hanoi import get_answer
    # prompt, target_output = get_answer(3, func_name="solve")

    from prompt_addition import get_answer
    
    # ランダムに100個のop1, op2ペアを生成
    np.random.seed(42)  # 再現性のためのシード
    random_pairs = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(10)]
    
    # ランダムに選ばれたペアについてresult_dfを取得し、結合する
    all_results = []
    for idx, (op1, op2) in enumerate(tqdm(random_pairs, desc="Processing random pairs")):
        prompt, target_output = get_answer(op1, op2)
        
        # logit変化量を計算
        result_df = get_attribution_to_logits_addition(
            target_output=target_output,
            text=prompt,
            filtered_features=filtered_features,
            task_type=task_type
        )
        
        # op1とop2の情報を追加
        result_df['op1'] = op1
        result_df['op2'] = op2
        
        all_results.append(result_df)
    
    # 全ての結果を結合して保存
    final_result_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal results: {len(final_result_df)} rows")
    final_result_df.to_parquet(os.path.join(attribution_dir, f"attribution_{task_type}_act_id1.parquet"))