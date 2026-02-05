import dotenv
import torch
import os
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from itertools import product
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr
from typing import Dict, Tuple

def load_activation_hanoi(data_dir_name: str) -> pd.DataFrame:
    """データ全体をまとめて扱えるようにするために、DataFrameに変換する"""

    dotenv.load_dotenv()
    data_dir = os.getenv(data_dir_name)

    # 読み込んだデータをまとめる変数を用意
    data_list = []

    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            data = torch.load(os.path.join(data_dir, file))
            # numpyに変換しておく
            data["act_ids"] = data["act_ids"].int().numpy()
            data["feature_acts"] = data["feature_acts"].float().numpy()
            data["target_probs"] = data["target_probs"].float().numpy()
            data_list.append(data)

    # column:
    # layer, n, func_name, act_ids, feature_acts, target_probs, target_output
    return pd.DataFrame(data_list)

def load_activation_addition(data_dir_name: str) -> pd.DataFrame:
    """データ全体をまとめて扱えるようにするために、DataFrameに変換する"""

    dotenv.load_dotenv()
    data_dir = os.getenv(data_dir_name)

    # 読み込んだデータをまとめる変数を用意
    data_list = []

    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            loaded = torch.load(os.path.join(data_dir, file))
            layer = loaded["layer"]
            for entry in loaded["data"]:
                # numpyに変換しておく
                data_list.append({
                    "layer": layer,
                    "op1": entry["op1"],
                    "op2": entry["op2"],
                    "act_ids": entry["act_ids"].int().numpy(),
                    "feature_acts": entry["feature_acts"].float().numpy(),
                    "target_probs": entry["target_probs"].float().numpy(),
                    "target_output": entry["target_output"],
                })

    # column:
    # layer, op1, op2, act_ids, feature_acts, target_probs, target_output
    return pd.DataFrame(data_list)

def get_total_seqlen(data: pd.DataFrame, group_keys: list) -> int:
    """
    group_keysの組み合わせごとにseq_lenを取得し合計の系列長を求める
    -> 特徴の発火有無を考慮せずに計算してみる
    """

    # group_keysの組み合わせでグループ化し、各グループから1つのサンプルを取得
    unique_data = data.groupby(group_keys).first().reset_index()
    
    # 各データのfeature_actsのshape[1]を取得して合計
    total_seqlen = 0
    for _, row in unique_data.iterrows():
        feature_acts = row["feature_acts"]  # shape: (1, seq_len, m)
        seq_len = feature_acts.shape[1]
        total_seqlen += seq_len
    
    return total_seqlen

def get_unique_ids(data: pd.DataFrame, layer: int) -> ArrayLike:
    act_ids = data.loc[data["layer"] == layer, "act_ids"]
    # 各行のnumpy配列を全て結合
    all_ids = np.concatenate(act_ids.values)
    # 一意な要素を返す
    return np.unique(all_ids)

def build_feature_matrix(data: pd.DataFrame, layer: int, unique_ids: ArrayLike, total_seqlen: int) -> np.ndarray:
    """
    全てのact_idに対するfeature_actsを事前に行列形式で構築
    返り値: shape (len(unique_ids), total_seqlen)
    """
    filtered_data = data[(data["layer"] == layer)]
    
    # 結果を格納する行列（ゼロ初期化）
    feature_matrix = np.zeros((len(unique_ids), total_seqlen), dtype=np.float32)
    
    # act_idからインデックスへのマッピング
    id_to_idx = {act_id: idx for idx, act_id in enumerate(unique_ids)}
    
    # 各行のデータを処理
    offset = 0
    for _, row in filtered_data.iterrows():
        act_ids = row["act_ids"]  # shape: (n,)
        feature_acts = row["feature_acts"]  # shape: (1, m, n)
        seq_len = feature_acts.shape[1]
        
        # 各act_idについて対応する位置に値を格納
        for i, act_id in enumerate(act_ids):
            if act_id in id_to_idx:
                idx = id_to_idx[act_id]
                feature_matrix[idx, offset:offset + seq_len] = feature_acts[0, :, i]
        
        offset += seq_len
    
    return feature_matrix


def calculate_correlations_hanoi(
    data1: pd.DataFrame, layer1: int,
    data2: pd.DataFrame, layer2: int,
    include_inactive: bool = False # 発火しなかったトークンを特徴として考慮するかどうか
) -> pd.DataFrame:
    """data1とdata2の相関係数を計算する（高速化版）"""

    if include_inactive:
        # ここは無視する
        # total_seqlen1 = get_total_seqlen(data1, ["n", "func_name"])
        # total_seqlen2 = get_total_seqlen(data2, ["n", "func_name"])
        assert False, "include_inactive is not supported yet"
    else:
        total_seqlen1 = get_total_seqlen(data1, ["n", "func_name"])
        total_seqlen2 = get_total_seqlen(data2, ["n", "func_name"])
        print(f"total_seqlen1: {total_seqlen1}, total_seqlen2: {total_seqlen2}")
        assert total_seqlen1 == total_seqlen2, "total_seqlen1 != total_seqlen2"

    # layer内の一意なact_idを取得
    unique_ids1 = get_unique_ids(data1, layer1)
    unique_ids2 = get_unique_ids(data2, layer2)
    
    # 事前に全てのfeature_actsを行列形式で構築
    feature_matrix1 = build_feature_matrix(data1, layer1, unique_ids1, total_seqlen1)
    feature_matrix2 = build_feature_matrix(data2, layer2, unique_ids2, total_seqlen2)
    
    print(f"Feature matrix shapes: {feature_matrix1.shape}, {feature_matrix2.shape}")
    print(f"Calculating correlations for {len(unique_ids1)} x {len(unique_ids2)} = {len(unique_ids1) * len(unique_ids2)} combinations...")
    
    # 相関行列を計算（ベクトル化版、np.corrcoefと同じ挙動）
    n = total_seqlen1
    
    # 中心化（平均を引く）
    mean1 = feature_matrix1.mean(axis=1, keepdims=True)
    mean2 = feature_matrix2.mean(axis=1, keepdims=True)
    centered1 = feature_matrix1 - mean1
    centered2 = feature_matrix2 - mean2
    
    # 不偏標準偏差を計算
    std1 = np.sqrt((centered1 ** 2).sum(axis=1, keepdims=True) / (n - 1))
    std2 = np.sqrt((centered2 ** 2).sum(axis=1, keepdims=True) / (n - 1))
    
    # ゼロ除算を防ぐ
    assert std1.all() != 0, "std1 is all zero"
    assert std2.all() != 0, "std2 is all zero"
    
    # 標準化
    normalized1 = centered1 / std1
    normalized2 = centered2 / std2
    
    # 相関行列 = 標準化したベクトルの内積 / n
    # これはnp.corrcoefと同じ計算
    correlation_matrix = (normalized1 @ normalized2.T) / n

    print(f"correlation_matrix: {correlation_matrix}")
    
    # 結果をDataFrameに変換
    results = []
    for i, act_id1 in enumerate(tqdm(unique_ids1, desc="Converting to DataFrame")):
        for j, act_id2 in enumerate(unique_ids2):
            results.append({
                "act_id1": act_id1,
                "act_id2": act_id2,
                "correlation": correlation_matrix[i, j]
            })
    
    return pd.DataFrame(results)

def calculate_correlations_addition(
    data1: pd.DataFrame, layer1: int,
    data2: pd.DataFrame, layer2: int,
    include_inactive: bool = False # 発火しなかったトークンを特徴として考慮するかどうか
) -> pd.DataFrame:
    """data1とdata2の相関係数を計算する（高速化版）"""

    if include_inactive:
        # ここは無視する
        assert False, "include_inactive is not supported yet"
    else:
        total_seqlen1 = get_total_seqlen(data1, ["op1", "op2"])
        total_seqlen2 = get_total_seqlen(data2, ["op1", "op2"])
        print(f"total_seqlen1: {total_seqlen1}, total_seqlen2: {total_seqlen2}")
        assert total_seqlen1 == total_seqlen2, "total_seqlen1 != total_seqlen2"

    # layer内の一意なact_idを取得
    unique_ids1 = get_unique_ids(data1, layer1)
    unique_ids2 = get_unique_ids(data2, layer2)
    
    # 事前に全てのfeature_actsを行列形式で構築
    feature_matrix1 = build_feature_matrix(data1, layer1, unique_ids1, total_seqlen1)
    feature_matrix2 = build_feature_matrix(data2, layer2, unique_ids2, total_seqlen2)
    
    print(f"Feature matrix shapes: {feature_matrix1.shape}, {feature_matrix2.shape}")
    print(f"Calculating correlations for {len(unique_ids1)} x {len(unique_ids2)} = {len(unique_ids1) * len(unique_ids2)} combinations...")
    
    # 相関行列を計算（ベクトル化版、np.corrcoefと同じ挙動）
    n = total_seqlen1
    
    # 中心化（平均を引く）
    mean1 = feature_matrix1.mean(axis=1, keepdims=True)
    mean2 = feature_matrix2.mean(axis=1, keepdims=True)
    centered1 = feature_matrix1 - mean1
    centered2 = feature_matrix2 - mean2
    
    # 不偏標準偏差を計算
    std1 = np.sqrt((centered1 ** 2).sum(axis=1, keepdims=True) / (n - 1))
    std2 = np.sqrt((centered2 ** 2).sum(axis=1, keepdims=True) / (n - 1))
    
    # ゼロ除算を防ぐ
    assert std1.all() != 0, "std1 is all zero"
    assert std2.all() != 0, "std2 is all zero"
    
    # 標準化
    normalized1 = centered1 / std1
    normalized2 = centered2 / std2
    
    # 相関行列 = 標準化したベクトルの内積 / n
    # これはnp.corrcoefと同じ計算
    correlation_matrix = (normalized1 @ normalized2.T) / n

    print(f"correlation_matrix: {correlation_matrix}")
    
    # 結果をDataFrameに変換
    results = []
    for i, act_id1 in enumerate(tqdm(unique_ids1, desc="Converting to DataFrame")):
        for j, act_id2 in enumerate(unique_ids2):
            results.append({
                "act_id1": act_id1,
                "act_id2": act_id2,
                "correlation": correlation_matrix[i, j]
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    llama_dir = "LLAMA_DIR"
    llama_distill_dir = "LLAMA_DISTILL_DIR"

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer1", type=int, required=True)
    parser.add_argument("--layer2", type=int, required=True)
    args = parser.parse_args()
    layer1 = args.layer1
    layer2 = args.layer2

    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    os.makedirs(os.path.join(data_dir, "correlations"), exist_ok=True)
    correlations_dir = os.path.join(data_dir, "correlations")

    # Hanoi
    # data_llama = load_activation_hanoi(llama_dir)
    # data_llama_distill = load_activation_hanoi(llama_distill_dir)
    # print(f"layer1: {layer1}, layer2: {layer2}")
    # correlations = calculate_correlations_hanoi(data_llama, layer1, data_llama_distill, layer2)
    
    # Addition
    data_llama = load_activation_addition(llama_dir)
    data_llama_distill = load_activation_addition(llama_distill_dir)
    print(f"layer1: {layer1}, layer2: {layer2}")
    correlations = calculate_correlations_addition(data_llama, layer1, data_llama_distill, layer2)
    
    correlations.to_parquet(os.path.join(correlations_dir, f"l1_{layer1}_l2_{layer2}.parquet"), index=False)