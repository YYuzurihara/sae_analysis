import dotenv
import torch
import os
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from itertools import product
from tqdm import tqdm
import argparse

def load_activation(data_dir_name: str) -> pd.DataFrame:
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

def get_total_seqlen(data: pd.DataFrame) -> int:
    """
    n, func_nameの組み合わせごとにseq_lenを取得し合計の系列長を求める
    -> 特徴の発火有無を考慮せずに計算してみる
    """

    # (n, func_name)の組み合わせでグループ化し、各グループから1つのサンプルを取得
    unique_data = data.groupby(["n", "func_name"]).first().reset_index()
    
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

def get_feature_acts(data: pd.DataFrame, layer: int, act_id: int) -> ArrayLike:
    filtered_data = data[(data["layer"] == layer)]
    
    # 各行からact_idに対応するfeature_actsを取得
    feature_acts_list = []
    for _, row in filtered_data.iterrows():
        act_ids = row["act_ids"]  # shape: (n,)
        feature_acts = row["feature_acts"]  # shape: (1, m, n)
        
        # act_idsの中からact_idのインデックスを探す
        indices = np.where(act_ids == act_id)[0]
        if len(indices) > 0:
            i = indices[0]
            feature_acts_list.append(feature_acts[0, :, i])  # shape: (m,)
    
    # すべてのfeature_actsを結合
    if len(feature_acts_list) > 0:
        return np.concatenate(feature_acts_list)
    else:
        return np.array([])


def calculate_correlations(
    data1: pd.DataFrame, layer1: int,
    data2: pd.DataFrame, layer2: int,
    include_inactive: bool = False # 発火しなかったトークンを特徴として考慮するかどうか
) -> pd.DataFrame:
    """data1とdata2の相関係数を計算する"""

    if include_inactive:
        # ここは無視する
        # total_seqlen1 = get_total_seqlen(data1)
        # total_seqlen2 = get_total_seqlen(data2)
        assert False, "include_inactive is not supported yet"
    else:
        total_seqlen1 = get_total_seqlen(data1)
        total_seqlen2 = get_total_seqlen(data2)
        print(f"total_seqlen1: {total_seqlen1}, total_seqlen2: {total_seqlen2}")
        assert total_seqlen1 == total_seqlen2, "total_seqlen1 != total_seqlen2"

    # layer内の一意なact_idを取得
    unique_ids1 = get_unique_ids(data1, layer1)
    unique_ids2 = get_unique_ids(data2, layer2)

    # 各act_idに対応するfeature_actsの平均と標準偏差を計算
    results = []
    total_combinations = len(unique_ids1) * len(unique_ids2)
    for act_id1, act_id2 in tqdm(product(unique_ids1, unique_ids2), total=total_combinations):
        feature_acts1 = get_feature_acts(data1, layer1, act_id1)
        feature_acts2 = get_feature_acts(data2, layer2, act_id2)

        # 非発火トークンを0で埋め、total_seqlenでshapeを揃える
        if feature_acts1.shape[0] != total_seqlen1:
            feature_acts1 = np.pad(feature_acts1, (0, total_seqlen1 - feature_acts1.shape[0]), mode="constant")
        if feature_acts2.shape[0] != total_seqlen2:
            feature_acts2 = np.pad(feature_acts2, (0, total_seqlen2 - feature_acts2.shape[0]), mode="constant")

        correlation = np.corrcoef(feature_acts1, feature_acts2) # shape: (2, 2)
        results.append({
            "act_id1": act_id1,
            "act_id2": act_id2,
            "correlation": correlation[0, 1]
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

    data_llama = load_activation(llama_dir)
    data_llama_distill = load_activation(llama_distill_dir)
    print(f"layer1: {layer1}, layer2: {layer2}")
    correlations = calculate_correlations(data_llama, layer1, data_llama_distill, layer2)

    correlations.to_parquet(os.path.join(correlations_dir, f"l1_{layer1}_l2_{layer2}.parquet"), index=False)