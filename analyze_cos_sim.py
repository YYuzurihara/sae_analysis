from numpy.typing import ArrayLike
import pandas as pd
import dotenv
import os
import numpy as np
from tqdm import tqdm
from analyze_correlation import load_activation_hanoi, load_activation_addition, get_total_seqlen, get_unique_ids, build_feature_matrix
import argparse

def calculate_cosine_similarity_addition(
    data1: pd.DataFrame, layer1: int,
    data2: pd.DataFrame, layer2: int,
    include_inactive: bool = False # 発火しなかったトークンを特徴として考慮するかどうか
) -> pd.DataFrame:
    """data1とdata2のコサイン類似度を計算する（高速化版）"""

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
    print(f"Calculating cosine similarities for {len(unique_ids1)} x {len(unique_ids2)} = {len(unique_ids1) * len(unique_ids2)} combinations...")
    
    # コサイン類似度を計算（ベクトル化版）
    # コサイン類似度 = (A · B) / (||A|| * ||B||)
    
    # L2ノルム（ベクトルの大きさ）を計算
    norm1 = np.sqrt((feature_matrix1 ** 2).sum(axis=1, keepdims=True))
    norm2 = np.sqrt((feature_matrix2 ** 2).sum(axis=1, keepdims=True))
    
    # ゼロ除算を防ぐ（ゼロベクトルがないことを確認）
    assert (norm1 > 0).all(), "norm1 has zero vectors"
    assert (norm2 > 0).all(), "norm2 has zero vectors"
    
    # 正規化（単位ベクトルにする）
    normalized1 = feature_matrix1 / norm1
    normalized2 = feature_matrix2 / norm2
    
    # コサイン類似度 = 正規化したベクトルの内積
    cosine_similarity_matrix = normalized1 @ normalized2.T

    print(f"cosine_similarity_matrix: {cosine_similarity_matrix}")
    
    # 結果をDataFrameに変換
    results = []
    for i, act_id1 in enumerate(tqdm(unique_ids1, desc="Converting to DataFrame")):
        for j, act_id2 in enumerate(unique_ids2):
            results.append({
                "act_id1": act_id1,
                "act_id2": act_id2,
                "cosine_similarity": cosine_similarity_matrix[i, j]
            })
    
    return pd.DataFrame(results)

def calculate_cosine_similarity_hanoi(
    data1: pd.DataFrame, layer1: int,
    data2: pd.DataFrame, layer2: int,
    include_inactive: bool = False # 発火しなかったトークンを特徴として考慮するかどうか
) -> pd.DataFrame:
    """data1とdata2のコサイン類似度を計算する（高速化版・hanoi用）"""

    if include_inactive:
        # ここは無視する
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
    print(f"Calculating cosine similarities for {len(unique_ids1)} x {len(unique_ids2)} = {len(unique_ids1) * len(unique_ids2)} combinations...")
    
    # コサイン類似度を計算（ベクトル化版）
    # コサイン類似度 = (A · B) / (||A|| * ||B||)
    
    # L2ノルム（ベクトルの大きさ）を計算
    norm1 = np.sqrt((feature_matrix1 ** 2).sum(axis=1, keepdims=True))
    norm2 = np.sqrt((feature_matrix2 ** 2).sum(axis=1, keepdims=True))
    
    # ゼロ除算を防ぐ（ゼロベクトルがないことを確認）
    assert (norm1 > 0).all(), "norm1 has zero vectors"
    assert (norm2 > 0).all(), "norm2 has zero vectors"
    
    # 正規化（単位ベクトルにする）
    normalized1 = feature_matrix1 / norm1
    normalized2 = feature_matrix2 / norm2
    
    # コサイン類似度 = 正規化したベクトルの内積
    cosine_similarity_matrix = normalized1 @ normalized2.T

    print(f"cosine_similarity_matrix: {cosine_similarity_matrix}")
    
    # 結果をDataFrameに変換
    results = []
    for i, act_id1 in enumerate(tqdm(unique_ids1, desc="Converting to DataFrame")):
        for j, act_id2 in enumerate(unique_ids2):
            results.append({
                "act_id1": act_id1,
                "act_id2": act_id2,
                "cosine_similarity": cosine_similarity_matrix[i, j]
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer1", type=int, required=True)
    parser.add_argument("--layer2", type=int, required=True)
    parser.add_argument("--dataset", type=str, choices=["addition", "hanoi"], required=True,
                        help="Dataset to use: 'addition' or 'hanoi'")
    args = parser.parse_args()

    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    similarity_dir = os.path.join(data_dir, "cos_sim")
    os.makedirs(similarity_dir, exist_ok=True)

    if args.dataset == "addition":
        data_llama = load_activation_addition("LLAMA_DIR")
        data_llama_distill = load_activation_addition("LLAMA_DISTILL_DIR")
        similarities = calculate_cosine_similarity_addition(data_llama, args.layer1, data_llama_distill, args.layer2)
        output_file = f"similarities_addition_layer{args.layer1}_layer{args.layer2}.parquet"
    elif args.dataset == "hanoi":
        data_llama = load_activation_hanoi("LLAMA_DIR")
        data_llama_distill = load_activation_hanoi("LLAMA_DISTILL_DIR")
        similarities = calculate_cosine_similarity_hanoi(data_llama, args.layer1, data_llama_distill, args.layer2)
        output_file = f"similarities_hanoi_layer{args.layer1}_layer{args.layer2}.parquet"
    
    similarities.to_parquet(os.path.join(similarity_dir, output_file))
    print(f"Saved to {os.path.join(similarity_dir, output_file)}")