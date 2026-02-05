import os
import dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_correlation(layer1: int = None, layer2: int = None) -> pd.DataFrame:
    """
    指定したlayer1またはlayer2について他方のlayerすべてのcorrelationを読み込む
    両方Noneはエラー
    .envのDATA_DIRがタスクの種類を決める
    レイヤー情報を列として追加する
    """
    if layer1 is not None and layer2 is not None:
        raise ValueError("どちらか一方だけ指定してください (layer1 または layer2)")
    if layer1 is None and layer2 is None:
        raise ValueError("layer1かlayer2のどちらかを指定してください")

    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    all_correlations = []
    if layer1 is not None:
        for l2 in tqdm(range(32), desc=f"Layer2 loop for fixed layer1={layer1}"):
            parquet_name = f"l1_{layer1}_l2_{l2}.parquet"
            parquet_path = os.path.join(data_dir, "correlations", parquet_name)
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"parquet not found: {parquet_path}")

            correlations = pd.read_parquet(parquet_path)
            # レイヤー情報を追加
            correlations['layer1'] = layer1
            correlations['layer2'] = l2
            all_correlations.append(correlations)
    elif layer2 is not None:
        for l1 in tqdm(range(32), desc=f"Layer1 loop for fixed layer2={layer2}"):
            parquet_name = f"l1_{l1}_l2_{layer2}.parquet"
            parquet_path = os.path.join(data_dir, "correlations", parquet_name)
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"parquet not found: {parquet_path}")

            correlations = pd.read_parquet(parquet_path)
            # レイヤー情報を追加
            correlations['layer1'] = l1
            correlations['layer2'] = layer2
            all_correlations.append(correlations)

    return pd.concat(all_correlations, ignore_index=True)

def find_max_correlation(
    layer1: int = None,
    layer2: int = None
) -> pd.Series:
    """
    指定したlayer1またはlayer2について、全レイヤとの相関のうち最大値を取得
    layer1指定時はact_id1ごとに、layer2指定時はact_id2ごとに最大値を返す
    """
    # layer1が指定されている場合はbase="act_id1"、layer2の場合は"act_id2"
    if layer1 is not None:
        base = "act_id1"
    elif layer2 is not None:
        base = "act_id2"
    else:
        raise ValueError("layer1かlayer2のどちらかを指定してください")
    
    correlations = load_correlation(layer1=layer1, layer2=layer2)
    # baseで指定したact_idごとに最大相関値を求める（全レイヤとの比較から）
    max_corrs = correlations.groupby(base)['correlation'].max()
    return max_corrs

def load_cosine_similarity(
    layer1: int = None,
    layer2: int = None,
    dataset: str = None
) -> pd.DataFrame:
    """
    指定したlayer1またはlayer2について、全レイヤとのコサイン類似度を読み込む
    両方Noneはエラー
    レイヤー情報を列として追加する
    """
    if layer1 is not None and layer2 is not None:
        raise ValueError("どちらか一方だけ指定してください (layer1 または layer2)")
    if layer1 is None and layer2 is None:
        raise ValueError("layer1かlayer2のどちらかを指定してください")
    
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    
    all_similarities = []
    if layer1 is not None:
        for l2 in tqdm(range(32), desc=f"Layer2 loop for fixed layer1={layer1}"):
            parquet_name = f"similarities_{dataset}_layer{layer1}_layer{l2}.parquet"
            parquet_path = os.path.join(data_dir, "cos_sim", parquet_name)
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"parquet not found: {parquet_path}")
            
            similarities = pd.read_parquet(parquet_path)
            # レイヤー情報を追加
            similarities['layer1'] = layer1
            similarities['layer2'] = l2
            all_similarities.append(similarities)
    elif layer2 is not None:
        for l1 in tqdm(range(32), desc=f"Layer1 loop for fixed layer2={layer2}"):
            parquet_name = f"similarities_{dataset}_layer{l1}_layer{layer2}.parquet"
            parquet_path = os.path.join(data_dir, "cos_sim", parquet_name)
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"parquet not found: {parquet_path}")
            
            similarities = pd.read_parquet(parquet_path)
            # レイヤー情報を追加
            similarities['layer1'] = l1
            similarities['layer2'] = layer2
            all_similarities.append(similarities)
    
    # column: act_id1, act_id2, cosine_similarity, layer1, layer2
    return pd.concat(all_similarities, ignore_index=True)

def find_max_cosine_similarity(
    layer1: int = None,
    layer2: int = None,
    dataset: str = None
) -> pd.Series:
    """
    指定したlayer1またはlayer2について、全レイヤとのコサイン類似度のうち最大値を取得
    layer1指定時はact_id1ごとに、layer2指定時はact_id2ごとに最大値を返す
    """
    assert dataset in ["addition", "hanoi"], "dataset must be 'addition' or 'hanoi'"
    
    # layer1が指定されている場合はbase="act_id1"、layer2の場合は"act_id2"
    if layer1 is not None:
        base = "act_id1"
    elif layer2 is not None:
        base = "act_id2"
    else:
        raise ValueError("layer1かlayer2のどちらかを指定してください")
    
    similarities = load_cosine_similarity(layer1=layer1, layer2=layer2, dataset=dataset)
    # baseで指定したact_idごとに最大コサイン類似度を求める（全レイヤとの比較から）
    max_sims = similarities.groupby(base)['cosine_similarity'].max()
    return max_sims


def visualize_correlation(
    save_path: str,
    base: str = "act_id1"
) -> None:
    """
    全レイヤについて相関を可視化
    各レイヤを固定したとき、対応するレイヤの頻度をヒートマップで表示
    マッチした特徴ペアをparquetファイルとして保存
    """
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"
    
    labels = ["Llama-3.1-8B", "DeepSeek-R1-Distill-Llama-8B"]
    
    heatmap = np.zeros((32, 32), dtype=float)
    all_matched_pairs = []
    
    if base == "act_id1":
        # layer1を固定して、各act_id1について最大相関を持つlayer2を特定
        for layer1 in tqdm(range(32), desc="Layer1 loop"):
            correlations = load_correlation(layer1=layer1)
            # 各act_id1について最大相関を持つレイヤ（対応するlayer2）を特定
            max_layer2 = correlations.loc[correlations.groupby('act_id1')['correlation'].idxmax()]
            all_matched_pairs.append(max_layer2)
            # layer2の頻度をカウント（列から直接取得）
            layer2_counts = max_layer2['layer2'].value_counts()
            for layer2, count in layer2_counts.items():
                heatmap[layer1, layer2] = count / len(max_layer2)
    else:  # base == "act_id2"
        # layer2を固定して、各act_id2について最大相関を持つlayer1を特定
        for layer2 in tqdm(range(32), desc="Layer2 loop"):
            correlations = load_correlation(layer2=layer2)
            # 各act_id2について最大相関を持つレイヤ（対応するlayer1）を特定
            max_layer1 = correlations.loc[correlations.groupby('act_id2')['correlation'].idxmax()]
            all_matched_pairs.append(max_layer1)
            # layer1の頻度をカウント（列から直接取得）
            layer1_counts = max_layer1['layer1'].value_counts()
            for layer1, count in layer1_counts.items():
                heatmap[layer1, layer2] = count / len(max_layer1)
    
    # マッチした特徴ペアを結合してparquetとして保存
    # matched_pairs_df = pd.concat(all_matched_pairs, ignore_index=True)
    # parquet_path = save_path.replace('.png', '.parquet')
    # matched_pairs_df.to_parquet(parquet_path)
    # print(f"Saved matched pairs to {parquet_path}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="frequency")
    if base == "act_id1":
        plt.xlabel(labels[1])
        plt.ylabel(labels[0])
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.title(f"Feature correlation layer correspondence frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_cosine_similarity(
    save_path: str,
    dataset: str,
    base: str = "act_id1"
) -> None:
    """
    全レイヤについてコサイン類似度を可視化
    各レイヤを固定したとき、対応するレイヤの頻度をヒートマップで表示
    マッチした特徴ペアをparquetファイルとして保存
    """
    assert base in ["act_id1", "act_id2"], "base must be 'act_id1' or 'act_id2'"
    
    labels = ["Llama-3.1-8B", "DeepSeek-R1-Distill-Llama-8B"]
    
    heatmap = np.zeros((32, 32), dtype=int)
    all_matched_pairs = []
    
    if base == "act_id1":
        # layer1を固定して、各act_id1について最大類似度を持つlayer2を特定
        for layer1 in tqdm(range(32), desc="Layer1 loop"):
            similarities = load_cosine_similarity(layer1=layer1, dataset=dataset)
            # 各act_id1について最大類似度を持つレイヤ（対応するlayer2）を特定
            max_layer2 = similarities.loc[similarities.groupby('act_id1')['cosine_similarity'].idxmax()]
            all_matched_pairs.append(max_layer2)
            # layer2の頻度をカウント（列から直接取得）
            layer2_counts = max_layer2['layer2'].value_counts()
            for layer2, count in layer2_counts.items():
                heatmap[layer1, layer2] = count
    else:  # base == "act_id2"
        # layer2を固定して、各act_id2について最大類似度を持つlayer1を特定
        for layer2 in tqdm(range(32), desc="Layer2 loop"):
            similarities = load_cosine_similarity(layer2=layer2, dataset=dataset)
            # 各act_id2について最大類似度を持つレイヤ（対応するlayer1）を特定
            max_layer1 = similarities.loc[similarities.groupby('act_id2')['cosine_similarity'].idxmax()]
            all_matched_pairs.append(max_layer1)
            # layer1の頻度をカウント（列から直接取得）
            layer1_counts = max_layer1['layer1'].value_counts()
            for layer1, count in layer1_counts.items():
                heatmap[layer1, layer2] = count
    
    # マッチした特徴ペアを結合してparquetとして保存
    matched_pairs_df = pd.concat(all_matched_pairs, ignore_index=True)
    parquet_path = save_path.replace('.png', '.parquet')
    matched_pairs_df.to_parquet(parquet_path)
    print(f"Saved matched pairs to {parquet_path}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="frequency")
    if base == "act_id1":
        plt.xlabel(labels[1])
        plt.ylabel(labels[0])
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.title(f"Feature cosine similarity layer correspondence frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_filtered_features(
    filtered_features: pd.DataFrame,
    save_path: str,
    title: str = None,
    correlation_threshold: float = None,
    similarity_threshold: float = None
) -> None:
    """
    analyze_attribution.filter_features で得られた特徴ペアのレイヤ対応をヒートマップで可視化
    
    Parameters:
    -----------
    filtered_features : pd.DataFrame
        filter_features関数の出力（列: act_id1, act_id2, layer1, layer2, correlation, cosine_similarity）
    save_path : str
        画像の保存先パス
    title : str, optional
        グラフのタイトル（Noneの場合は自動生成）
    correlation_threshold : float, optional
        相関係数の閾値（タイトル表示用）
    similarity_threshold : float, optional
        コサイン類似度の閾値（タイトル表示用）
    """
    labels = ["Llama-3.1-8B", "DeepSeek-R1-Distill-Llama-8B"]
    
    # layer1とlayer2の組み合わせごとにカウント
    heatmap = np.zeros((32, 32), dtype=int)
    layer_counts = filtered_features.groupby(['layer1', 'layer2']).size()
    
    for (layer1, layer2), count in layer_counts.items():
        heatmap[int(layer1), int(layer2)] = count
    
    # 統計情報を表示
    total_pairs = len(filtered_features)
    nonzero_cells = np.sum(heatmap > 0)
    print(f"Total filtered feature pairs: {total_pairs}")
    print(f"Non-zero layer pairs: {nonzero_cells}")
    print(f"Max frequency: {np.max(heatmap)}")
    
    # ヒートマップを作成
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Frequency")
    
    ax.set_xlabel(f"{labels[1]} (layer)", fontsize=12)
    ax.set_ylabel(f"{labels[0]} (layer)", fontsize=12)
    
    # タイトルを設定
    if title is None:
        title_parts = ["Filtered Feature Layer Correspondence"]
        if correlation_threshold is not None and similarity_threshold is not None:
            title_parts.append(f"\n(correlation > {correlation_threshold}, cosine similarity > {similarity_threshold})")
        title = "".join(title_parts)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 軸の目盛りを設定
    ax.set_xticks(np.arange(0, 32, 4))
    ax.set_yticks(np.arange(0, 32, 4))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved filtered features heatmap to {save_path}")

def plot_combined_distribution_boxplot(
    df_corr1: pd.DataFrame,
    df_corr2: pd.DataFrame,
    df_sim1: pd.DataFrame,
    df_sim2: pd.DataFrame,
    task_type: str,
    save_path: str
) -> None:
    """
    4つのDataFrameから相関係数とコサイン類似度の分布をまとめて箱ひげ図で可視化
    レイヤ情報は無視し、値の分布のみを表示
    
    Parameters:
    -----------
    df_corr1 : pd.DataFrame
        act_id1ベースの相関係数データ（"correlation"列を含む）
    df_corr2 : pd.DataFrame
        act_id2ベースの相関係数データ（"correlation"列を含む）
    df_sim1 : pd.DataFrame
        act_id1ベースのコサイン類似度データ（"cosine_similarity"列を含む）
    df_sim2 : pd.DataFrame
        act_id2ベースのコサイン類似度データ（"cosine_similarity"列を含む）
    task_type : str
        タスクの種類（例: "hanoi", "addition"）
    save_path : str
        画像の保存先パス
    """
    # モデル名を定義
    model_names = ["Llama-3.1-8B", "DeepSeek-R1-Distill-Llama-8B"]
    
    # データを抽出（レイヤ情報は無視）
    data = [
        df_corr1['correlation'].values,
        df_corr2['correlation'].values,
        df_sim1['cosine_similarity'].values,
        df_sim2['cosine_similarity'].values
    ]
    
    labels_list = [
        f'Correlation\n({model_names[0]})',
        f'Correlation\n({model_names[1]})',
        f'Cosine Sim.\n({model_names[0]})',
        f'Cosine Sim.\n({model_names[1]})'
    ]
    
    # 箱ひげ図を作成
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=labels_list, patch_artist=True)
    
    # 箱の色を設定（視認性の良い色の組み合わせ）
    colors = ['#FF7F50', '#FF6347', '#4682B4', '#1E90FF']  # coral, tomato, steelblue, dodgerblue
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 中央値の線を黒色に設定（視認性向上）
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # グリッドを追加
    ax.grid(True, alpha=0.3, axis='y')
    
    # ラベルとタイトルを設定
    ax.set_ylabel("Value", fontsize=12)
    task_title = task_type.capitalize()
    ax.set_title(f"Matched Feature Distribution ({task_title})", fontsize=14, fontweight='bold')
    
    # 統計情報を追加（平均値と中央値）
    for i, (d, label) in enumerate(zip(data, labels_list), 1):
        mean_val = np.mean(d)
        median_val = np.median(d)
        print(f"{label.replace(chr(10), ' ')}: mean={mean_val:.4f}, median={median_val:.4f}, "
              f"min={np.min(d):.4f}, max={np.max(d):.4f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined box plot to {save_path}")

if __name__ == "__main__":

    task_type = "addition"

    # 相関を可視化
    correlation_dir = "images/correlation"
    os.makedirs(correlation_dir, exist_ok=True)
    visualize_correlation(save_path=os.path.join(correlation_dir, f"correlation_{task_type}_act_id1.png"), base="act_id1")
    visualize_correlation(save_path=os.path.join(correlation_dir, f"correlation_{task_type}_act_id2.png"), base="act_id2")

    # # コサイン類似度を可視化
    # cosine_similarity_dir = "images/cosine_similarity"
    # os.makedirs(cosine_similarity_dir, exist_ok=True)
    # visualize_cosine_similarity(
    #     save_path=os.path.join(cosine_similarity_dir, f"cosine_similarity_{task_type}_act_id1.png"),
    #     dataset=task_type, base="act_id1"
    # )
    # visualize_cosine_similarity(
    #     save_path=os.path.join(cosine_similarity_dir, f"cosine_similarity_{task_type}_act_id2.png"),
    #     dataset=task_type, base="act_id2"
    # )
    
    # # 4つの分布をまとめて箱ひげ図で可視化
    # boxplot_dir = "images"
    # os.makedirs(boxplot_dir, exist_ok=True)
    
    # task_type = "addition"
    # df_corr1 = pd.read_parquet(f"images/correlation/correlation_{task_type}_act_id1.parquet")
    # df_corr2 = pd.read_parquet(f"images/correlation/correlation_{task_type}_act_id2.parquet")
    # df_sim1 = pd.read_parquet(f"images/cosine_similarity/cosine_similarity_{task_type}_act_id1.parquet")
    # df_sim2 = pd.read_parquet(f"images/cosine_similarity/cosine_similarity_{task_type}_act_id2.parquet")
    
    # plot_combined_distribution_boxplot(
    #     df_corr1=df_corr1,
    #     df_corr2=df_corr2,
    #     df_sim1=df_sim1,
    #     df_sim2=df_sim2,
    #     task_type=task_type,
    #     save_path=os.path.join(boxplot_dir, f"combined_distribution_{task_type}.png")
    # )
    
    # filter_featuresで得られた特徴ペアのレイヤ対応をヒートマップで可視化
    # from analyze_attribution import filter_features
    
    # filtered_dir = "images/filtered_features"
    # os.makedirs(filtered_dir, exist_ok=True)
    
    # correlation_threshold = 0.99
    # similarity_threshold = 0.99
    
    # # 相関とコサイン類似度のデータを読み込み
    # correlations = pd.read_parquet(f"images/correlation/correlation_{task_type}_act_id1.parquet")
    # similarities = pd.read_parquet(f"images/cosine_similarity/cosine_similarity_{task_type}_act_id1.parquet")
    
    # # filter_featuresで特徴ペアをフィルタリング
    # filtered_features = filter_features(
    #     correlations, 
    #     similarities,
    #     correlation_threshold=correlation_threshold,
    #     similarity_threshold=similarity_threshold
    # )
    
    # # ヒートマップを可視化
    # visualize_filtered_features(
    #     filtered_features=filtered_features,
    #     save_path=os.path.join(filtered_dir, f"filtered_features_{task_type}.png"),
    #     correlation_threshold=correlation_threshold,
    #     similarity_threshold=similarity_threshold
    # )