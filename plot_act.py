import pandas as pd
import matplotlib.pyplot as plt
import ast

def load_act(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, keep_default_na=False)
    # colums = [token, label, layer_n]
    return df

def _plot_act(
    df: pd.DataFrame,
    fig_path: str,
    layer: int,
    label: str
    ) -> None:

    if label == "None": return

    # 同じlabelの中で反応した特徴量のインデックスのリストを取得
    _df = df[df["label"] == label]
    ids_list = []
    for lst in _df[f"layer_{layer}"]:
        if lst == "": continue
        ids_list.extend([int(x) for x in ast.literal_eval(lst)])
    
    ids = pd.DataFrame(ids_list, columns=["id"])
    counts = ids["id"].value_counts()
    plt.figure()
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), counts.index, rotation=90)
    
    plt.xlabel("Feature ID")
    plt.ylabel("Counts")
    plt.title(f"Feature Activation Counts for label '{label}' in layer {layer}")
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=150)
    plt.close()

    return

def plot_act(
    df: pd.DataFrame,
    fig_dir: str,
    layers: list[int]
    ) -> None:
    for layer in layers:
        for label in df["label"].unique():
            label_formatted = label.replace(" ", "-")
            fig_path = f'{fig_dir}/L{layer}_{label_formatted}.png'
            _plot_act(df, fig_path, layer, label)
    return

if __name__ == "__main__":
    N_DISKS = 4
    csv_path = f'csv/act_ids_hanoi{N_DISKS}.csv'
    fig_dir = f'images/accuracy/N{N_DISKS}'
    layers = [8,16,24]

    df = load_act(csv_path)
    print(df["label"].unique())
    plot_act(df, fig_dir, layers)