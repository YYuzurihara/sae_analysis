from prompt_hanoi import get_answer
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List
from plot_acc import get_accuracy, add_labels, load_model

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

if __name__ == "__main__":
    n_list = [3, 4, 5, 6]
    func_name_list = [
        "solve",
        "hanoi",
        "solve_hanoi",
        "tower_of_hanoi",
        "hanoi_recursive",
        "hanoi_steps",
        "solve_hanoi_recursive",
    ]

    model, sae = load_model()
    df = pd.DataFrame(columns=["label", "accuracy"])
    for n in n_list:
        for func_name in func_name_list:
            prompt, target_output = get_answer(n, func_name=func_name)
            accuracy = get_accuracy(model, sae, prompt+target_output, target_output)
            labels = add_labels(target_output, model.tokenizer, n)
            df = pd.concat([df, pd.DataFrame({"label": labels, "accuracy": accuracy})], ignore_index=True)
    
    # ラベルごとのaccuracyの箱ひげ図を作成
    labels = df["label"].unique()
    plt.figure(figsize=(12, 6))
    data_to_plot = [df[df["label"] == label]["accuracy"].values for label in labels]
    plt.boxplot(data_to_plot, tick_labels=labels, whis=2.0)
    plt.xlabel("Label")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Distribution by Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"images/accuracy/aggregate_acc.png")
    plt.close()