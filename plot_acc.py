from prompt_hanoi import get_answer, POS_TO_START_SOLVE
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import os
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer
from typing import List, Tuple

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def load_model(layer: int|None=None) -> tuple[HookedTransformer, SAE|None]:
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device=device,
        dtype=torch.bfloat16, # bf16で推論,これが実行時間の面で非常に重要
    )
    if layer is not None:
        sae = SAE.from_pretrained(
            release=f"llama_scope_lxr_32x",
            sae_id=f"l{layer}r_32x",
            device=device
        )
    else:
        sae = None
    return model, sae

def add_labels(pred_text: str, tokenizer: PreTrainedTokenizer) -> List[str]:
    # 各トークンIDに対応するラベルをリスト化する
    labels = []
    
    n = -1
    for line in pred_text.split("\n"):
        if "CALL" in line:
            n = int(line.split("(")[1].split(",")[0])
            token_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            # print(tokenizer.decode(token_ids))
            [labels.append(f"n{n} call") for _ in token_ids]

        elif "move" in line:
            token_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            # print(tokenizer.decode(token_ids))
            n = int(line.split("move")[-1].split(" ")[1])
            [labels.append(f"n{n} move") for _ in token_ids]

        elif "RETURN" in line:
            token_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            # print(tokenizer.decode(token_ids))
            [labels.append(f"n{n} return") for _ in token_ids]
            n += 1
    return labels

@torch.no_grad()
def get_accuracy(
    model: HookedTransformer,
    sae: SAE|None,
    text: str
) -> Tuple[torch.Tensor, str]:

    if sae is not None:
        sae.eval()
    model.eval()

    # tokenize
    input_ids = model.to_tokens(text, prepend_bos=True, truncate=False)[:, :-1] # (1, n)
    target_idx = input_ids[:, POS_TO_START_SOLVE:].view(-1) # (m,)
    m = target_idx.size(0)

    logits = model.run_with_hooks(input_ids, return_type="logits") # (1, n, vocab_size)
    logits = logits[:, POS_TO_START_SOLVE-1:-1, :] # (1, m, vocab_size)
    probs = logits.softmax(dim=-1).view(-1, logits.shape[-1]) # (m, vocab_size)
    target_probs = probs[torch.arange(m), target_idx] # (m,)
    return target_probs.float().cpu(), "".join(model.to_str_tokens(target_idx))

def plot_accuracy(accuracy: torch.Tensor, data_dir: str, labels: List[str]) -> None:
    plt.figure(figsize=(14, 6))
    
    # 連続する同じラベルをグループ化し、背景色を追加
    i = 0
    colors = plt.cm.Pastel1.colors  # 淡い色のパレット
    color_idx = 0

    while i < len(labels):
        start = i
        current_label = labels[i]
        while i < len(labels) and labels[i] == current_label:
            i += 1
        
        # 背景に色付きスパンを追加
        plt.axvspan(start, i-1, alpha=0.3, color=colors[color_idx % len(colors)])
        
        # グループの中心にラベルを配置
        center = (start + i - 1) / 2
        plt.text(center, plt.ylim()[1] * 0.4, current_label, 
                rotation=90, ha='center', va='center', fontsize=8)
        
        color_idx += 1
    
    plt.plot(accuracy, color='black', linewidth=1.5)
    plt.xlabel("Position")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Position for N={N_DISKS}")
    plt.tight_layout()
    plt.savefig(f"{data_dir}/accuracy.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    N_DISKS = 5

    data_dir = f"images/accuracy/N{N_DISKS}"
    os.makedirs(data_dir, exist_ok=True)

    model, sae = load_model()
    print("loaded model and sae")
    text = get_answer(N_DISKS)
    accuracy, target_text = get_accuracy(model, sae, text)
    labels = add_labels(target_text, model.tokenizer)
    plot_accuracy(accuracy, data_dir, labels)