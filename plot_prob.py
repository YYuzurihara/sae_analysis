from prompt_hanoi import get_answer
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import os
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer
from typing import List
from transformers import AutoModelForCausalLM
from model_config import ModelConfig, llama_scope_lxr_32x, llama_scope_r1_distill

def load_model(
    model_config: ModelConfig,
    skip_model: bool=False,
    skip_sae: bool=False,
    skip_hf_model: bool=False
    ) -> tuple[HookedTransformer, SAE|None]:

    if model_config.hf_model_name is not None and not skip_hf_model:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_model_name,
            dtype=torch.bfloat16
        )
    else:
        hf_model = None

    if not skip_model:
        model = HookedTransformer.from_pretrained(
            model_config.model_name,
            device=model_config.device,
            hf_model=hf_model,
            dtype=torch.bfloat16, # bf16で推論,これが実行時間の面で非常に重要
        )
    else:
        model = None

    if not skip_sae:
        sae = SAE.from_pretrained(
            release=model_config.release,
            sae_id=model_config.sae_id,
            device=model_config.device
        )
    else:
        sae = None
    return model, sae

def add_labels(pred_text: str, tokenizer: PreTrainedTokenizer, n: int) -> List[str]:
    # 各トークンIDに対応するラベルをリスト化する
    labels = []
    
    for line in pred_text.split("\n"):
        if "CALL" in line:
            depth = n - int(line.split("(")[1].split(",")[0])
            token_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            # print(tokenizer.decode(token_ids))
            [labels.append(f"d{depth} call") for _ in token_ids]

        elif "move" in line:
            token_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            # print(tokenizer.decode(token_ids))
            depth = n - int(line.split("move")[-1].split(" ")[1])
            [labels.append(f"d{depth} move") for _ in token_ids]

        elif "RETURN" in line:
            token_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            # print(tokenizer.decode(token_ids))
            [labels.append(f"d{depth} return") for _ in token_ids]
            depth -= 1
        else:
            assert line == "", "Invalid line: " + line
    return labels

@torch.no_grad()
def get_probability(
    model: HookedTransformer,
    sae: SAE|None,
    text: str,
    target_output: str
) -> torch.Tensor:

    if sae is not None:
        sae.eval()
    model.eval()

    # tokenize
    input_ids = model.to_tokens(text, prepend_bos=True, truncate=False) # (1, n)
    target_idx = model.to_tokens(target_output, prepend_bos=False, truncate=False).view(-1) # (m,)
    m = target_idx.size(0)

    print(model.to_string(target_idx))

    logits = model.run_with_hooks(input_ids, return_type="logits") # (1, n, vocab_size)
    logits = logits[:, -m-1:-1, :] # (1, m, vocab_size)
    probs = logits.softmax(dim=-1).view(-1, logits.shape[-1]) # (m, vocab_size)
    target_probs = probs[torch.arange(m), target_idx] # (m,)

    return target_probs.float().cpu()

def plot_probability(probability: torch.Tensor, data_dir: str, labels: List[str]) -> None:
    plt.figure(figsize=(14, 6))
    
    # ラベルの種類に応じた色を定義
    label_colors = {
        'call': '#AEC6CF',      # ライトブルー
        'move': '#C1E1C1',      # ライトグリーン
        'return': '#FFB6A3'     # ライトサーモン
    }
    
    # 連続する同じラベルをグループ化し、背景色を追加
    i = 0

    while i < len(labels):
        start = i
        current_label = labels[i]
        while i < len(labels) and labels[i] == current_label:
            i += 1
        
        # ラベルの種類に応じて色を選択
        color = '#DDDDDD'  # デフォルト色
        for key, value in label_colors.items():
            if key in current_label:
                color = value
                break
        
        # 背景に色付きスパンを追加
        plt.axvspan(start, i-1, alpha=0.3, color=color)
        
        # グループの中心にラベルを配置
        center = (start + i - 1) / 2
        plt.text(center, plt.ylim()[1] * 0.4, current_label, 
                rotation=90, ha='center', va='center', fontsize=8)
    
    plt.plot(probability, color='black', linewidth=1.5)
    plt.xlabel("Sequence Length")
    plt.ylabel("Probability")
    plt.title(f"Probability vs Token Length for N={N_DISKS}")
    plt.tight_layout()
    plt.savefig(f"{data_dir}/probability.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    N_DISKS = 3
    FUNC_NAME = "solve"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_CONFIG = llama_scope_lxr_32x(DEVICE)
    # MODEL_CONFIG = llama_scope_r1_distill(DEVICE)

    data_dir = f"images/probability/{MODEL_CONFIG.dir_name}/N{N_DISKS}"
    os.makedirs(data_dir, exist_ok=True)

    model, sae = load_model(MODEL_CONFIG, skip_sae=True)
    print("loaded model and sae")
    prompt, target_output = get_answer(N_DISKS, func_name=FUNC_NAME)
    probability = get_probability(model, sae, prompt+target_output, target_output)
    labels = add_labels(target_output, model.tokenizer, N_DISKS)
    plot_probability(probability, data_dir, labels)