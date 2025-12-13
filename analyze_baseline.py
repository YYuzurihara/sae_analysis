# baselineのlogitsからaccuracy, lossを計算する

import torch
import torch.nn.functional as F
from prompt_hanoi import get_answer, POS_TO_START_SOLVE
from transformer_lens import HookedTransformer
import json
import os


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# baselineのlogitsを読み込む(1, n, vocab_size)
baseline_logits = torch.load("logits/baseline/logits_baseline.pt").to(device)

# modelを読み込む
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", device=device)

# 正解のトークンを取得
text = get_answer()
target_tokens = model.to_tokens(text)[0, POS_TO_START_SOLVE:-1] # (n)

# baselineのlogitsから, lossを計算する
loss = F.cross_entropy(baseline_logits[0,:,:], target_tokens, reduction="none")

probs = F.softmax(baseline_logits, dim=-1) # (1, n, vocab_size)
correct_probs = probs[0, torch.arange(probs.shape[1]), target_tokens[0]] # (n)
geom_mean = torch.exp(torch.mean(torch.log(correct_probs)))

os.makedirs("results", exist_ok=True)
with open("results/results_baseline.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": geom_mean.item(),
        "losses": {
            f"position_{i}": loss[i].item() for i in range(loss.shape[0])
        }
    }, f, indent=2, ensure_ascii=False)