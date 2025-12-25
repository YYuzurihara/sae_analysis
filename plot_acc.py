from prompt_hanoi import get_answer, POS_TO_START_SOLVE
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import os
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def load_model(layer: int|None=None) -> tuple[HookedTransformer, SAE|None]:
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device=device,
        dtype=torch.bfloat16  # bf16で推論,これが実行時間の面で非常に重要
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

@torch.no_grad()
def get_accuracy(
    model: HookedTransformer,
    sae: SAE|None,
    text: str
) -> torch.Tensor:

    if sae is not None:
        sae.eval()
    model.eval()

    # tokenize
    input_ids = model.to_tokens(text, prepend_bos=True)[:, :-1] # (1, n)
    target_idx = input_ids[:, POS_TO_START_SOLVE:].view(-1) # (m,)
    m = target_idx.size(0)

    logits = model.run_with_hooks(input_ids, return_type="logits") # (1, n, vocab_size)
    logits = logits[:, POS_TO_START_SOLVE-1:-1, :] # (1, m, vocab_size)
    probs = logits.softmax(dim=-1).view(-1, logits.shape[-1]) # (m, vocab_size)
    target_probs = probs[torch.arange(m), target_idx] # (m,)
    return target_probs.float().cpu()

def plot_accuracy(accuracy: torch.Tensor, data_dir: str) -> None:
    plt.plot(accuracy)
    plt.xlabel("Position")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Position for N={N_DISKS}")
    plt.savefig(f"{data_dir}/accuracy.png")
    plt.close()

if __name__ == "__main__":
    N_DISKS = 7

    data_dir = f"images/accuracy/N{N_DISKS}"
    os.makedirs(data_dir, exist_ok=True)

    model, sae = load_model()
    print("loaded model and sae")
    text = get_answer(N_DISKS)
    accuracy = get_accuracy(model, sae, text)
    plot_accuracy(accuracy, data_dir)