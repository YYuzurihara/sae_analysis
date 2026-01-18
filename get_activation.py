from prompt_hanoi import get_answer
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from typing import List, Callable
from functools import partial
from itertools import product
from plot_prob import add_labels, load_model
from model_config import llama_scope_lxr_32x, llama_scope_r1_distill
import argparse
import dotenv
import os

def collection_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    encode: Callable[[torch.Tensor], torch.Tensor],
    tail_length: int,
    collections: List[torch.Tensor],
    decode: Callable[[torch.Tensor], torch.Tensor]|None = None
) -> torch.Tensor:
    feature_acts = encode(acts)
    feature_acts = feature_acts[:, -tail_length:, :]
    if decode is not None:
        target_acts = decode(feature_acts)
        mse = torch.nn.functional.mse_loss(target_acts, acts[:, -tail_length:, :], reduction="mean")
        print(f"MSE: {mse.item()}")

    # collect activated features indices
    _, _, act_feat_ids = torch.where(feature_acts > 0)
    act_feat_ids = act_feat_ids.unique()

    collections.append(act_feat_ids)
    collections.append(feature_acts[:, :, act_feat_ids])
    return acts

@torch.no_grad()
def get_act_prob(
    model: HookedTransformer,
    sae: SAE,
    text: str,
    target_output: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    input_ids = model.to_tokens(text, prepend_bos=True, truncate=False) # (1, n)
    target_idx = model.to_tokens(target_output, prepend_bos=False, truncate=False).view(-1) # (m,)
    m = target_idx.size(0)

    collections = []
    logits = model.run_with_hooks(
        input_ids,
        return_type="logits",
        fwd_hooks=[
            (sae.cfg.metadata.hook_name,
                partial(collection_hook, tail_length=m, encode=sae.encode, collections=collections, decode=sae.decode)
            )
        ]
    ) # (1, n, vocab_size)

    act_ids, feature_acts = collections[0], collections[1] # (num_acts,), (b, m, num__acts)

    # logitsからtarget_outputの確率を計算
    logits = logits[:, -m-1:-1, :] # (1, m, vocab_size)
    probs = logits.softmax(dim=-1).view(-1, logits.shape[-1]) # (m, vocab_size)
    target_probs = probs[torch.arange(m), target_idx] # (m,)

    return act_ids, feature_acts, target_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    args = parser.parse_args()
    
    dotenv.load_dotenv()
    data_dir = os.getenv("LLAMA_DISTILL_DIR")

    # NUM_LAYERS = 32
    LAYER = args.layer
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"

    # model config
    # MODEL_CONFIG = llama_scope_lxr_32x(DEVICE, LAYER)
    MODEL_CONFIG = llama_scope_r1_distill(DEVICE, LAYER)
    
    N_DISKS_LIST = [3, 4, 5]
    FUNC_NAME_LIST = ["solve", "hanoi", "solve_hanoi", "tower_of_hanoi", "f"]
    
    skip_flag = False
    for n, func_name in product(N_DISKS_LIST, FUNC_NAME_LIST):
        print(f"n: {n}, func_name: {func_name}")

        if not skip_flag:
            model, sae = load_model(MODEL_CONFIG)
            skip_flag = True
        prompt, target_output = get_answer(n, func_name)
        act_ids, feature_acts, target_probs = get_act_prob(model, sae, prompt+target_output, target_output)
        torch.save(
            {
                "layer": LAYER,
                "n": n,
                "func_name": func_name,
                "act_ids": act_ids, # (num_acts,)
                "feature_acts": feature_acts, # (1, m, num_acts)
                "target_probs": target_probs, # (m,)
                "target_output": target_output,
            },
            os.path.join(data_dir, f"acts_{LAYER}_{n}_{func_name}.pt")
        )