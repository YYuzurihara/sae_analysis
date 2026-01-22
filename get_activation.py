from prompt_hanoi import get_answer
from prompt_addition import get_answer as get_answer_addition
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from typing import List, Callable
from functools import partial
from itertools import product
from plot_prob import add_labels, load_model
from model_config import llama_scope_lxr_32x, llama_scope_r1_distill, ModelConfig
import argparse
import dotenv
import os
from tqdm import tqdm

def collection_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    encode: Callable[[torch.Tensor], torch.Tensor],
    collections: List[torch.Tensor],
    tail_length: int|None = None,
    decode: Callable[[torch.Tensor], torch.Tensor]|None = None
) -> torch.Tensor:
    feature_acts = encode(acts)
    if tail_length is not None:
        feature_acts = feature_acts[:, -tail_length:, :]
    else:
        feature_acts = feature_acts[:, 1:, :]

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

    input_ids = model.to_tokens(text, prepend_bos=True, truncate=False).to(model.cfg.device) # (1, n)
    target_idx = model.to_tokens(target_output, prepend_bos=False, truncate=False).view(-1).to(model.cfg.device) # (m,)
    m = target_idx.size(0)

    collections = []
    logits = model.run_with_hooks(
        input_ids.to(model.cfg.device),
        return_type="logits",
        fwd_hooks=[
            (sae.cfg.metadata.hook_name,
                # partial(collection_hook, tail_length=m, encode=sae.encode, collections=collections, decode=sae.decode)
                partial(collection_hook, tail_length=m, encode=sae.encode, collections=collections)
            )
        ]
    ) # (1, n, vocab_size)

    act_ids, feature_acts = collections[0], collections[1] # (num_acts,), (b, m, num__acts)

    # logitsからtarget_outputの確率を計算
    logits = logits[:, -m-1:-1, :] # (1, m, vocab_size)
    probs = logits.softmax(dim=-1).view(-1, logits.shape[-1]) # (m, vocab_size)
    target_probs = probs[torch.arange(m), target_idx] # (m,)

    return act_ids.cpu(), feature_acts.cpu(), target_probs.cpu()

def get_act_prob_hanoi(
    layer: int,
    model_config: ModelConfig,
    data_dir: str
) -> None:
    N_DISKS_LIST = [3, 4, 5]
    FUNC_NAME_LIST = ["solve", "hanoi", "solve_hanoi", "tower_of_hanoi", "f"]
    
    # モデルをループの外で一度だけロード
    model, sae = load_model(model_config)
    total = len(N_DISKS_LIST) * len(FUNC_NAME_LIST)
    for n, func_name in tqdm(product(N_DISKS_LIST, FUNC_NAME_LIST), total=total, desc="Processing Hanoi", ncols=100):
        prompt, target_output = get_answer(n, func_name)
        act_ids, feature_acts, target_probs = get_act_prob(model, sae, prompt+target_output, target_output)
        torch.save(
            {
                "layer": layer,
                "n": n,
                "func_name": func_name,
                "act_ids": act_ids.cpu(), # (num_acts,)
                "feature_acts": feature_acts.cpu(), # (1, m, num_acts)
                "target_probs": target_probs.cpu(), # (m,)
                "target_output": target_output,
            },
            os.path.join(data_dir, f"acts_{layer}_{n}_{func_name}.pt")
        )
    
def get_act_prob_addition(
    layer: int,
    model_config: ModelConfig,
    data_dir: str
) -> None:
    OP1_LIST = range(100)
    OP2_LIST = range(100)
    
    # モデルをループの外で一度だけロード
    model, sae = load_model(model_config)
    all_data = []
    
    total = len(OP1_LIST) * len(OP2_LIST)
    for op1, op2 in tqdm(product(OP1_LIST, OP2_LIST), total=total, desc="Processing additions", ncols=100):
        prompt, target_output = get_answer_addition(op1, op2)
        act_ids, feature_acts, target_probs = get_act_prob(model, sae, prompt+target_output, target_output)
        
        all_data.append({
            "op1": op1,
            "op2": op2,
            "act_ids": act_ids,
            "feature_acts": feature_acts,
            "target_probs": target_probs,
            "target_output": target_output,
        })
    
    os.makedirs(data_dir, exist_ok=True)
    # 全データを1ファイルに保存
    torch.save(
        {
            "layer": layer,
            "data": all_data,
        },
        os.path.join(data_dir, f"acts_{layer}_all.pt")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    
    dotenv.load_dotenv()
    data_dir = os.getenv("LLAMA_DIR")
    
    # 進捗バーの出力を抑制（オプション）
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # NUM_LAYERS = 32
    LAYER = args.layer
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model config
    MODEL_CONFIG = llama_scope_lxr_32x(DEVICE, LAYER)
    # MODEL_CONFIG = llama_scope_r1_distill(DEVICE, LAYER)
    
    # get_act_prob_hanoi(LAYER, MODEL_CONFIG, data_dir)
    get_act_prob_addition(LAYER, MODEL_CONFIG, data_dir)