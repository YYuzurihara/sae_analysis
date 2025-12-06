import os
from typing import Any
import random
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from huggingface_hub import login
from prompt_hanoi import PROMPT_HANOI # pyright: ignore[reportUnusedImport]
import torch.nn.functional as F

# Device setup
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

token = os.getenv("hf_token")
login(token=token)

print(f"Using device: {device}")

def load_model_and_sae(layer: int) -> tuple[HookedTransformer, SAE[Any]]:
    print("Loading Model...")
    model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", device=device)
    
    print(f"Loading SAE for layer {layer}...")
    # Using llama_scope_lxr_32x release for residual stream
    # ID format seems to be l{layer}r_32x based on yaml inspection
    sae_release = "llama_scope_lxr_32x" 
    sae_id = f"l{layer}r_32x"
    
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device
    )
    
    return model, sae

def inspect_logits(
    model: HookedTransformer, 
    sae: SAE[Any], 
    text: str, 
    target_ids: list[int], 
    flag_steer: bool = True,
    ans_labels: torch.Tensor|None = None
) -> None:
    """
    3. llmのlogitを確認できるようにする
       (And optionally projected SAE features)
    """
    def steering_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # resid shape: [batch, pos, d_model]
        steer_feature = sae.encode(resid)
        diff = resid - sae.decode(steer_feature)

        nonzero_indices = (steer_feature[:, -1, :] != 0).nonzero(as_tuple=True)[1]  # get feature dim indices at last position
        if len(nonzero_indices) > 0:
            feature_idx = random.choice(nonzero_indices.tolist())
            print(f"feature_idx: {feature_idx} is set to zero")
            steer_feature[:, -1, feature_idx] = 0
        resid = sae.decode(steer_feature) + diff
        return resid

    print(f"\n--- Logit Inspection ---")
    hook_name:str = sae.cfg.metadata.hook_name

    if flag_steer:
        fwd_hooks = [
            (hook_name, steering_hook),
        ]
    else:
        fwd_hooks = []

    with model.hooks(fwd_hooks=fwd_hooks):  # pyright: ignore[reportArgumentType]
        logits, _ = model.run_with_cache(text, prepend_bos=True)
    
    # Logits shape: [batch, pos, vocab_size]
    # Look at the last token's logits (prediction for next token)
    last_token_logits = logits[..., -1, :] # type: ignore

    if ans_labels is not None:
        loss = F.cross_entropy(last_token_logits, ans_labels)
        print(f"Loss: {loss.item()}")

    if target_ids != []:
        for target_id in target_ids:
            target_logit = last_token_logits[:, target_id]
            print(f"{model.to_string([target_id])}: {target_logit.item():.4f}")
    
    top_logits, top_indices = torch.topk(last_token_logits, k=10)
    top_logits = top_logits.view(-1)
    top_indices = top_indices.view(-1)
    
    print("\nModel Top Predictions (Next Token):")
    for score, idx in zip(top_logits, top_indices):
        token_str = model.to_string(idx)
        print(f"  '{token_str}': {score.item():.4f}")

if __name__ == "__main__":
    # Parameters
    TARGET_LAYER = 16 # 指定した層 (User can change this)
    
    try:
        # モデルのロード
        model, sae = load_model_and_sae(layer=TARGET_LAYER)

        # フックの名前を確認
        # hook_names = list(model.hook_dict.keys())
        # print(f"Hook names: {hook_names}")

        inspect_logits(model, sae, PROMPT_HANOI, target_ids=[], flag_steer=False)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
