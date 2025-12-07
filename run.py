import os
from typing import Any
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from huggingface_hub import login
from prompt_hanoi import PROMPT_HANOI, get_answer # pyright: ignore[reportUnusedImport]
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

def collect_active_features(
    model: HookedTransformer,
    sae: SAE[Any],
    text: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    1回目の推論: 活性化した特徴量を収集する
    Returns:
        baseline_logits: 1回目の推論のlogits [1, pos, vocab_size]
        collected_features: 各トークン位置で収集された特徴量のインデックスのリスト (リストの長さはseq_len)
        diff: saeでの推論との差分
    """
    hook_name: str = sae.cfg.metadata.hook_name

    # リストで扱うことで関数内部での変更がそのまま保存される
    collected_features: list[torch.Tensor] = []
    diff = []

    def collection_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        with torch.no_grad():
            features = sae.encode(resid)  # shape: [1, seq_len, enc_dim]
            seq_len = features.shape[1]

            for pos in range(seq_len):
                pos_features = features[0, pos, :] # shape: [enc_dim]
                indices = pos_features.nonzero(as_tuple=True)[0] # shape: [num_nonzero]
                collected_features.append(indices.cpu())
        diff.append(resid - sae.decode(features))
        return resid

    with model.hooks(fwd_hooks=[(hook_name, collection_hook)]):
        baseline_logits, _ = model.run_with_cache([text], prepend_bos=True)

    print(f"Collected {len(collected_features)} active features across all positions")
    return baseline_logits, collected_features, diff[0]

def inspect_logits(
    model: HookedTransformer, 
    sae: SAE[Any], 
    text: str,
    pos:int,
    feature_id: int,
    diff: torch.Tensor
) -> torch.Tensor:

    def ablation_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # resid shape: [1, pos, d_model]
        ablated_feature = sae.encode(resid) # shape: [1, pos, enc_dim]
        
        ablated_feature[0, pos, feature_id] = 0
        ablated_feature = sae.decode(ablated_feature) # shape: [1, pos, d_model]
        resid = ablated_feature + diff
        return resid

    print(f"\n--- Logit Inspection ---")
    hook_name:str = sae.cfg.metadata.hook_name

    fwd_hooks = [
        (hook_name, ablation_hook),
    ]

    with model.hooks(fwd_hooks=fwd_hooks):  # pyright: ignore[reportArgumentType]
        print(f"Running model with cache")
        logits, _ = model.run_with_cache(text, prepend_bos=True)
    
    return logits


if __name__ == "__main__":
    # Parameters
    TARGET_LAYER = 16 # 指定した層 (User can change this)
    POS_TO_START = 332 # 推論を開始するトークン位置
    
    try:
        # モデルのロード
        model, sae = load_model_and_sae(layer=TARGET_LAYER)

        # フックの名前を確認
        # hook_names = list(model.hook_dict.keys())
        # print(f"Hook names: {hook_names}")

        text = get_answer()
        tokens = model.to_str_tokens(text, prepend_bos=True)
        token_ids = model.to_tokens(text, prepend_bos=True).tolist()[0]
        # 最後のトークンは"\n"なので不要
        text = ''.join(tokens[:-2])
        # 正解のトークンid
        target_ids = token_ids[1:-1]

        # 1: 活性化した特徴量を収集する
        baseline_logits, collected_features, diff = collect_active_features(model, sae, text)

        # 2: 各特徴量をゼロにして推論を行う
        for pos, feature_ids in enumerate(collected_features):
            for feature_id in feature_ids:
                logits = inspect_logits(model, sae, text, pos, feature_id.item(), diff)
                print(f"Logits shape: {logits.shape}")

                # logits[POS_TO_START]の保存
                logits_to_save = logits[:, POS_TO_START:, :]
                torch.save(logits_to_save, f"logits/logits_p{pos}_f{feature_id.item()}.pt")
                print(f"Saved logits at position {pos} to logits/logits_p{pos}_f{feature_id.item()}.pt")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
