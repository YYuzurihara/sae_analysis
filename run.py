import os
from typing import Any
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from huggingface_hub import login
from prompt_hanoi import PROMPT_HANOI, get_answer # pyright: ignore[reportUnusedImport]
import torch.nn.functional as F
import json
from tqdm import tqdm

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

    hook_name:str = sae.cfg.metadata.hook_name

    fwd_hooks = [
        (hook_name, ablation_hook),
    ]

    with model.hooks(fwd_hooks=fwd_hooks):  # pyright: ignore[reportArgumentType]
        print(f"Running model with cache")
        logits, _ = model.run_with_cache(text, prepend_bos=True)
    
    return logits

def analyze_logits(
    logits: torch.Tensor, 
    target_ids: list[int],
    pos_to_start: int
) -> list[dict[str, Any]]:
    results = []

    # logits: [0, seq_len, vocab_size], target_ids: [seq_len]
    # logitsは推論すべき位置が0になっている
    target_tensor = torch.tensor(target_ids, device=logits.device)

    # loss: [seq_len], top_10_logits: [seq_len, 10]
    loss = F.cross_entropy(logits[0,:,:], target_tensor, reduction="none")
    for i in range(len(target_ids)):
        results.append({
            "loss": loss[i].item(),
            "position": i + pos_to_start,
            "target_id": target_ids[i]
        })
    return results

if __name__ == "__main__":
    # Parameters
    TARGET_LAYER = 24 # 指定した層 (User can change this)
    POS_TO_START = 331 # 推論を開始するトークン位置 -> 331番目のトークン予測から観察する
    CHECK_POINT = 301
    
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
        target_ids = token_ids[POS_TO_START:-1]

        # 1: 活性化した特徴量を収集する
        baseline_logits, collected_features, diff = collect_active_features(model, sae, text)

        # 特徴量を操作していないときのlogitsを保存
        torch.save(baseline_logits[:, POS_TO_START:, :], f"logits/baseline/logits_baseline.pt")
        print(f"Saved logits at position {POS_TO_START} to logits/baseline/logits_baseline.pt")

        # for test
        # test_logits = baseline_logits[0, POS_TO_START, :]
        # top10_logits, top10_indices = torch.topk(test_logits, 10, dim=-1)
        # print("Top-10 token IDs (wo target) at POS_TO_START:")
        # for i in range(10):
        #     token_id = top10_indices[i].item()
        #     token_str = model.to_string([token_id]).strip()
        #     logit = top10_logits[i].item()
        #     print(f"{i+1}: token_id={token_id}, token='{token_str}', logit={logit:.3f}")

        for pos, feature_ids in enumerate(tqdm(collected_features[CHECK_POINT:], desc="Positions"), start=CHECK_POINT):
            tqdm_features = tqdm(feature_ids, desc=f"Features at pos {pos}", leave=False)
            for feature_id in tqdm_features:
                logits = inspect_logits(model, sae, text, pos, feature_id.item(), diff)
                # print(f"Logits shape: {logits.shape}")

                # logitsの保存
                logits_to_save = logits[:, POS_TO_START:, :]
                results = analyze_logits(logits_to_save, target_ids, POS_TO_START)
                os.makedirs(f"logits/position{pos}", exist_ok=True)
                with open(f"logits/position{pos}/logits_pruned_f{feature_id.item()}.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                tqdm_features.set_postfix_str(f"saved f{feature_id.item()}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
