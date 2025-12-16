import os
from typing import Any
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from huggingface_hub import login
from transformers.generation.continuous_batching.continuous_api import ContinuousBatchingManager
from prompt_hanoi import get_answer, POS_TO_START_SOLVE # pyright: ignore[reportUnusedImport]
import torch.nn.functional as F
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

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
    活性化した特徴量を収集する
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
    feature_ids: torch.Tensor,
    diff: torch.Tensor
) -> torch.Tensor:

    # Normalize feature ids to 1D tensor so we can run a batch where each
    # element ablates a potentially different feature at the same position.
    batch_size = feature_ids.numel()
    prompts = [text] * batch_size

    def ablation_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # resid shape: [batch, pos, d_model]
        ablated_feature = sae.encode(resid) # shape: [batch, pos, enc_dim]

        batch_indices = torch.arange(resid.shape[0], device=resid.device)
        ablated_feature[batch_indices, pos, feature_ids.to(resid.device)] = 0

        ablated_feature = sae.decode(ablated_feature) # shape: [batch, pos, d_model]
        resid = ablated_feature + diff
        return resid

    hook_name:str = sae.cfg.metadata.hook_name

    fwd_hooks = [
        (hook_name, ablation_hook),
    ]

    with model.hooks(fwd_hooks=fwd_hooks):  # pyright: ignore[reportArgumentType]
        logits, _ = model.run_with_cache(prompts, return_cache_object=False, prepend_bos=True)
    
    return logits

def get_ce_loss(
    logits: torch.Tensor, 
    target_ids: list[int],
) -> torch.Tensor:

    # logits: [batch, seq_len, vocab_size] -> [batch, vocab_size, seq_len]に変換
    logits = logits.transpose(1, 2) # shape: [batch, vocab_size, seq_len]

    target_tensor = torch.tensor(target_ids, device=logits.device) # shape: [seq_len]
    target_tensor = target_tensor.repeat(logits.shape[0], 1) # shape: [batch, seq_len]

    loss = F.cross_entropy(logits, target_tensor, reduction="none") # shape: [batch, seq_len]
    return loss

if __name__ == "__main__":
    # Parameters
    TARGET_LAYER = 8 # 指定した層
    CHECK_POINT = 2 # 特徴量を収集する位置
    BATCH_SIZE = 4 # バッチサイズ

    data_dir = os.getenv("DATA_DIR")
    os.makedirs(f"{data_dir}/L{TARGET_LAYER}", exist_ok=True)
    os.makedirs(f"{data_dir}/L{TARGET_LAYER}/baseline", exist_ok=True)

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
    target_ids = token_ids[POS_TO_START_SOLVE:-1]

    # 活性化した特徴量を収集する
    baseline_logits, collected_features, diff = collect_active_features(model, sae, text)

    # 特徴量を操作していないときのloss, diffを保存
    ce_loss = get_ce_loss(baseline_logits[:, POS_TO_START_SOLVE:, :], target_ids)
    # diff: [batch, seq_len, hidden_dim]
    reconstruction_loss = diff.norm(p=2, dim=2).mean()
    torch.save(reconstruction_loss, f"{data_dir}/L{TARGET_LAYER}/baseline/reconstruction_loss.pt")
    torch.save(ce_loss, f"{data_dir}/L{TARGET_LAYER}/baseline/ce_loss.pt")
    print(f"Saved baseline ce_loss and reconstruction_loss")

    for pos, feature_ids in enumerate(tqdm(collected_features[CHECK_POINT:], desc="Positions"), start=CHECK_POINT):
        os.makedirs(f"{data_dir}/L{TARGET_LAYER}/position{pos}", exist_ok=True)
        batch_feature_ids_tuple = torch.split(feature_ids, BATCH_SIZE) # batch分のタプルに分割

        for batch_feature_ids in batch_feature_ids_tuple:
            logits = inspect_logits(model, sae, text, pos, batch_feature_ids, diff)
            # print(f"Logits shape: {logits.shape}")

            # loss
            logits_to_save = logits[:, POS_TO_START_SOLVE:, :]
            loss = get_ce_loss(logits_to_save, target_ids)
            for i, feature_id in enumerate(batch_feature_ids):
                torch.save(loss[i], f"{data_dir}/L{TARGET_LAYER}/position{pos}/feature{feature_id.item()}.pt")
            
            del logits, logits_to_save, loss
            torch.cuda.empty_cache()
