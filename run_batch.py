from typing import Callable, List
from prompt_hanoi import get_answer, POS_TO_START_SOLVE
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import os
import dotenv
from functools import partial
from tqdm import tqdm
import time

dotenv.load_dotenv()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")
data_dir = os.getenv("DATA_DIR")

def load_model_and_sae(layer: int):
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device=device,
        dtype=torch.bfloat16  # bf16で推論,これが実行時間の面で非常に重要
    )
    sae = SAE.from_pretrained(
        release=f"llama_scope_lxr_32x",
        sae_id=f"l{layer}r_32x",
        device=device
    )

    print("=============debug================")
    print(f"normalize_activations: {sae.cfg.normalize_activations}")
    print(f"dtype: {sae.cfg.dtype}")
    print(f"W_enc norm: {sae.W_enc.norm().item()}")
    print(f"W_dec norm: {sae.W_dec.norm().item()}")
    print("=================================")
    return model, sae

def collection_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    encode: Callable[[torch.Tensor], torch.Tensor],
    decode: Callable[[torch.Tensor], torch.Tensor],
    collections: List[torch.Tensor]
) -> torch.Tensor:
    feature_acts = encode(acts)
    sae_out = decode(feature_acts)

    # collect diff
    diff = acts - sae_out
    # collect activated features indices
    _, act_pos_ids, act_feat_ids = torch.where(feature_acts > 0)

    collections.append(diff)
    collections.append(act_pos_ids)
    collections.append(act_feat_ids)
    return acts

# ablate_feature_idsは[pos, feature_id]の形で与える
def ablation_hook(
    acts: torch.Tensor, # shape: [b, n, d_model]
    hook: HookPoint,
    ablate_ids: torch.Tensor, # shape: [b, 2]
    diff: torch.Tensor, # shape: [n, d_model],
    encode: Callable[[torch.Tensor], torch.Tensor],
    decode: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    features = encode(acts) # shape: [b, n, enc_dim]
    
    # ablateする対象の軸(b, n, enc_dim)を取得
    abl_batch_ids = torch.arange(ablate_ids.shape[0], device=acts.device) # b - GPUテンソルを明示的に作成
    abl_pos_ids = ablate_ids[:, 0] # n
    abl_feat_ids = ablate_ids[:, 1] # enc_dim

    # ablation実行
    features[abl_batch_ids, abl_pos_ids, abl_feat_ids] = 0

    # 復元
    acts_restored = decode(features) + diff # shape: [b, n, d_model]
    return acts_restored

def collect(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
) -> torch.Tensor:
    collections = []
    
    base_ce_loss = model.run_with_hooks(
        tokens,
        return_type="loss",
        loss_per_token=True,
        fwd_hooks=[
            (sae.cfg.metadata.hook_name, partial(collection_hook, encode=sae.encode, decode=sae.decode, collections=collections))
        ]
    )
    return base_ce_loss[:, POS_TO_START_SOLVE:], collections

@torch.no_grad()
def run_batch(
    model: HookedTransformer,
    sae: SAE,
    text: str,
    batch_size: int,
    pos_start_abl: int
) -> None:

    sae.eval()
    model.eval()

    # tokenize
    input_ids = model.to_tokens(text)[:, :-1] # (1, n)

    # collect diff
    print("run_batch: collecting diff")
    start_time = time.time()
    base_ce_loss, collections = collect(model, sae, input_ids)
    end_time = time.time()
    torch.save(base_ce_loss, f"{data_dir}/L{TARGET_LAYER}/base_ce_loss.pt")
    print(f"run_batch: collecting diff took {end_time - start_time} sec, loss: {base_ce_loss.mean().item()}")

    # split collections into diff, act_pos_ids, act_feat_ids
    diff, act_pos_ids, act_feat_ids = collections

    # collect reconstruction loss
    reconstruction_loss = (diff[:, 1:, :]**2).sum(dim=-1) # BOSトークンは無視する
    print(f"reconstruction_loss: {reconstruction_loss}")
    torch.save(reconstruction_loss, f"{data_dir}/L{TARGET_LAYER}/reconstruction_loss.pt")

    act_pos_ids = act_pos_ids[pos_start_abl:]
    act_feat_ids = act_feat_ids[pos_start_abl:]
    act_ids = torch.stack([act_pos_ids, act_feat_ids], dim=1) # (b, 2)

    # split activated feature ids into batch_size groups
    batches = act_ids.split(batch_size)
    
    # ablation
    batch_inputs = input_ids.repeat(batch_size, 1)
    for i,ids in tqdm(enumerate(batches)):
        ce_loss = model.run_with_hooks(
            batch_inputs[:ids.shape[0], :], # 指定したバッチサイズを超えないように末尾をスライス
            return_type="loss",
            loss_per_token=True,
            fwd_hooks=[
                (
                    sae.cfg.metadata.hook_name,
                    partial(ablation_hook, ablate_ids=ids, diff=diff, encode=sae.encode, decode=sae.decode)
                )
            ]
        )
        ce_loss = torch.cat([ids, ce_loss[:, POS_TO_START_SOLVE:]], dim=1).cpu() # (b, 2), (b, 1) -> (b, 3)
        torch.save(ce_loss, f"{data_dir}/L{TARGET_LAYER}/ce_loss_batch{i}.pt")
    return

if __name__ == "__main__":
    TARGET_LAYER = 16

    os.makedirs(f"{data_dir}/L{TARGET_LAYER}", exist_ok=True)

    model, sae = load_model_and_sae(layer=TARGET_LAYER)
    print("loaded model and sae")
    text = get_answer()
    run_batch(model, sae, text, batch_size=128, pos_start_abl=2)
