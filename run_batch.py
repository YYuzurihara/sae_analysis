from typing import Callable, List, Tuple
from prompt_hanoi import get_answer
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import os
import dotenv
from functools import partial
from tqdm import tqdm
import time
import argparse
import pandas as pd

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

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
    _, _, act_feat_ids = torch.where(feature_acts > 0)
    # act_feat_idsの重複を削除
    act_feat_ids = torch.unique(act_feat_ids)

    collections.append(diff)
    collections.append(act_feat_ids)
    return acts

# ablate_feature_idsは[pos, feature_id]の形で与える
def ablation_hook(
    acts: torch.Tensor, # shape: [b, n, d_model]
    hook: HookPoint,
    ablate_feat_ids: torch.Tensor, # shape: [b]
    diff: torch.Tensor, # shape: [n, d_model],
    encode: Callable[[torch.Tensor], torch.Tensor],
    decode: Callable[[torch.Tensor], torch.Tensor],
    pos_start_abl: int=2
) -> torch.Tensor:
    features = encode(acts) # shape: [b, n, enc_dim]
    
    # ablateする対象の軸(b, n, enc_dim)を取得
    abl_batch_ids = torch.arange(ablate_feat_ids.shape[0], device=acts.device) # b - GPUテンソルを明示的に作成
    features[abl_batch_ids, pos_start_abl:, ablate_feat_ids] = 0

    # 復元
    acts_restored = decode(features) + diff # shape: [b, n, d_model]
    return acts_restored

def collect(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
) -> List[torch.Tensor]:
    collections: List[torch.Tensor] = []
    
    # 一番最後のトークンから予測されるトークンには答えがないので長さが1短くなる
    base_ce_loss = model.run_with_hooks(
        tokens,
        return_type="loss",
        loss_per_token=True,
        fwd_hooks=[
            (sae.cfg.metadata.hook_name, partial(collection_hook, encode=sae.encode, decode=sae.decode, collections=collections))
        ]
    )
    return collections

@torch.no_grad()
def run_batch(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    target_output: str,
    batch_size: int,
    pos_start_abl: int=2
) -> Tuple[torch.Tensor, torch.Tensor]:

    sae.eval()
    model.eval()

    # tokenize
    text = prompt + target_output
    input_ids = model.to_tokens(text, prepend_bos=True) # (1, n)
    target_ids = model.to_tokens(target_output, prepend_bos=False).view(-1) # (m,)
    m = target_ids.shape[0]


    # collect diff
    print("run_batch: collecting diff")
    start_time = time.time()
    collections = collect(model, sae, input_ids)
    end_time = time.time()
    print(f"run_batch: collecting diff took {end_time - start_time} sec")

    # split collections into diff, act_feat_ids
    diff = collections[0]
    act_feat_ids = collections[1] # [num_features]

    # split activated feature ids into batch_size groups
    batches = act_feat_ids.split(batch_size) # [b] * num_batches
    
    # ablation
    accuracy = []
    batch_inputs = input_ids.repeat(batch_size, 1) # [b, seq_len]
    for feat_ids in tqdm(batches):
        proc_batch_size = feat_ids.shape[0]
        logits = model.run_with_hooks(
            batch_inputs[:proc_batch_size, :], # splitしたバッチのサイズを超えないように末尾をスライス
            return_type="logits",
            loss_per_token=True,
            fwd_hooks=[
                (
                    sae.cfg.metadata.hook_name,
                    partial(
                        ablation_hook,
                        ablate_feat_ids=feat_ids,
                        diff=diff,
                        encode=sae.encode,
                        decode=sae.decode,
                        pos_start_abl=pos_start_abl
                    )
                )
            ]
        )
        logits = logits[:, -m-1:-1, :] # (b, m, vocab_size)
        probs = logits.softmax(dim=-1) # (b, m, vocab_size)
        target_probs = probs[:, torch.arange(m), target_ids] # (b, m)
        acc = target_probs.float().cpu()
        # accuracyに対応するアブレーション結果を追加
        accuracy.append(acc)
    return torch.cat(accuracy, dim=0).cpu(), act_feat_ids.cpu()

def save_results(accuracy: torch.Tensor, act_feat_ids: torch.Tensor, data_dir: str):
    # accuracy: (f, m), act_feat_ids: (f,)
    # DataFrameを作成: feature_id, accuracy のカラムを持つ
    df = pd.DataFrame({
        "feature_id": act_feat_ids.numpy(),
        "accuracy": list(accuracy.numpy())  # 各行に(m,)のaccuracy配列を格納
    })
    
    # CSVとして保存
    save_path = os.path.join(data_dir, "results.csv")
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pos_start_abl", type=int, default=2)
    args = parser.parse_args()
    TARGET_LAYER = args.layer
    BATCH_SIZE = args.batch_size
    N_DISKS = args.n

    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    data_dir = f"{data_dir}/L{TARGET_LAYER}/N{N_DISKS}"
    os.makedirs(data_dir, exist_ok=True)

    model, sae = load_model_and_sae(layer=TARGET_LAYER)
    print("loaded model and sae")
    prompt, target_output = get_answer(N_DISKS)
    accuracy, act_feat_ids = run_batch(
        model, sae, prompt, target_output, batch_size=BATCH_SIZE)
    save_results(accuracy, act_feat_ids, data_dir)