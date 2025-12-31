from prompt_hanoi import get_answer, POS_TO_START_SOLVE
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from typing import List, Dict, Callable
from functools import partial
import pandas as pd
from plot_acc import add_labels

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def load_model(layers: List[int]) -> List[tuple[HookedTransformer, SAE|None]]:
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device=device,
        dtype=torch.bfloat16, # bf16で推論,これが実行時間の面で非常に重要
    )
    saes = []
    for layer in layers:
        sae = SAE.from_pretrained(
            release=f"llama_scope_lxr_32x",
            sae_id=f"l{layer}r_32x",
                device=device
            )
        saes.append(sae)
    return model, saes

def ids_collection_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    encode: Callable[[torch.Tensor], torch.Tensor],
    act_ids: List[torch.Tensor]
) -> torch.Tensor:
    feature_acts = encode(acts)

    # collect activated features indices
    _, act_pos_ids, act_feat_ids = torch.where(feature_acts > 0)

    act_ids.append(torch.stack([act_pos_ids, act_feat_ids], dim=1))
    return acts

def get_act(model: HookedTransformer, sae: SAE, text: str) -> pd.DataFrame:
    act_ids = []
    tokens = model.to_tokens(text, prepend_bos=True)
    tokens = tokens.to(device)
    model.run_with_hooks(
        tokens,
        return_type="loss",
        loss_per_token=True,
        fwd_hooks=[
            (sae.cfg.metadata.hook_name,
                partial(ids_collection_hook, encode=sae.encode, act_ids=act_ids)
            )
        ]
    )

    # ラベルを付与
    target_idx = tokens[:, POS_TO_START_SOLVE:].view(-1)
    target_text = "".join(model.to_str_tokens(target_idx))
    target_labels = ["None"]*(POS_TO_START_SOLVE-1) + add_labels(target_text, model.tokenizer)
    # act_idsを展開する
    act_ids = act_ids[0].view(-1, 2) # (n, 2), nはactivated featureの数

    # dfを定義して埋めていく
    df = pd.DataFrame(index=range(tokens.shape[1]-1), columns=["token", "label", "act_ids"])
    df["token"] = [model.tokenizer.decode(token, skip_special_tokens=True) for token in tokens[0,1:]]
    # 各トークンごとに対応するact_idsを抽出してリストで格納する
    for i in range(df.index.max()):
        df.at[i, "act_ids"] = act_ids[act_ids[:, 0] == i][:, 1].tolist()
    df["label"] = target_labels
    return df

if __name__ == "__main__":
    layers = [8,16,24]
    model, saes = load_model(layers)
    n = 3
    text = get_answer(n)

    df_total = pd.DataFrame(columns=["token", "label"] + [f"layer_{layer}" for layer in layers])
    for layer, sae in zip(layers, saes):
        df = get_act(model, sae, text)
        if df_total["token"].isnull().all():
            df_total["token"] = df["token"]
        df_total[f"layer_{layer}"] = df["act_ids"]
        df_total["label"] = df["label"]
    df_total.to_csv(f"act_ids_hanoi{n}.csv", index=False)