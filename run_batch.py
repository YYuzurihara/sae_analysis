from typing import Callable, List
from prompt_addition import get_answer
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import os
import dotenv
from functools import partial
from itertools import product
from tqdm import tqdm
import argparse
import pandas as pd
from model_config import get_model_config, load_model_and_sae

def collection_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    encode: Callable[[torch.Tensor], torch.Tensor],
    collections: List[torch.Tensor]
) -> torch.Tensor:
    feature_acts = encode(acts) # shape: [b, n, enc_dim]
    feature_acts = feature_acts[:, 1:, :] # <bos>を削除
    
    # collect activated features indices
    _, _, act_feat_ids = torch.where(feature_acts > 0)
    # act_feat_idsの重複を削除
    act_feat_ids = torch.unique(act_feat_ids)
    
    collections.append(act_feat_ids)
    return acts

def collect_activated_features(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
) -> torch.Tensor:
    collections: List[torch.Tensor] = []
    
    model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[
            (sae.cfg.metadata.hook_name, partial(
                collection_hook, encode=sae.encode, collections=collections
            ))
        ]
    )
    
    # 活性化した特徴IDを返す
    return collections[0]

@torch.no_grad()
def get_activated_features(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    target_output: str,
) -> torch.Tensor:
    """
    指定されたプロンプトと目標出力に対して活性化する特徴IDを取得する
    
    Returns:
        act_feat_ids: 活性化した特徴IDのテンソル (CPU)
    """
    sae.eval()
    model.eval()

    # tokenize
    text = prompt + target_output
    input_ids = model.to_tokens(text, prepend_bos=True) # (1, n)

    # 活性化した特徴IDを収集
    act_feat_ids = collect_activated_features(model, sae, input_ids)
    
    return act_feat_ids.cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    TARGET_LAYER = args.layer

    dotenv.load_dotenv()
    # ここでllama, distillを決める
    data_dir = os.getenv("LLAMA_DIR")
    os.makedirs(data_dir, exist_ok=True)

    # ModelConfigを取得してモデルとSAEをロード
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config("llama_scope_lxr_32x", device, layer=TARGET_LAYER)
    model, sae = load_model_and_sae(config)
    print(f"Loaded model and SAE for layer {TARGET_LAYER}")

    # 全ての層の特徴idを収集
    all_data = []
    
    for op1, op2 in tqdm(
        product(range(100), range(100)), total=10000, desc=f"Layer {TARGET_LAYER}"
    ):
        prompt, target_output = get_answer(op1, op2)
        
        # 活性化した特徴IDを取得
        act_feat_ids = get_activated_features(
            model=model,
            sae=sae,
            prompt=prompt,
            target_output=target_output,
        )
        
        # 各特徴IDに対してop1, op2を追加
        for feat_id in act_feat_ids.tolist():
            all_data.append({
                "op1": op1,
                "op2": op2,
                "feature_id": feat_id
            })
    
    # DataFrameを作成
    df = pd.DataFrame(all_data)
    
    # 層ごとに1ファイルとして保存
    save_path = os.path.join(data_dir, f"L{TARGET_LAYER}.parquet")
    df.to_parquet(save_path, index=False, engine='pyarrow')
    print(f"Results saved to {save_path}")
    print(f"Total rows: {len(df)}")
