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
from model_config import get_model_config
from plot_prob import load_model

def collection_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    encode: Callable[[torch.Tensor], torch.Tensor],
    collections: List[List[torch.Tensor]]
) -> torch.Tensor:
    feature_acts = encode(acts) # shape: [b, n, enc_dim]
    feature_acts = feature_acts[:, 1:, :] # <bos>を削除
    
    # バッチ内の各サンプルごとに活性化した特徴を収集
    batch_size = feature_acts.shape[0]
    for b in range(batch_size):
        # 各サンプルの活性化特徴を取得
        sample_acts = feature_acts[b]  # shape: [n, enc_dim]
        _, act_feat_ids = torch.where(sample_acts > 0)
        # 重複を削除
        act_feat_ids = torch.unique(act_feat_ids)
        collections.append(act_feat_ids)
    
    return acts

def collect_activated_features(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
) -> List[torch.Tensor]:
    collections: List[List[torch.Tensor]] = []
    
    model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[
            (sae.cfg.metadata.hook_name, partial(
                collection_hook, encode=sae.encode, collections=collections
            ))
        ]
    )
    
    # 活性化した特徴IDのリストを返す (バッチサイズ分)
    return collections

@torch.no_grad()
def get_activated_features_batch(
    model: HookedTransformer,
    sae: SAE,
    prompts: List[str],
    target_outputs: List[str],
) -> List[torch.Tensor]:
    """
    バッチで指定されたプロンプトと目標出力に対して活性化する特徴IDを取得する
    
    Args:
        prompts: プロンプトのリスト
        target_outputs: ターゲット出力のリスト
    
    Returns:
        act_feat_ids_list: 各サンプルの活性化した特徴IDのリスト (CPU)
    """
    sae.eval()
    model.eval()

    # tokenize
    texts = [p + t for p, t in zip(prompts, target_outputs)]
    input_ids = model.to_tokens(texts, prepend_bos=True) # (batch_size, max_len)

    # 活性化した特徴IDを収集
    act_feat_ids_list = collect_activated_features(model, sae, input_ids)
    
    # CPUに移動
    return [feat_ids.cpu() for feat_ids in act_feat_ids_list]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()
    TARGET_LAYER = args.layer
    BATCH_SIZE = args.batch_size
    # MODEL_NAME = "llama"
    MODEL_NAME = "distill"

    dotenv.load_dotenv()
    # ここでllama, distillを決める
    if MODEL_NAME == "llama":
        data_dir = os.getenv("LLAMA_DIR")
    elif MODEL_NAME == "distill":
        data_dir = os.getenv("LLAMA_DISTILL_DIR")
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")
    os.makedirs(data_dir, exist_ok=True)

    # ModelConfigを取得してモデルとSAEをロード
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(MODEL_NAME, device, layer=TARGET_LAYER)
    model, sae = load_model(config)
    print(f"Loaded model and SAE for layer {TARGET_LAYER}")
    print(f"Using batch size: {BATCH_SIZE}")

    # 全ての層の特徴idを収集
    all_data = []
    
    # op1, op2のペアを生成
    all_pairs = list(product(range(100), range(100)))
    
    # バッチごとに処理
    for batch_start in tqdm(
        range(0, len(all_pairs), BATCH_SIZE),
        total=(len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc=f"Layer {TARGET_LAYER}"
    ):
        batch_end = min(batch_start + BATCH_SIZE, len(all_pairs))
        batch_pairs = all_pairs[batch_start:batch_end]
        
        # バッチのプロンプトとターゲット出力を生成
        prompts = []
        target_outputs = []
        for op1, op2 in batch_pairs:
            prompt, target_output = get_answer(op1, op2)
            prompts.append(prompt)
            target_outputs.append(target_output)
        
        # バッチで活性化した特徴IDを取得
        batch_act_feat_ids = get_activated_features_batch(
            model=model,
            sae=sae,
            prompts=prompts,
            target_outputs=target_outputs,
        )
        
        # 各サンプルの結果を処理
        for (op1, op2), act_feat_ids in zip(batch_pairs, batch_act_feat_ids):
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
