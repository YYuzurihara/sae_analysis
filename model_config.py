from dataclasses import dataclass
from transformer_lens import HookedTransformer
from sae_lens import SAE
from typing import Tuple

@dataclass
class ModelConfig:
    model_name: str
    release: str
    dir_name: str
    device: str
    sae_id: str|None=None
    hf_model_name: str|None=None

llama_scope_lxr_32x = lambda device, layer=None: ModelConfig(
    model_name="meta-llama/Llama-3.1-8B",
    release="llama_scope_lxr_32x",
    sae_id=f"l{layer}r_32x" if layer is not None else None,
    dir_name="llama3.1-8b",
    device=device
)

llama_scope_r1_distill = lambda device, layer=None: ModelConfig(
    model_name="meta-llama/Llama-3.1-8B",
    release="llama_scope_r1_distill",
    hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    sae_id=f"l{layer}r_800m_slimpajama" if layer is not None else None,
    dir_name="llama3.1-8b-r1-distill",
    device=device
)

def get_model_config(config_name: str, device: str, layer: int = None) -> ModelConfig:
    """指定された設定名からModelConfigを取得する"""
    if config_name == "llama_scope_lxr_32x":
        return llama_scope_lxr_32x(device, layer)
    elif config_name == "llama_scope_r1_distill":
        return llama_scope_r1_distill(device, layer)
    else:
        raise ValueError(f"Unknown config name: {config_name}")

def load_model_and_sae(config: ModelConfig) -> Tuple[HookedTransformer, SAE]:
    """ModelConfigを使用してモデルとSAEをロードする"""
    # モデルをロード
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=config.device,
        hf_model_name=config.hf_model_name
    )
    
    # SAEをロード（sae_idが指定されている場合）
    if config.sae_id is None:
        raise ValueError("sae_id must be specified in ModelConfig")
    
    sae, _, _ = SAE.from_pretrained(
        release=config.release,
        sae_id=config.sae_id,
        device=config.device
    )
    
    return model, sae