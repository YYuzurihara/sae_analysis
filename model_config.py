from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    release: str
    dir_name: str
    sae_id: str|None=None
    hf_model_name: str|None=None

llama_scope_lxr_32x = lambda layer: ModelConfig(
    model_name="meta-llama/Llama-3.1-8B",
    release="llama_scope_lxr_32x",
    sae_id=f"l{layer}r_32x" if layer is not None else None,
    dir_name="llama3.1-8b"
)

llama_scope_r1_distill = lambda layer: ModelConfig(
    model_name="meta-llama/Llama-3.1-8B",
    release="llama_scope_r1_distill",
    hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    sae_id=f"l{layer}r_800m_slimpajama" if layer is not None else None,
    dir_name="llama3.1-8b-r1-distill"
)