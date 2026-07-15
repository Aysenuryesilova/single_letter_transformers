"""Model configuration for the tiny Qwen3-style transformer.

Everything the model needs to know about its own shape lives in this one
dataclass. The defaults are deliberately tiny so training runs on a CPU in
seconds while still using the real Qwen3 dense recipe.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 512        # Hepsi 512 olmalı!
    hidden_size: int = 64        # CPU için güvenli bir orta değer
    num_layers: int = 2          
    num_heads: int = 4           
    num_kv_heads: int = 2        
    head_dim: int = 16           
    intermediate_size: int = 128 
    max_seq_len: int = 64        # Tokenizer'ın uzunluk kapasitesi
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
