"""Model configuration for the tiny Qwen3.5-style transformer.

Qwen3.5 is a *hybrid*: most layers are cheap Gated DeltaNet (linear attention),
and every Nth layer is full softmax attention. `full_attn_every` controls that
ratio (Qwen3.5 uses 4 -> 1 full-attention layer for every 3 linear ones).
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
