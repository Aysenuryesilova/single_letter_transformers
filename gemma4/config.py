"""Model configuration for the tiny Gemma-style transformer.

Gemma's distinctive choices (vs Qwen):
  * 5 local (sliding-window) attention layers for every 1 global layer.
  * Local layers use a small RoPE base; global layers use a large one.
  * "Sandwich" normalization: an RMSNorm BEFORE and AFTER each sub-layer.
  * GeGLU (gelu) feed-forward instead of SwiGLU (silu).
  * Input embeddings are scaled by sqrt(hidden_size).
  * Per-Layer Embeddings (a small extra embedding added at every layer).
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
