from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 514       
    hidden_size: int = 128       
    num_layers: int = 4          
    num_heads: int = 8           
    num_kv_heads: int = 4
    head_dim: int = 16           
    intermediate_size: int = 256 
    max_seq_len: int = 128       
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6