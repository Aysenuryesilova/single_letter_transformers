# qwen3 — Qwen3 dense (the baseline)

The clean modern decoder that the other two folders are variations of.

**Read in this order:** `config.py` → `rms_norm.py` → `rotary.py` → `attention.py`
→ `mlp.py` → `block.py` → `model.py`.

Signature ingredients:
- **Pre-norm** blocks: `x += attn(norm(x)); x += mlp(norm(x))`
- **RMSNorm** (no mean subtraction, no bias)
- **QK-Norm**: RMSNorm on per-head queries/keys, then **RoPE**
- **GQA**: fewer key/value heads than query heads
- **SwiGLU** MLP, all linears bias-free, tied input/output embeddings

Run: `python3 train.py` then `python3 generate.py 20`.
