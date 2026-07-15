import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from lora import LoRAConfig
from inject import inject
from base_model import model_class, base_checkpoint

# LoRA eğitimi için temel ayarlar
model_name = "qwen3" # İstediğin modeli buraya yaz
model = model_class(model_name)(cfg=None) 
ckpt = torch.load(base_checkpoint(model_name), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])

# Adaptör enjeksiyonu
lcfg = LoRAConfig(r=4, alpha=8.0)
inject(model, lcfg, method="lora")

print("LoRA adaptörü başarıyla yüklendi ve eğitime hazır!")