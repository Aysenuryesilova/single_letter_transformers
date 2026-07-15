
import sys
import os
import torch

# Ana dizini ve alt klasörleri sisteme ekle
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "qwen3")) # TinyQwen için

from minbpe import BasicTokenizer
from config import ModelConfig
from model import TinyQwen # TinyGemma için bunu değiştirip tekrar çalıştırabilirsin

# Ayarlar
DEVICE = "cpu" # Test için CPU güvenlidir
MODEL_PATH = "tiny_qwen_muzik.pt"

# Yükleme
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = TinyQwen(ckpt["cfg"]).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = BasicTokenizer()
tokenizer.load("muzik_256.model")

# Üretim
def generate_names(n=5):
    start = torch.zeros((n, 1), dtype=torch.long, device=DEVICE)
    out = model.generate(start, max_new_tokens=20)
    for row in out.tolist():
        print(tokenizer.decode(row))

print("Modelin ürettiği sonuçlar:")
generate_names()