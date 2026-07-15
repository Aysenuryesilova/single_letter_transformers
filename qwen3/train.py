import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from minbpe import BasicTokenizer
from config import ModelConfig
from model import TinyQwen 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BasicTokenizer()
tokenizer.load(r"C:\Users\aysenur\Desktop\MAGİBU YAPAY ZEKA MİMARİSİ\single_letter_transformers\muzik_1024.model")

text = open(os.path.join(os.path.dirname(__file__), "..", "data", "muzik_final.txt"), "r", encoding="utf-8").read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
BLOCK_SIZE, BATCH_SIZE = 16, 64

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    return torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(DEVICE), torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix]).to(DEVICE)

model = TinyQwen(ModelConfig(vocab_size=1024)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
for step in range(1, 5001):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 500 == 0: print(f"Qwen3 Adım {step}: Loss {loss.item():.4f}")

torch.save({"model": model.state_dict(), "cfg": model.cfg}, "tiny_qwen_muzik_v2.pt")