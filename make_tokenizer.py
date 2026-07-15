import os
from minbpe.basic import BasicTokenizer

# Dosya yollarını güvenli hale getirelim
data_path = os.path.join("data", "muzik_final.txt")
model_save_name = "muzik_512"

# 1. Metni oku
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# 2. Tokenizer nesnesini oluştur ve eğit
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=512) 

# 3. Kaydet
tokenizer.save(model_save_name) 

print(f"BAŞARILI: {model_save_name}.model ve {model_save_name}.vocab oluşturuldu.")