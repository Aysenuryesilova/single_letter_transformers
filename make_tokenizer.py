from minbpe.basic import BasicTokenizer

# 1. Metni oku
with open(r"data/muzik_final.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Tokenizer nesnesini oluştur
tokenizer = BasicTokenizer()

# 3. Eğitimi tetikle
tokenizer.train(text, vocab_size=512) 

# 4. Kaydet
tokenizer.save("muzik_512") 
print("BPE başarıyla eğitildi! muzik_512.model ve muzik_512.vocab oluşturuldu.")