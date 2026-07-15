from minbpe.basic import BasicTokenizer

# 1. Metni oku
with open(r"data\temiz_muzik.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Tokenizer nesnesini oluştur
tokenizer = BasicTokenizer()

# 3. ÖNEMLİ: Eğitimi tetikle (vocab_size burada çok önemli)
# Metindeki karakterleri anlamlı gruplara ayırması için 512 veya 1024 yap
tokenizer.train(text, vocab_size=512) 

# 4. Kaydet
tokenizer.save("muzik_bpe") # Bu isimle kaydedince .model ve .vocab oluşacak
print("BPE başarıyla eğitildi ve kurallar oluşturuldu!")