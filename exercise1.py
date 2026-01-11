from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("merhaba")
print("bilgisayara iyi ve kotu mesajlari ogretiyorum")

cumleler = [
    "iyi gunler",
    "bugun hava guzel",
    "yazilidan 100 aldim",
    "bugun hava kotu",
    "grip oldum",
    "yazilidan 0 aldim"
]

etiketler = [
    0,
    0,
    0,
    1,
    1,
    1
]
donusturucu = CountVectorizer()
X = donusturucu.fit_transform(cumleler)

zeka = MultinomialNB()
zeka.fit(X, etiketler)

print("bilgisayar ogrendi")

mesaj = input("bir mesaj yaz")

mesaj_sayi = donusturucu.transform([mesaj])
sonuc = zeka.predict(mesaj_sayi)

if sonuc[0] == 0:
  print("bu mesaj guzel")
else:
  print("bu mesaj kotu")
