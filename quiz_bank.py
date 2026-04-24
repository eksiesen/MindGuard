# quiz_bank.py

# Alanlar:
# - id: benzersiz quiz kimliği
# - question/options/answer: içerik
# - tags: NLP/keyword eşleştirme için etiketler (v0.1/sonrası)
# - difficulty: 1 (kolay) ... 3 (zor)
QUIZZES = [
    {
        "id": "tr_gram_001",
        "question": "Aşağıdakilerden hangisi bir isimdir?",
        "options": ["Koşmak", "Kitap", "Hızlı"],
        "answer": "Kitap",
        "tags": ["turkce", "dilbilgisi", "isim"],
        "difficulty": 1,
    },
    {
        "id": "tr_gram_002",
        "question": "Aşağıdakilerden hangisi bir fiildir?",
        "options": ["Sevgi", "Koşmak", "Mavi"],
        "answer": "Koşmak",
        "tags": ["turkce", "dilbilgisi", "fiil"],
        "difficulty": 1,
    },
    {
        "id": "tr_gram_003",
        "question": "Aşağıdakilerden hangisi bir sıfattır?",
        "options": ["Güzel", "Güzellik", "Güzelleşmek"],
        "answer": "Güzel",
        "tags": ["turkce", "dilbilgisi", "sifat"],
        "difficulty": 1,
    },
    {
        "id": "tr_gram_004",
        "question": "“Hızlıca” sözcüğünün türü aşağıdakilerden hangisidir?",
        "options": ["Zarf", "İsim", "Zamir"],
        "answer": "Zarf",
        "tags": ["turkce", "dilbilgisi", "zarf"],
        "difficulty": 2,
    },
    {
        "id": "tr_math_001",
        "question": "12 + 8 işleminin sonucu kaçtır?",
        "options": ["18", "20", "22"],
        "answer": "20",
        "tags": ["matematik", "toplama"],
        "difficulty": 1,
    },
    {
        "id": "tr_math_002",
        "question": "36 / 6 işleminin sonucu kaçtır?",
        "options": ["5", "6", "7"],
        "answer": "6",
        "tags": ["matematik", "bolme"],
        "difficulty": 1,
    },
    {
        "id": "tr_math_003",
        "question": "Bir üçgenin iç açıları toplamı kaç derecedir?",
        "options": ["90", "180", "360"],
        "answer": "180",
        "tags": ["matematik", "geometri"],
        "difficulty": 2,
    },
    {
        "id": "tr_sci_001",
        "question": "Aşağıdakilerden hangisi bir hal değişimidir?",
        "options": ["Donma", "Kırılma", "Paslanma"],
        "answer": "Donma",
        "tags": ["fen", "madde", "hal_degisimleri"],
        "difficulty": 2,
    },
    {
        "id": "tr_sci_002",
        "question": "Bitkiler fotosentez yaparken hangi gazı kullanır?",
        "options": ["Oksijen", "Karbondioksit", "Azot"],
        "answer": "Karbondioksit",
        "tags": ["fen", "biyoloji", "fotosentez"],
        "difficulty": 2,
    },
    {
        "id": "tr_hist_001",
        "question": "Türkiye Cumhuriyeti hangi yıl ilan edilmiştir?",
        "options": ["1919", "1923", "1938"],
        "answer": "1923",
        "tags": ["tarih", "cumhuriyet"],
        "difficulty": 1,
    },
]
