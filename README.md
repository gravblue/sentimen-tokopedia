# 🛍️ Analisis Sentimen Ulasan Aplikasi Tokopedia

Proyek ini melakukan analisis sentimen (positif, netral, negatif) terhadap ulasan pengguna aplikasi Tokopedia di Google Play Store menggunakan pendekatan Machine Learning (SVM, Random Forest, Logistic Regression) dan Deep Learning (GRU dengan Word2Vec).

## 📖 Ringkasan

Data ulasan diambil langsung dari Google Play Store menggunakan library `google-play-scraper`, kemudian diberi label sentimen otomatis menggunakan pendekatan lexicon based (kamus kata positif dan negatif berbahasa Indonesia). Setelah itu, data dibersihkan dan diproses melalui tahapan text preprocessing standar (case folding, cleaning, tokenizing, stopword removal, dan stemming dengan Sastrawi) sebelum digunakan untuk melatih empat model klasifikasi.

## 📁 Struktur Proyek

```
.
├── scrapping.ipynb          # 🕸️ Scraping ulasan dari Google Play Store
├── pelatihan_model.ipynb    # 🧠 Preprocessing, pelabelan, EDA, dan pelatihan model
├── inference.ipynb          # 🔍 Uji coba prediksi dengan model yang sudah dilatih
├── tokopedia_reviews.csv    # 📊 Dataset hasil scraping ulasan Tokopedia
├── requirements.txt         # 📦 Daftar dependency Python
└── README.md
```

Catatan: notebook ini dibuat dan dijalankan di Google Colab. File model hasil pelatihan (misalnya `gru_model.h5`, `svm_model.pkl`, `random_forest_model.pkl`, `logistic_regression_model.pkl`, `tfidf_vectorizer.pkl`, `tokenizer.json`) dihasilkan saat runtime dan tidak disertakan langsung di repo. Jalankan `pelatihan_model.ipynb` untuk menghasilkan file tersebut, atau unduh dari rilis/Google Drive jika tersedia.

## 🔄 Alur Kerja

1. **🕸️ Scraping (`scrapping.ipynb`)**
   - Mengambil ulasan aplikasi Tokopedia (`com.tokopedia.tkpd`) dari Google Play Store menggunakan `google-play-scraper`.
   - Menyimpan hasil ke `tokopedia_reviews.csv`.

2. **🧠 Pelatihan Model (`pelatihan_model.ipynb`)**
   - **Preprocessing Text**: cleaning, case folding, tokenizing, stopword removal (NLTK), dan stemming (Sastrawi).
   - **Pelabelan**: pelabelan otomatis sentimen (positif/negatif/netral) menggunakan lexicon kata opini berbahasa Indonesia.
   - **Eksplorasi Data**: visualisasi distribusi label dan word cloud.
   - **Ekstraksi Fitur**: TF-IDF Vectorizer untuk model machine learning, dan Word2Vec untuk model GRU.
   - **Pelatihan Model**: melatih 4 model klasifikasi (SVM, Random Forest, Logistic Regression, GRU).
   - **Penyimpanan Model**: menyimpan model dan artefak (vectorizer, tokenizer) ke file.

3. **🔍 Inference (`inference.ipynb`)**
   - Memuat kembali model dan artefak yang sudah disimpan.
   - Melakukan prediksi sentimen terhadap contoh ulasan baru menggunakan keempat model sekaligus.

## 📦 Dependency

Library utama yang digunakan dalam proyek ini:

```
pandas
numpy
scikit-learn
tensorflow
gensim
nltk
Sastrawi
wordcloud
joblib
matplotlib
google-play-scraper
requests
tqdm
```

## 🚀 Cara Penggunaan

1. Jalankan `scrapping.ipynb` untuk mengumpulkan data ulasan terbaru dari Google Play Store, atau langsung gunakan `tokopedia_reviews.csv` yang sudah disediakan di repo ini.
2. Jalankan `pelatihan_model.ipynb` secara berurutan dari atas ke bawah untuk melakukan preprocessing, pelabelan, pelatihan, dan penyimpanan model. Notebook ini akan menghasilkan file berikut:
   - `svm_model.pkl`
   - `random_forest_model.pkl`
   - `logistic_regression_model.pkl`
   - `gru_model.h5`
   - `tfidf_vectorizer.pkl`
   - `tokenizer.json`
3. Jalankan `inference.ipynb` untuk mencoba prediksi sentimen pada teks ulasan baru. Pastikan seluruh file model dan artefak di atas berada di folder yang sama dengan notebook.

Contoh output dari `inference.ipynb`:

```
Review 1: Barang sesuai deskripsi, pengiriman cepat!
GRU Model           : positive
SVM Classifier      : positive
Random Forest       : positive
Logistic Regression : positive

Review 2: Pelayanan toko sangat buruk, saya kecewa.
GRU Model           : negative
SVM Classifier      : negative
Random Forest       : negative
Logistic Regression : negative
```

## 📈 Hasil Model

Perbandingan akurasi pada data uji:

| Model                | Akurasi Test |
|-----------------------|:------------:|
| SVM (TF-IDF)           | 92.68%       |
| Random Forest (TF-IDF) | 86.47%       |
| Logistic Regression (TF-IDF) | 88.91% |
| GRU (Word2Vec)         | 92.59%       |

## ✅ Kesimpulan

- Jika fokus utama adalah akurasi maksimum dengan representasi TF-IDF, SVM menjadi pilihan terbaik.
- Jika konteks dan urutan kata lebih penting, terutama pada teks panjang atau kompleks, GRU dengan Word2Vec lebih sesuai meskipun akurasinya sedikit lebih rendah.
- Logistic Regression dan Random Forest masih layak digunakan, tetapi performanya lebih rendah dibanding SVM dan GRU pada dataset ini.
