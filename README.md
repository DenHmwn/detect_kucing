# Detect Kucing

Aplikasi **Detect Kucing** adalah model berbasis **Computer Vision dan Convolutional Neural Network (CNN)** yang digunakan untuk **mendeteksi apakah gambar berisi kucing atau bukan kucing**. Model ini dilatih menggunakan dataset berlabel dengan jumlah gambar:

## Datasets
-  **10.000 gambar kucing (positif)**
-  **40.000 gambar non-kucing (negatif)**
```text
Datasets Tidak di Ungggah ke dalam Repository karna ukuran file yang terlalu besae
```

Proyek ini dibuat untuk tugas klasifikasi biner dan dapat digunakan sebagai dasar untuk aplikasi deteksi real-time atau integrasi API/web.

Proyek ini merupakan **monorepo** yang terdiri dari:

- **Model Training** → file `train.py`  
- **Prediksi Gambar** → file `predict.py`
- **Dependencies** → `requirements.txt`  
- **Dataset & Model Output** → folder `dataset/`, `models/`

---

##  Fitur Utama

- Klasifikasi gambar **kucing vs non-kucing**  
-  CNN (Convolutional Neural Network) untuk feature learning  
-  Preprocessing data (resize, normalisasi, augmentasi)  
-  Training model dengan dataset besar  
-  Evaluasi akurasi & loss  
-  Prediksi dari gambar baru

---

## Struktur Folder
``` text
detect_kucing/
├── dataset/
│ ├── train/
│ │ ├── cat/
│ │ └── non_cat/
│ ├── test/
│ │ ├── cat/
│ │ └── non_cat/
├── models/
│ └── best_model.h5 (hasil training)
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
└── README.md
```

---

##  Penjelasan File Utama

### `train.py`

Script untuk:

- memuat dataset dari folder `dataset/train/`
- preprocessing gambar (resize & normalisasi)
- membangun arsitektur CNN
- melatih model
- menyimpan model terbaik ke `models/`

### `predict.py`

Script untuk:

- memuat model yang sudah dilatih
- melakukan prediksi pada file gambar baru
- menampilkan hasil prediksi (“cat” atau “non-cat”)

Cara Menggunakan:

```bash
python predict.py --image path_ke_gambar.jpg
```

---

## Instalasi Atau Cara Running

### Clone Repo
```bash
git clone https://github.com/DenHmwn/detect_kucing.git
cd detect_kucing
```

### Install Depedencies
```bash
pip install -r requirements.txt
```

### Training Model
```bash
python train.py
```

### Prediksi Image
```bash
Prediksi : CAT
Confidence : 0.87
```

---

## Author
```text
Deni Himawan
```

---

## Lisensi
```text
MIT
```

