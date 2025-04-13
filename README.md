# Real-Time SIBI Detection with YOLO and Deep Learning Models

Proyek ini merupakan penerapan _YOLO_ (You Only Look Once) untuk deteksi gerakan tangan dan jari dalam Bahasa Isyarat Indonesia (SIBI) secara **real-time** menggunakan webcam. Sistem ini akan menampilkan bounding box dan alfabet hasil deteksi beserta skor confidence-nya.

## 🔍 Fitur Utama

- Deteksi real-time gerakan jari/tangan untuk SIBI menggunakan webcam.
- Menggunakan YOLOv8 untuk deteksi bounding box tangan.
- Tersedia 4 model _image classification_ yang dapat dipilih:
  - **DenseNet121** (📈 _89.882%_ akurasi) – ⭐ _Recommended_
  - EfficientNet-B0 (89.089%)
  - ResNet18 (89.228%)
  - MobileNetV3 Small (85.459%)
- Opsi untuk training ulang model dengan parameter yang dapat dikonfigurasi.
- Jalankan sistem secara langsung tanpa perlu training ulang.

## 📦 Dataset

Dataset diambil dari Kaggle:  
📎 [SIBI Dataset by Alvin Bintang](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset?utm_medium=social&utm_campaign=kaggle-dataset-share&utm_source=twitter)

## 💻 Requirement

- CUDA **12.8** WAJIB digunakan apabila ingin melakukan training model.
- Python 3.9+
- OS: Windows 10/11 (karena menggunakan `.bat` files)

## 🚀 Cara Menjalankan

### Opsi 1: **Training Model Sendiri**

1. **Jalankan `install (train).bat`**

   - Membuat virtual environment
   - Menginstall `requirements.txt`
   - Menginstall PyTorch yang sesuai dengan CUDA 12.8

2. **Jalankan `download.bat`**

   - Mengecek ketersediaan dataset secara lokal
   - Akan otomatis mendownload dari Kaggle jika belum ada

3. **Jalankan `training.bat`**

   - Memulai proses training untuk keempat model
   - Anda akan diminta memasukkan jumlah **epoch** dan **batch size**
     - Default: `epoch=10`, `batch_size=32`

4. **Jalankan `run.bat`**
   - Pilih model hasil training
   - Membuka kamera dan mulai deteksi real-time
   - Tekan `q` untuk keluar

---

### Opsi 2: **Langsung Jalankan Tanpa Training**

1. **Jalankan `install (no train).bat`**

   - Membuat virtual environment
   - Menginstall requirement tanpa PyTorch

2. **Jalankan `run.bat`**
   - Pilih model yang sudah disediakan dalam repository
   - Membuka kamera dan mulai deteksi real-time
   - Tekan `q` untuk keluar

## 📦 Model YOLO (Bounding Box)

Model YOLO yang digunakan untuk mendeteksi tangan:  
🔗 [`hand_yolov8s.pt`](https://huggingface.co/xingren23/comfyflow-models/blob/976de8449674de379b02c144d0b3cfa2b61482f2/ultralytics/bbox/hand_yolov8s.pt)

---

## 📊 Akurasi Model Klasifikasi

| Model             | Akurasi     |
| ----------------- | ----------- |
| **DenseNet121**   | **89.882%** |
| ResNet18          | 89.228%     |
| EfficientNet-B0   | 89.089%     |
| MobileNetV3 Small | 85.459%     |

---

## 📸 Real-Time Demo

> Kamera akan langsung aktif saat `run.bat` dijalankan. Lakukan gestur bahasa isyarat, dan sistem akan menampilkan bounding box dan hasil prediksi huruf.

---

## 🧠 Catatan

- Pastikan Anda menggunakan GPU dengan driver yang kompatibel dengan CUDA 12.8 jika ingin melakukan training.
- Tidak disarankan menggunakan mode training di perangkat dengan CPU-only.

---

## 🤝 Kontribusi

Pull request dan issue sangat terbuka untuk pengembangan lebih lanjut. Jangan ragu untuk menyempurnakan model atau menambahkan fitur baru!

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah MIT License.
