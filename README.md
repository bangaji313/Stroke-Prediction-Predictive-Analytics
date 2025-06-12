# Proyek Predictive Analytics: Prediksi Risiko Stroke

Repository ini berisi proyek *machine learning* untuk memprediksi risiko stroke pada pasien berdasarkan berbagai atribut kesehatan. Proyek ini merupakan bagian dari submission pada modul **Machine Learning Terapan** dalam program **Coding Camp 2025 by DBS Foundation**.

| Keterangan | Informasi |
| :--- | :--- |
| **Nama Lengkap** | Maulana Seno Aji Yudhantara |
| **Cohort ID** | MC117D5Y1789 |
| **Mentor** | Yeftha Joshua Ezekiel |

---

## 📝 Deskripsi Proyek

Tujuan dari proyek ini adalah membangun sebuah model klasifikasi yang dapat mengidentifikasi individu dengan risiko tinggi terkena stroke. Dengan prediksi dini, diharapkan tindakan pencegahan dapat dilakukan untuk mengurangi angka kejadian dan kecacatan akibat stroke.

## 🚀 Latar Belakang

Stroke adalah penyebab kematian dan kecacatan utama di seluruh dunia. Program **Coding Camp 2025 by DBS Foundation** berkolaborasi dengan **Dicoding** memberikan kesempatan untuk mengaplikasikan ilmu *machine learning* dalam menyelesaikan masalah nyata, salah satunya di domain kesehatan. Proyek ini adalah implementasi praktis dari materi yang telah dipelajari untuk menciptakan solusi berbasis data.

## 📁 Struktur Repository

Repository ini terdiri dari beberapa file dan folder utama:

- `[PredictiveAnalytics]_Submission_1_MLT_MaulanaSenoAjiYudhantara.ipynb`: Notebook Jupyter yang berisi seluruh proses analisis dan pemodelan secara detail.
- `Laporan_Proyek_Predictive_Analytics_MaulanaSenoAjiYudhantara.md`: Laporan lengkap dalam format Markdown yang menjelaskan setiap tahapan proyek.
- `stroke_prediction_script.py`: Skrip Python bersih yang mengimplementasikan alur kerja dari model terbaik.
- `healthcare-dataset-stroke-data.csv`: Dataset mentah yang digunakan dalam proyek ini.
- `images/`: Folder yang berisi semua gambar dan visualisasi yang digunakan dalam laporan.
- `requirements.txt`: Daftar library Python yang dibutuhkan untuk menjalankan proyek ini.

## 🛠️ Alur Kerja Proyek

Proyek ini dikerjakan dengan mengikuti alur kerja *data science* standar:
1.  **Data Understanding**: Memahami karakteristik dataset dan melakukan analisis data eksplorasi (EDA) untuk menemukan wawasan awal.
2.  **Data Preparation**: Membersihkan data, menangani *missing values*, melakukan *encoding* dan *scaling*, serta menyeimbangkan data menggunakan SMOTE.
3.  **Modeling**: Membangun dan melatih beberapa model klasifikasi, termasuk Logistic Regression, KNN, Random Forest, dan SVM.
4.  **Evaluation**: Mengevaluasi model menggunakan metrik yang relevan (terutama Recall) dan memilih model terbaik.

## 📊 Hasil

Model **Logistic Regression** dipilih sebagai model akhir terbaik karena memiliki nilai **Recall tertinggi (0.80)**. Dalam konteks medis, kemampuan untuk mendeteksi sebanyak mungkin kasus positif (meminimalkan *false negative*) lebih diutamakan daripada metrik lainnya.

## 📄 Laporan Lengkap

Untuk analisis yang lebih mendalam mengenai setiap tahapan, justifikasi, dan visualisasi, silakan merujuk ke laporan lengkap proyek:
- **[Laporan_Proyek_Predictive_Analytics_MaulanaSenoAjiYudhantara.md](Laporan_Proyek_Predictive_Analytics_MaulanaSenoAjiYudhantara.md)**

## 🚀 Cara Menjalankan Proyek

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/bangaji313/Stroke-Prediction-Predictive-Analytics.git
    cd Stroke-Prediction-Predictive-Analytics
    ```

2.  **(Opsional tapi direkomendasikan) Buat virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Notebook atau Skrip:**
    - Untuk melihat analisis detail, buka dan jalankan file `.ipynb` menggunakan Jupyter Notebook atau Jupyter Lab.
    - Untuk menjalankan pipeline model terbaik secara langsung, eksekusi skrip Python:
      ```bash
      python stroke_prediction_script.py
      ```