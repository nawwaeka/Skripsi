# Aplikasi Analisis Kualitas Udara (ISPU) — Flask

Aplikasi web untuk analisis kualitas udara DKI Jakarta: upload data → EDA →
penanganan missing value & outlier → transformasi → association rule mining
(FP-Growth) → output hasil.

## Struktur Halaman

1. **Upload & Preview Data** (`/`) — unggah file ISPU + Meteorologi, lihat preview tabel gabungan.
2. **Visualisasi & EDA** (`/eda`) — statistik deskriptif, missing value, boxplot, korelasi — per stasiun.
3. **Analisis Data** (`/analisis`) — seluruh tahap preprocessing + FP-Growth ditampilkan step-by-step.
4. **Output Program** (`/output`) — rules asosiasi terpilih + tombol download CSV.

## Cara Menjalankan

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi
python app.py
```

Buka browser ke: **http://127.0.0.1:5000**

## Format File yang Diharapkan

- **Data ISPU**: kolom `periode_data`, `bulan`, `tanggal`, `stasiun`, parameter polutan
  (`pm_sepuluh`, `pm_duakomalima`, `sulfur_dioksida`, `karbon_monoksida`, `ozon`,
  `nitrogen_dioksida`), dan kolom `kategori` (label ISPU harian).
- **Data Meteorologi**: kolom `periode_data`, `tanggal`, `stasiun_pengamatan`,
  `temp_air`, `hum_air`, `ff`, `rain` (data per jam, akan diagregasi harian otomatis).

Mendukung format `.xls` (termasuk file HTML-table lama), `.xlsx`, dan `.csv`.

## Catatan Penting

- **Penyimpanan data bersifat in-memory (single user)** — cocok untuk demo/skripsi.
  Kalau aplikasi di-restart, data yang sudah diupload akan hilang dan perlu upload ulang.
  Untuk multi-user sungguhan, `STORE` di `app.py` perlu diganti ke penyimpanan per-session
  (misalnya Redis atau database).
- Ambang batas outlier (`BATAS_DOMAIN` di `pipeline.py`) dan batas kategori ISPU
  (`kategori_ispu` di `pipeline.py`) bisa disesuaikan sesuai kebutuhan/referensi yang dipakai.
- Deteksi outlier menjalankan Isolation Forest untuk **setiap kolom × setiap stasiun**,
  jadi proses di halaman "Analisis Data" bisa memakan waktu beberapa detik saat pertama kali
  dibuka (hasil kemudian di-cache di memori sampai reset/upload ulang).

## Struktur File

```
flask_ispu/
├── app.py              # Routing Flask (4 halaman)
├── pipeline.py          # Seluruh logika pemrosesan data & plotting
├── requirements.txt
├── templates/
│   ├── base.html         # Layout + sidebar navigasi
│   ├── upload.html        # Halaman 1
│   ├── eda.html            # Halaman 2
│   ├── analysis.html        # Halaman 3
│   └── output.html           # Halaman 4
├── static/css/style.css
└── uploads/              # (kosong, disiapkan untuk keperluan lain bila perlu)
```
