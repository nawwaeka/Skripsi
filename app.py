""""
app.py
======
"""


import io
import math
import os

import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, jsonify
)

import pipeline as pl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "ganti-dengan-secret-key-anda-sendiri"  # wajib diganti untuk produksi

# Penyimpanan hasil pemrosesan (in-memory, single user)
STORE = {
    "df_ispu_mentah": None,        # hasil baca file ISPU, belum diproses/digabung
    "df_meteo_mentah": None,       # hasil baca file Meteorologi, belum diproses/digabung
    "sudah_gabung": False,         # sudah klik "Gabungkan Data"?
    "gabungan_perstasiun": None,   # data hasil gabung (sebelum missing/outlier ditangani)
    "daftar_stasiun": None,
    "dataset_dedup": None,         # setelah duplikasi data ditangani
    "tabel_duplikasi": None,
    "dataset_missing_ok": None,    # setelah missing value ditangani
    "tabel_missing": None,
    "dataset_bersih": None,        # setelah outlier ditangani
    "tabel_deteksi_outlier": None,
    "tabel_validasi_outlier": None,
    "dataset_transformasi": None,  # setelah diskritisasi
    "hasil_fpgrowth": None,
}

PREVIEW_PER_PAGE = 15


def data_belum_diupload():
    return STORE["df_ispu_mentah"] is None


def data_belum_digabung():
    return STORE["gabungan_perstasiun"] is None


def df_to_html_table(df, max_rows=15):
    if df is None or df.empty:
        return "<p class='text-muted fst-italic'>Tidak ada data.</p>"
    return df.head(max_rows).to_html(
        classes="table table-sm table-striped table-hover align-middle",
        border=0, index=False, na_rep="-"
    )


# ============================================================
# HALAMAN 1 — UPLOAD & PREVIEW DATA
# ============================================================

def _daftar_stasiun_mentah():
    """Union nama stasiun (selain 'TIDAK DIKETAHUI') dari file ISPU & Meteo mentah."""
    if data_belum_diupload():
        return []
    s_ispu = set(STORE["df_ispu_mentah"]["stasiun"].unique())
    s_meteo = set(STORE["df_meteo_mentah"]["stasiun"].unique())
    return sorted((s_ispu | s_meteo) - {"TIDAK DIKETAHUI"})


@app.route("/", methods=["GET"])
def index():
    # Preview data TIDAK dirender di server lagi (dulu ini yang bikin load
    # halaman berat karena semua stasiun langsung digabung jadi tabel HTML
    # sekaligus). Sekarang halaman hanya kirim daftar stasiun; data per
    # stasiun + per halaman diambil belakangan lewat /api/preview via fetch(),
    # jadi hanya 15 baris yang diproses & dikirim setiap kali.
    #
    # Alur juga dipecah 2 tahap: upload (baca file saja, ringan) -> preview
    # data mentah -> tombol "Gabungkan Data" (baru di sini proses berat:
    # parsing periode, resample harian, merge) dijalankan.
    uploaded = not data_belum_diupload()
    sudah_gabung = uploaded and not data_belum_digabung()
    return render_template(
        "upload.html",
        uploaded=uploaded,
        sudah_gabung=sudah_gabung,
        daftar_stasiun=STORE["daftar_stasiun"] if sudah_gabung else _daftar_stasiun_mentah(),
    )


@app.route("/upload", methods=["POST"])
def upload():
    file_ispu = request.files.get("file_ispu")
    file_meteo = request.files.get("file_meteo")

    if not file_ispu or file_ispu.filename == "" or not file_meteo or file_meteo.filename == "":
        flash("Mohon unggah KEDUA file: data ISPU dan data Meteorologi.", "danger")
        return redirect(url_for("index"))

    try:
        # TAHAP RINGAN SAJA: baca file + rapikan kolom + standarisasi nama
        # stasiun. Belum ada parsing tanggal/resample/merge, jadi cepat.
        df_ispu_mentah, df_meteo_mentah = pl.baca_mentah(file_ispu, file_meteo)

        # Reset seluruh STORE, lalu isi ulang dari awal
        for key in STORE:
            STORE[key] = None
        STORE["sudah_gabung"] = False
        STORE["df_ispu_mentah"] = df_ispu_mentah
        STORE["df_meteo_mentah"] = df_meteo_mentah

        session["uploaded"] = True
        flash("Berhasil! File terbaca. Silakan cek preview data mentah, "
              "lalu klik \"Gabungkan Data\" untuk melanjutkan.", "success")
    except Exception as e:
        flash(f"Gagal membaca file: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/gabungkan", methods=["POST"])
def gabungkan():
    if data_belum_diupload():
        flash("Silakan unggah data terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    try:
        # TAHAP BERAT: parsing periode, resample harian meteorologi, merge.
        gabungan_perstasiun, daftar_stasiun = pl.gabung_dari_mentah(
            STORE["df_ispu_mentah"], STORE["df_meteo_mentah"]
        )
        if len(daftar_stasiun) == 0:
            flash("Tidak ada stasiun yang berhasil dicocokkan antara data ISPU dan Meteorologi. "
                  "Periksa kembali format file Anda.", "danger")
            return redirect(url_for("index"))

        STORE["gabungan_perstasiun"] = gabungan_perstasiun
        STORE["daftar_stasiun"] = daftar_stasiun
        STORE["sudah_gabung"] = True
        # Reset hasil tahap-tahap selanjutnya (kalau sebelumnya sudah pernah dijalankan)
        STORE["dataset_dedup"] = None
        STORE["tabel_duplikasi"] = None
        STORE["dataset_missing_ok"] = None
        STORE["tabel_missing"] = None
        STORE["dataset_bersih"] = None
        STORE["tabel_deteksi_outlier"] = None
        STORE["tabel_validasi_outlier"] = None
        STORE["dataset_transformasi"] = None
        STORE["hasil_fpgrowth"] = None

        flash(f"Berhasil! {len(daftar_stasiun)} stasiun terdeteksi dan data telah digabungkan.", "success")
    except Exception as e:
        flash(f"Gagal menggabungkan data: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    for key in STORE:
        STORE[key] = None
    STORE["sudah_gabung"] = False
    session.clear()
    flash("Data berhasil direset. Silakan unggah file baru.", "info")
    return redirect(url_for("index"))


# ============================================================
# API — PREVIEW DATA PER STASIUN (PAGINATED)
# ============================================================
# Dipanggil dari JS di upload.html lewat fetch(). Hanya mengembalikan
# potongan (max PREVIEW_PER_PAGE baris) dari SATU stasiun per request,
# supaya tidak perlu build tabel HTML besar untuk 5 stasiun sekaligus.

@app.route("/api/preview/<stasiun>", methods=["GET"])
def api_preview(stasiun):
    if data_belum_diupload():
        return jsonify({"error": "belum_upload"}), 400

    sudah_gabung = not data_belum_digabung()

    if sudah_gabung:
        if stasiun not in STORE["daftar_stasiun"]:
            return jsonify({"error": "stasiun_tidak_ditemukan"}), 404
        df = STORE["gabungan_perstasiun"][stasiun]
    else:
        # Mode preview MENTAH (sebelum "Gabungkan Data" diklik): sumber
        # dipilih lewat query param ?sumber=ispu|meteo, filter manual per
        # stasiun dari df mentah (belum ada resample/merge, jadi ringan).
        sumber = request.args.get("sumber", "ispu")
        df_mentah = STORE["df_meteo_mentah"] if sumber == "meteo" else STORE["df_ispu_mentah"]
        if stasiun not in set(df_mentah["stasiun"].unique()):
            return jsonify({"error": "stasiun_tidak_ditemukan"}), 404
        df = df_mentah[df_mentah["stasiun"] == stasiun].reset_index(drop=True)

    try:
        page = int(request.args.get("page", 1))
    except (TypeError, ValueError):
        page = 1

    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / PREVIEW_PER_PAGE))
    page = min(max(page, 1), total_pages)

    start = (page - 1) * PREVIEW_PER_PAGE
    end = start + PREVIEW_PER_PAGE
    subset = df.iloc[start:end].copy()

    if "periode" in subset.columns:
        subset["periode"] = subset["periode"].dt.strftime("%Y-%m-%d")

    # Konversi NaN -> None supaya valid JSON, lalu jadi list-of-list
    subset = subset.astype(object).where(pd.notnull(subset), None)

    return jsonify({
        "stasiun": stasiun,
        "kolom": list(subset.columns),
        "baris": subset.values.tolist(),
        "page": page,
        "total_pages": total_pages,
        "total_rows": total_rows,
        "per_page": PREVIEW_PER_PAGE,
        "sudah_gabung": sudah_gabung,
    })


# ============================================================
# HALAMAN 2 — VISUALISASI & EDA PER STASIUN
# ============================================================

@app.route("/eda", methods=["GET"])
def eda():
    if data_belum_digabung():
        flash("Silakan unggah data terlebih dahulu di halaman Upload.", "warning")
        return redirect(url_for("index"))

    daftar_stasiun = STORE["daftar_stasiun"]
    stasiun_dipilih = request.args.get("stasiun", daftar_stasiun[0])
    if stasiun_dipilih not in daftar_stasiun:
        stasiun_dipilih = daftar_stasiun[0]

    df = STORE["gabungan_perstasiun"][stasiun_dipilih]

    # Halaman EDA fokus PENUH ke satu stasiun yang dipilih, dan langsung
    # berupa grafik (tanpa kartu "profil singkat"/ringkasan verbal) — tiap
    # grafik dipilih supaya gampang dibaca sekilas, dan urutannya sengaja
    # dibuat "bercerita": komposisi kategori -> polutan mana yang tinggi ->
    # tren dari waktu ke waktu -> hubungan dengan cuaca. Urutan ini juga
    # yang menjembatani ke pertanyaan penelitian di halaman Analisis
    # (pola asosiasi meteorologi & polutan lewat FP-Growth).
    gambar_distribusi_kategori_stasiun = pl.plot_distribusi_kategori_stasiun(df, stasiun_dipilih)
    profil_polutan_stasiun = pl.hitung_profil_polutan_stasiun(df, stasiun_dipilih)
    gambar_tren_variabel_stasiun = pl.plot_tren_variabel_stasiun(df, stasiun_dipilih)
    gambar_perbandingan_tahunan_stasiun = pl.plot_perbandingan_tahunan_stasiun(df, stasiun_dipilih)
    kpi_cuaca_stasiun = pl.hitung_kpi_cuaca_stasiun(df, stasiun_dipilih)

    return render_template(
        "eda.html",
        uploaded=True,
        sudah_gabung=True,
        daftar_stasiun=daftar_stasiun,
        stasiun_dipilih=stasiun_dipilih,
        jumlah_baris=len(df),
        gambar_distribusi_kategori_stasiun=gambar_distribusi_kategori_stasiun,
        profil_polutan_stasiun=profil_polutan_stasiun,
        gambar_tren_variabel_stasiun=gambar_tren_variabel_stasiun,
        gambar_perbandingan_tahunan_stasiun=gambar_perbandingan_tahunan_stasiun,
        kpi_cuaca_stasiun=kpi_cuaca_stasiun,
    )


# ============================================================
# HALAMAN 3 — ANALISIS DATA
# ============================================================

@app.route("/analisis", methods=["GET"])
def analisis():
    if data_belum_digabung():
        flash("Silakan unggah data terlebih dahulu di halaman Upload.", "warning")
        return redirect(url_for("index"))

    daftar_stasiun = STORE["daftar_stasiun"]

    # ---- Pipeline belum dijalankan: tampilkan ringkasan teknis + tombol "ANALISIS SEKARANG" ----
    if STORE["hasil_fpgrowth"] is None:
        ringkasan = pl.ringkasan_praanalisis(STORE["gabungan_perstasiun"], daftar_stasiun)

        deskripsi_praanalisis_html = {
            stasiun: df_to_html_table(tabel, max_rows=20)
            for stasiun, tabel in ringkasan["deskripsi_perstasiun"].items()
        }

        # Visualisasi teknis per stasiun (dipindah dari halaman EDA): sebaran
        # missing value, distribusi/boxplot tiap parameter (untuk screening
        # outlier visual), dan korelasi antar variabel.
        gambar_missing_praanalisis = {
            stasiun: pl.plot_missing_value(STORE["gabungan_perstasiun"][stasiun], stasiun)
            for stasiun in daftar_stasiun
        }
        gambar_outlier_praanalisis = {
            stasiun: pl.plot_screening_outlier(STORE["gabungan_perstasiun"][stasiun], stasiun)
            for stasiun in daftar_stasiun
        }

        return render_template(
            "analysis.html",
            uploaded=True,
        sudah_gabung=True,
            daftar_stasiun=daftar_stasiun,
            sudah_dijalankan=False,
            deskripsi_praanalisis_html=deskripsi_praanalisis_html,
            tabel_missing_praanalisis_html=df_to_html_table(ringkasan["tabel_missing_gabungan"], max_rows=50),
            gambar_missing_praanalisis=gambar_missing_praanalisis,
            gambar_outlier_praanalisis=gambar_outlier_praanalisis,
        )

    # ---- Pipeline sudah dijalankan: tampilkan ringkasan sebelum/sesudah ----
    fp = STORE["hasil_fpgrowth"]

    # Contoh preview transformasi/kategorisasi (stasiun pertama)
    stasiun_contoh = daftar_stasiun[0]
    kolom_kategori = [c for c in STORE["dataset_transformasi"][stasiun_contoh].columns if c.startswith("kategori_")]
    preview_transformasi_html = df_to_html_table(
        STORE["dataset_transformasi"][stasiun_contoh][["periode"] + kolom_kategori], max_rows=8
    )

    # ---- Ringkasan Duplikasi Data: sebelum vs sesudah ----
    tabel_duplikasi_ringkas = STORE["tabel_duplikasi"].rename(columns={
        "baris_sebelum": "jumlah_baris_sebelum",
        "baris_sesudah": "jumlah_baris_sesudah",
    })[["stasiun", "jumlah_baris_sebelum", "jumlah_duplikat", "jumlah_baris_sesudah"]]

    # ---- Ringkasan Missing Value: sebelum vs sesudah ----
    # 'missing_sebelum_single_gap' = total missing sebelum tahap ini dijalankan
    # (termasuk nilai nol->NaN pada kolom gas), 'sisa_sequential_gap' = missing
    # yang masih tersisa setelah penanganan single gap.
    tabel_missing_ringkas = STORE["tabel_missing"].rename(columns={
        "missing_sebelum_single_gap": "missing_sebelum",
        "sisa_sequential_gap": "missing_sesudah",
    })[["stasiun", "missing_sebelum", "missing_sesudah"]]

    # ---- Ringkasan Outlier: sebelum vs sesudah ----
    # 'sebelum' = total nilai yang terdeteksi sebagai outlier (ambang domain +
    # Isolation Forest) sebelum ditangani. Sesudah ditangani, seluruh nilai
    # outlier sudah diimputasi ulang (atau barisnya di-drop kalau gap-nya
    # terlalu panjang), jadi sesudah selalu 0.
    tabel_outlier_ringkas = (
        STORE["tabel_deteksi_outlier"]
        .groupby("stasiun", as_index=False)["outlier_terdeteksi"]
        .sum()
        .rename(columns={"outlier_terdeteksi": "outlier_sebelum"})
    )
    tabel_outlier_ringkas["outlier_sesudah"] = 0

    # ---- Rules final pada kombinasi support/confidence terpilih ----
    rules_terpilih_html = {}
    for stasiun, r in fp["rules_terpilih"].items():
        if r is not None and not r.empty:
            tampil = r[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).copy()
            tampil['antecedents'] = tampil['antecedents'].apply(lambda x: ", ".join(sorted(x)))
            tampil['consequents'] = tampil['consequents'].apply(lambda x: ", ".join(sorted(x)))
            rules_terpilih_html[stasiun] = df_to_html_table(tampil, max_rows=10)
        else:
            rules_terpilih_html[stasiun] = "<p class='text-muted fst-italic'>Tidak ada rule.</p>"

    return render_template(
        "analysis.html",
        uploaded=True,
        sudah_gabung=True,
        daftar_stasiun=daftar_stasiun,
        sudah_dijalankan=True,
        stasiun_contoh=stasiun_contoh,

        tabel_duplikasi_html=df_to_html_table(tabel_duplikasi_ringkas, max_rows=20),
        tabel_missing_html=df_to_html_table(tabel_missing_ringkas, max_rows=20),
        tabel_outlier_html=df_to_html_table(tabel_outlier_ringkas, max_rows=20),
        preview_transformasi_html=preview_transformasi_html,

        tabel_transaksi_html=df_to_html_table(fp["tabel_transaksi"], max_rows=5),
        support_final=fp["support_final"],
        confidence_final=fp["confidence_final"],
        rules_terpilih_html=rules_terpilih_html,
    )


@app.route("/analisis/jalankan", methods=["POST"])
def jalankan_analisis():
    if data_belum_digabung():
        flash("Silakan unggah data terlebih dahulu di halaman Upload.", "warning")
        return redirect(url_for("index"))

    daftar_stasiun = STORE["daftar_stasiun"]

    dataset_dedup, tabel_duplikasi = pl.tangani_duplikasi(STORE["gabungan_perstasiun"])
    STORE["dataset_dedup"] = dataset_dedup
    STORE["tabel_duplikasi"] = tabel_duplikasi

    dataset_missing_ok, tabel_missing = pl.tangani_missing_value(STORE["dataset_dedup"])
    STORE["dataset_missing_ok"] = dataset_missing_ok
    STORE["tabel_missing"] = tabel_missing

    dataset_bersih, tabel_deteksi, tabel_validasi = pl.tangani_outlier(STORE["dataset_missing_ok"])
    STORE["dataset_bersih"] = dataset_bersih
    STORE["tabel_deteksi_outlier"] = tabel_deteksi
    STORE["tabel_validasi_outlier"] = tabel_validasi

    STORE["dataset_transformasi"] = pl.transformasi_diskritisasi(STORE["dataset_bersih"])
    STORE["hasil_fpgrowth"] = pl.jalankan_fpgrowth(STORE["dataset_transformasi"], daftar_stasiun)

    flash("Analisis berhasil dijalankan.", "success")
    return redirect(url_for("analisis"))


# ============================================================
# HALAMAN 4 — OUTPUT PROGRAM
# ============================================================

@app.route("/output", methods=["GET"])
def output():
    if data_belum_digabung():
        flash("Silakan unggah data terlebih dahulu di halaman Upload.", "warning")
        return redirect(url_for("index"))

    if STORE["hasil_fpgrowth"] is None:
        flash("Silakan buka halaman Analisis Data terlebih dahulu untuk menjalankan pemrosesan.", "warning")
        return redirect(url_for("analisis"))

    fp = STORE["hasil_fpgrowth"]
    daftar_stasiun = STORE["daftar_stasiun"]

    rules_terpilih_html = {}
    jumlah_rule_perstasiun = []
    for stasiun in daftar_stasiun:
        r = fp["rules_terpilih"].get(stasiun)
        if r is not None and not r.empty:
            tampil = r[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            tampil['antecedents'] = tampil['antecedents'].apply(lambda x: ", ".join(sorted(x)))
            tampil['consequents'] = tampil['consequents'].apply(lambda x: ", ".join(sorted(x)))
            rules_terpilih_html[stasiun] = df_to_html_table(tampil, max_rows=25)
            jumlah_rule_perstasiun.append({"stasiun": stasiun, "jumlah_rule": len(r)})
        else:
            rules_terpilih_html[stasiun] = "<p class='text-muted fst-italic'>Tidak ada rule pada kombinasi terpilih.</p>"
            jumlah_rule_perstasiun.append({"stasiun": stasiun, "jumlah_rule": 0})

    # Retensi data akhir (baris awal vs akhir setelah preprocessing)
    retensi = []
    for stasiun in daftar_stasiun:
        n_awal = len(STORE["gabungan_perstasiun"][stasiun])
        n_akhir = len(STORE["dataset_bersih"][stasiun])
        retensi.append({
            "stasiun": stasiun, "baris_awal": n_awal, "baris_akhir": n_akhir,
            "persen_tersisa": round(n_akhir / n_awal * 100, 2) if n_awal > 0 else 0
        })
    tabel_retensi_html = df_to_html_table(pd.DataFrame(retensi))

    # Visualisasi yang langsung menjawab rumusan masalah:
    # 1) pola asosiasi kondisi meteorologi & polutan terhadap kualitas udara
    # 2) aturan asosiasi terkuat berdasarkan Support, Confidence, Lift
    gambar_pola_asosiasi = pl.plot_frekuensi_item_rules(fp["rules_terpilih"], daftar_stasiun)
    gambar_peta_kekuatan_rules = pl.plot_scatter_support_confidence(fp["rules_terpilih"], daftar_stasiun)
    gambar_top_rules = pl.plot_top_rules_lift(fp["rules_terpilih"], daftar_stasiun, n=10)

    # Scatter cuaca vs polutan per stasiun (pindah dari halaman EDA) — jadi
    # tinjauan awal (korelasi linear, data BERSIH pasca-preprocessing) yang
    # ditaruh berdampingan dengan hasil formal FP-Growth di atas, supaya
    # kedua sudut pandang hubungan cuaca-polusi (korelasi vs association
    # rule) ada di satu halaman yang sama.
    gambar_hubungan_cuaca_polutan = {
        stasiun: pl.plot_hubungan_cuaca_polutan_stasiun(STORE["dataset_bersih"][stasiun], stasiun)
        for stasiun in daftar_stasiun
    }

    return render_template(
        "output.html",
        uploaded=True,
        sudah_gabung=True,
        daftar_stasiun=daftar_stasiun,
        support_final=fp["support_final"],
        confidence_final=fp["confidence_final"],
        rules_terpilih_html=rules_terpilih_html,
        tabel_jumlah_rule_html=df_to_html_table(pd.DataFrame(jumlah_rule_perstasiun)),
        tabel_retensi_html=tabel_retensi_html,
        gambar_pola_asosiasi=gambar_pola_asosiasi,
        gambar_peta_kekuatan_rules=gambar_peta_kekuatan_rules,
        gambar_top_rules=gambar_top_rules,
        gambar_hubungan_cuaca_polutan=gambar_hubungan_cuaca_polutan,
    )


# ============================================================
# DOWNLOAD CSV
# ============================================================

@app.route("/download/rules")
def download_rules():
    if data_belum_digabung() or STORE["hasil_fpgrowth"] is None:
        flash("Belum ada hasil untuk diunduh.", "warning")
        return redirect(url_for("index"))

    fp = STORE["hasil_fpgrowth"]
    frames = []
    for stasiun, r in fp["rules_terpilih"].items():
        if r is not None and not r.empty:
            tampil = r[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            tampil['antecedents'] = tampil['antecedents'].apply(lambda x: ", ".join(sorted(x)))
            tampil['consequents'] = tampil['consequents'].apply(lambda x: ", ".join(sorted(x)))
            tampil.insert(0, "stasiun", stasiun)
            frames.append(tampil)

    if not frames:
        flash("Tidak ada rule pada kombinasi terpilih untuk diunduh.", "warning")
        return redirect(url_for("output"))

    hasil = pd.concat(frames, ignore_index=True)
    buf = io.StringIO()
    hasil.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name="hasil_association_rules.csv"
    )


@app.route("/download/dataset/<stasiun>")
def download_dataset(stasiun):
    if data_belum_digabung() or STORE["dataset_bersih"] is None or stasiun not in STORE["dataset_bersih"]:
        flash("Data tidak ditemukan.", "warning")
        return redirect(url_for("index"))

    df = STORE["dataset_bersih"][stasiun]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    nama_file = stasiun.replace(" ", "_") + "_bersih.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=nama_file)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)