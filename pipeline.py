"""
pipeline.py
============
Berisi seluruh logika pemrosesan data untuk aplikasi ISPU (Metode B, mengikuti
interpolasi__2_.py):
  1. Selection (load & gabung data ISPU + Meteorologi)
  2. Penanganan Missing Value:
       a) Nilai NOL pada kolom gas (SO2, CO, O3, NO2) -> NaN
       b) HANYA single gap yang diisi (linear); sequential gap dibiarkan NaN
  3. Deteksi & Penanganan Outlier:
       a) Ambang domain + Isolation Forest SAJA (IQR & Z-score tidak dipakai
          di pipeline final)
       b) Sequential gap yang lebih panjang dari ambang IQR (dihitung dari
          panjang gap) dikecualikan dari imputasi (tetap NaN, di-drop di akhir)
       c) Sisanya diimputasi: single gap -> linear, sequential gap -> PCHIP
  4. Transformasi / Diskritisasi
  5. Association Rule Mining (FP-Growth)

Modul ini dipisah dari app.py supaya app.py fokus pada routing saja.
"""

import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-GUI, wajib untuk server Flask
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from scipy.interpolate import PchipInterpolator
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ============================================================
# KONSTANTA
# ============================================================

KOLOM_NUMERIK = [
    'pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida',
    'ozon', 'nitrogen_dioksida', 'temp_air', 'hum_air', 'ff', 'rain'
]

BATAS_DOMAIN = {
    'pm_sepuluh':        (0, 500),
    'pm_duakomalima':    (0, 500),
    'sulfur_dioksida':   (0, 1000),
    'karbon_monoksida':  (0, 40000),
    'ozon':              (0, 1000),
    'nitrogen_dioksida': (0, 3000),
    'temp_air':          (15, 45),
    'hum_air':           (0, 100),
    'ff':                (0, 32.6),
    'rain':              (0, 400),
}

PARAMETER_ISPU = [
    'pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida',
    'karbon_monoksida', 'ozon', 'nitrogen_dioksida'
]

# Kolom gas -- nilai NOL dianggap tidak valid (konsentrasi gas tidak pernah
# benar-benar nol) dan diubah jadi NaN SEBELUM tahap missing value apa pun
# dijalankan. TIDAK diterapkan ke 'rain'/meteo lain, karena 0 di sana adalah
# kondisi VALID (mis. tidak hujan). Sesuai Metode B (interpolasi__2_.py 2.1b).
KOLOM_GAS = ['sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']

KOLOM_ANTECEDENT = [
    'kategori_pm_sepuluh', 'kategori_pm_duakomalima', 'kategori_sulfur_dioksida',
    'kategori_karbon_monoksida', 'kategori_ozon', 'kategori_nitrogen_dioksida',
    'kategori_suhu', 'kategori_kelembaban', 'kategori_angin', 'kategori_hujan'
]
KOLOM_CONSEQUENT = 'kategori'

PREFIX_KOLOM = {
    'kategori_pm_sepuluh':        'pm10',
    'kategori_pm_duakomalima':    'pm25',
    'kategori_sulfur_dioksida':   'so2',
    'kategori_karbon_monoksida':  'co',
    'kategori_ozon':              'o3',
    'kategori_nitrogen_dioksida': 'no2',
    'kategori_suhu':              'suhu',
    'kategori_kelembaban':        'lembab',
    'kategori_angin':             'angin',
    'kategori_hujan':             'hujan',
}

SKENARIO_SUPPORT = [0.5, 0.6, 0.7]
VARIASI_CONFIDENCE = [0.5, 0.6, 0.7, 0.8, 0.9]


# ============================================================
# UTILITAS
# ============================================================

def ringkasan_praanalisis(gabungan_perstasiun, daftar_stasiun):
    """
    Ringkasan teknis kondisi data mentah SEBELUM pipeline penanganan dijalankan:
    statistik deskriptif, missing value, dan screening outlier cepat (berbasis
    ambang domain/batas fisis) untuk seluruh stasiun sekaligus.
    """
    deskripsi_perstasiun = {}
    daftar_missing = []
    daftar_outlier = []

    for stasiun in daftar_stasiun:
        df = gabungan_perstasiun[stasiun]

        deskripsi_perstasiun[stasiun] = (
            df[KOLOM_NUMERIK].describe().round(2).reset_index().rename(columns={"index": "statistik"})
        )

        missing_summary = df[KOLOM_NUMERIK].isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        for kolom, jumlah in missing_summary.items():
            daftar_missing.append({
                "stasiun": stasiun,
                "kolom": kolom,
                "jumlah_missing": int(jumlah),
            })

        for kolom in KOLOM_NUMERIK:
            batas_bawah, batas_atas = BATAS_DOMAIN[kolom]
            jumlah_luar = int(((df[kolom] < batas_bawah) | (df[kolom] > batas_atas)).sum())
            if jumlah_luar > 0:
                daftar_outlier.append({
                    "stasiun": stasiun,
                    "kolom": kolom,
                    "jumlah_di_luar_domain": jumlah_luar,
                })

    kolom_missing = ["stasiun", "kolom", "jumlah_missing"]
    kolom_outlier = ["stasiun", "kolom", "jumlah_di_luar_domain"]

    return {
        "deskripsi_perstasiun": deskripsi_perstasiun,
        "tabel_missing_gabungan": pd.DataFrame(daftar_missing, columns=kolom_missing),
        "tabel_outlier_gabungan": pd.DataFrame(daftar_outlier, columns=kolom_outlier),
    }


def fig_to_base64(fig):
    """Ubah matplotlib figure jadi string base64 supaya bisa ditaruh langsung di tag <img>."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def standarisasi_stasiun(nama):
    nama = str(nama).upper()
    if 'DKI1' in nama:
        return 'DKI1 BUNDARAN HI'
    elif 'DKI2' in nama:
        return 'DKI2 KELAPA GADING'
    elif 'DKI3' in nama:
        return 'DKI3 JAGAKARSA'
    elif 'DKI4' in nama:
        return 'DKI4 LUBANG BUAYA'
    elif 'DKI5' in nama:
        return 'DKI5 KEBUN JERUK'
    else:
        return 'TIDAK DIKETAHUI'


def _baca_file(file_storage):
    """
    Baca file upload (FileStorage Flask) baik format .xls (html table),
    .xlsx, maupun .csv. Mengembalikan DataFrame.
    """
    nama = file_storage.filename.lower()
    data = file_storage.read()
    buf = io.BytesIO(data)

    if nama.endswith(".csv"):
        return pd.read_csv(buf)

    # Coba baca sebagai Excel biner dulu
    try:
        buf.seek(0)
        return pd.read_excel(buf)
    except Exception:
        pass

    # Fallback: banyak file ".xls" dari sistem lama sebenarnya adalah tabel HTML
    try:
        buf.seek(0)
        tabel = pd.read_html(buf)
        return tabel[0]
    except Exception as e:
        raise ValueError(f"Gagal membaca file '{file_storage.filename}': {e}")


# ============================================================
# TAHAP 1 — SELECTION
# ============================================================

def baca_mentah(file_ispu, file_meteo):
    """
    TAHAP RINGAN (dipanggil saat submit form upload): hanya baca file +
    bersihkan nama kolom + standarisasi nama stasiun. TIDAK melakukan
    parsing tanggal, resample, atau merge -> makanya cepat, cocok untuk
    langsung ditampilkan sebagai preview data mentah.
    Mengembalikan (df_ispu, df_meteo) mentah.
    """
    df_ispu = _baca_file(file_ispu)
    df_ispu.columns = [str(c).strip() for c in df_ispu.columns]
    if 'max' in df_ispu.columns:
        df_ispu = df_ispu.drop(columns=['max'])
    if 'parameter_pencemar_kritis' in df_ispu.columns:
        df_ispu = df_ispu.drop(columns=['parameter_pencemar_kritis'])
    df_ispu['stasiun'] = df_ispu['stasiun'].apply(standarisasi_stasiun)

    df_meteo = _baca_file(file_meteo)
    df_meteo.columns = [str(c).strip() for c in df_meteo.columns]
    if 'grad' in df_meteo.columns:
        df_meteo = df_meteo.drop(columns=['grad'])
    if 'stasiun_pengamatan' in df_meteo.columns:
        df_meteo = df_meteo.rename(columns={'stasiun_pengamatan': 'stasiun'})
    df_meteo['stasiun'] = df_meteo['stasiun'].apply(standarisasi_stasiun)

    return df_ispu, df_meteo


def gabung_dari_mentah(df_ispu, df_meteo):
    """
    TAHAP BERAT (dipanggil saat klik tombol "Gabungkan Data"): build kolom
    periode, agregasi harian meteorologi, lalu gabungkan per stasiun.
    Menerima df_ispu & df_meteo hasil baca_mentah().
    Mengembalikan dict: {nama_stasiun: DataFrame}
    """
    # ---------- ISPU ----------
    ispu_perstasiun = {}
    for stasiun in df_ispu['stasiun'].unique():
        if stasiun == 'TIDAK DIKETAHUI':
            continue
        df_s = df_ispu[df_ispu['stasiun'] == stasiun].copy().reset_index(drop=True)
        df_s['periode'] = (
            df_s['periode_data'].astype(str).str[:4] + '-' +
            df_s['bulan'].astype(str) + '-' +
            df_s['tanggal'].astype(str)
        )
        df_s['periode'] = pd.to_datetime(df_s['periode'], format='%Y-%m-%d', errors='coerce')
        df_s = df_s.drop(columns=['periode_data', 'bulan', 'tanggal'])
        cols = ['periode', 'stasiun'] + [c for c in df_s.columns if c not in ['periode', 'stasiun']]
        ispu_perstasiun[stasiun] = df_s[cols].sort_values('periode').reset_index(drop=True)

    # ---------- METEOROLOGI ----------
    meteo_perstasiun = {}
    for stasiun in df_meteo['stasiun'].unique():
        if stasiun == 'TIDAK DIKETAHUI':
            continue
        df_s = df_meteo[df_meteo['stasiun'] == stasiun].copy().reset_index(drop=True)
        df_s['periode'] = (
            df_s['periode_data'].astype(str).str[:4] + '-' +
            df_s['periode_data'].astype(str).str[4:].str.zfill(2) + '-' +
            df_s['tanggal'].astype(str).str.zfill(2)
        )
        df_s['periode'] = pd.to_datetime(df_s['periode'], format='%Y-%m-%d', errors='coerce')
        df_s = df_s.drop(columns=['periode_data', 'tanggal'])
        cols = ['periode', 'stasiun'] + [c for c in df_s.columns if c not in ['periode', 'stasiun']]
        meteo_perstasiun[stasiun] = df_s[cols].sort_values('periode').reset_index(drop=True)

    # ---------- AGREGASI HARIAN METEOROLOGI ----------
    meteo_harian = {}
    for stasiun, df_s in meteo_perstasiun.items():
        df_s = df_s.set_index('periode').drop(columns=['stasiun'])
        kolom_mean = [c for c in df_s.columns if c != 'rain']
        df_mean = df_s[kolom_mean].resample('D').mean(numeric_only=True)
        df_sum = df_s[['rain']].resample('D').sum() if 'rain' in df_s.columns else pd.DataFrame()
        df_agg = pd.concat([df_mean, df_sum], axis=1)
        df_agg['stasiun'] = stasiun
        cols = ['stasiun'] + [c for c in df_agg.columns if c != 'stasiun']
        meteo_harian[stasiun] = df_agg[cols].reset_index()

    # ---------- GABUNGKAN ----------
    daftar_stasiun = sorted(set(ispu_perstasiun.keys()) & set(meteo_harian.keys()))
    gabungan_perstasiun = {}
    for stasiun in daftar_stasiun:
        df_gabung = pd.merge(
            ispu_perstasiun[stasiun], meteo_harian[stasiun],
            on=['periode', 'stasiun'], how='inner'
        )
        gabungan_perstasiun[stasiun] = df_gabung.sort_values('periode').reset_index(drop=True)

    return gabungan_perstasiun, daftar_stasiun


# ============================================================
# TAHAP 1B — DUPLIKASI DATA
# ============================================================
# Dijalankan sesudah data digabung per stasiun, SEBELUM penanganan missing
# value. Duplikat didefinisikan sebagai baris dengan 'periode' (tanggal) yang
# sama pada stasiun yang sama -- seharusnya cuma ada 1 baris per hari per
# stasiun. Kemunculan pertama dipertahankan, sisanya dibuang.

def tangani_duplikasi(gabungan_perstasiun):
    """
    Deteksi & buang baris duplikat per stasiun (duplikat berdasarkan kolom
    'periode', fallback ke duplikat baris penuh kalau kolom itu tidak ada).
    Kembalikan (dataset_hasil, tabel_rekap_duplikasi) dengan rekap
    baris_sebelum / jumlah_duplikat / baris_sesudah per stasiun.
    """
    dataset = {}
    rekap = []
    for stasiun, df in gabungan_perstasiun.items():
        n_awal = len(df)
        subset = ['periode'] if 'periode' in df.columns else None
        n_duplikat = int(df.duplicated(subset=subset).sum())
        df_bersih = df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
        rekap.append({
            'stasiun': stasiun,
            'baris_sebelum': n_awal,
            'jumlah_duplikat': n_duplikat,
            'baris_sesudah': len(df_bersih),
        })
        dataset[stasiun] = df_bersih
    return dataset, pd.DataFrame(rekap)


# ============================================================
# TAHAP 2 — MISSING VALUE (Metode B)
# ============================================================
# Urutan sesuai flowchart acuan: Handle Zero Values -> Handle Single Missing
# Values -> (baru nanti) Outlier Detection -> PCHIP. Jadi di tahap ini HANYA
# zero->NaN (kolom gas) dan pengisian single gap (linear) yang dilakukan.
# Sequential gap SENGAJA dibiarkan NaN -- baru diimputasi PCHIP di tahap
# outlier (tangani_outlier), SETELAH deteksi outlier selesai.

def imputasi_linear_pchip(df, kolom):
    """Linear interpolation untuk single gap, PCHIP untuk sequential gap, fallback ffill/bfill di boundary.
    Dipertahankan sebagai fungsi publik (dipakai ulang di tahap outlier)."""
    s = df[kolom].copy()
    s = s.interpolate(method='linear', limit_area='inside')
    if s.isnull().sum() > 0:
        valid = s.notna()
        if valid.sum() >= 2:
            pchip = PchipInterpolator(df.index[valid], s[valid], extrapolate=False)
            idx_nan = df.index[s.isnull()]
            s.loc[idx_nan] = pchip(idx_nan.to_numpy())
    return s.ffill().bfill()


def _isi_single_gap_saja(s):
    """HANYA mengisi single gap (NaN diapit dua nilai valid) dengan linear.
    Sequential gap dibiarkan NaN."""
    return s.interpolate(method='linear', limit_area='inside')


def tangani_missing_value(gabungan_perstasiun):
    """
    TAHAP 2 (Metode B):
      1) Nilai NOL pada kolom gas (SO2, CO, O3, NO2) diubah jadi NaN dulu.
      2) Hanya SINGLE GAP yang diisi (linear) di sini. SEQUENTIAL GAP
         sengaja dibiarkan NaN, baru ditangani (PCHIP) di tangani_outlier(),
         setelah deteksi outlier berjalan.
    Kembalikan (dataset_hasil, tabel_rekap_missing).
    """
    dataset = {s: df.copy() for s, df in gabungan_perstasiun.items()}
    rekap = []
    for stasiun, df in dataset.items():
        # ---- 2.1: zero -> NaN, khusus kolom gas ----
        kolom_gas_ada = [k for k in KOLOM_GAS if k in df.columns]
        n_zero = 0
        for kolom in kolom_gas_ada:
            n_zero += int((df[kolom] == 0).sum())
            df.loc[df[kolom] == 0, kolom] = np.nan

        # ---- 2.2: isi single gap saja (linear) ----
        kolom_ada = [k for k in KOLOM_NUMERIK if k in df.columns]
        n_sebelum = int(df[kolom_ada].isnull().sum().sum())
        for kolom in kolom_ada:
            df[kolom] = _isi_single_gap_saja(df[kolom])
        n_sesudah = int(df[kolom_ada].isnull().sum().sum())

        rekap.append({
            'stasiun': stasiun,
            'nol_gas_jadi_missing': n_zero,
            'missing_sebelum_single_gap': n_sebelum,
            'terisi_single_gap_linear': n_sebelum - n_sesudah,
            'sisa_sequential_gap': n_sesudah,
        })
        dataset[stasiun] = df
    return dataset, pd.DataFrame(rekap)


# ============================================================
# TAHAP 3 — OUTLIER (Metode B)
# ============================================================
# Mengikuti jurnal acuan: deteksi outlier pakai ambang domain + Isolation
# Forest SAJA (IQR & Z-Score TIDAK dipakai di pipeline final -- keduanya
# cuma metode pembanding pada tahap evaluasi jurnal acuan). Dijalankan di
# atas dataset yang sudah melalui zero->NaN & single-gap linear (TAHAP 2).
#
# Sequential gap yang PANJANGNYA melebihi ambang IQR (dihitung dari seluruh
# panjang gap, digabung semua stasiun & kolom) dikecualikan dari imputasi --
# tetap dibiarkan NaN, dan barisnya di-drop di akhir (bukan dipaksa diisi).

def _deteksi_outlier_isoforest(df, kolom):
    X = df[[kolom]].dropna()
    if len(X) < 10:
        return pd.Series(False, index=df.index)
    model = IsolationForest(n_estimators=1000, contamination=0.02, random_state=42)
    pred = model.fit_predict(X)
    hasil = pd.Series(False, index=df.index)
    hasil.loc[X.index[pred == -1]] = True
    return hasil


def _klasifikasi_gap(s):
    mask_na = s.isnull()
    prev_ok = s.shift(1).notna()
    next_ok = s.shift(-1).notna()
    single_gap = mask_na & prev_ok & next_ok
    sequential_gap = mask_na & ~single_gap
    return single_gap, sequential_gap


def _cari_run_berurutan(mask_bool):
    """Cari posisi (start, end, panjang) tiap 'run' True yang berurutan pada sebuah mask boolean."""
    runs = []
    arr = mask_bool.to_numpy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            j = i
            while j < n and arr[j]:
                j += 1
            runs.append((i, j - 1, j - i))
            i = j
        else:
            i += 1
    return runs


def _imputasi_dengan_pengecualian(df, kolom, idx_dikecualikan):
    """Sama seperti imputasi_linear_pchip(), tapi posisi pada idx_dikecualikan
    (gap yang terlalu panjang) TETAP dibiarkan NaN, tidak dipaksa diimputasi."""
    s_asli = df[kolom].copy()
    mask_awal_nan = s_asli.isnull()
    s_full = imputasi_linear_pchip(df, kolom)
    hasil = s_asli.copy()
    if len(idx_dikecualikan) == 0:
        boleh_isi = mask_awal_nan
    else:
        boleh_isi = mask_awal_nan & ~df.index.isin(idx_dikecualikan)
    hasil.loc[boleh_isi] = s_full.loc[boleh_isi]
    return hasil


def tangani_outlier(dataset_missing_ok):
    """
    TAHAP 3 (Metode B):
      1) Deteksi outlier di SEMUA kolom numerik: ambang domain + Isolation
         Forest. Outlier diubah jadi NaN.
      2) Gap diklasifikasi ulang (single vs sequential).
      3) Ambang IQR dihitung dari PANJANG seluruh sequential gap (gabung
         semua stasiun & kolom) -- gap yang lebih panjang dari ambang ini
         DIKECUALIKAN dari imputasi (tetap NaN).
      4) Imputasi ulang: single gap -> linear, sequential gap (non-excluded)
         -> PCHIP.
      5) Validasi statistik (delta mean/varians/median < 5%); baris yang
         dikecualikan tidak ikut dibandingkan.
      6) Baris yang masih NaN (gap dikecualikan) di-drop dari dataset akhir,
         karena FP-Growth butuh dataset bebas NaN.
    Kembalikan (dataset_bersih, tabel_deteksi, tabel_validasi).
    """
    dataset = {s: df.copy() for s, df in dataset_missing_ok.items()}
    rekap_deteksi = []
    semua_panjang_gap = []          # pooled semua stasiun & kolom -> ambang IQR
    info_run_perkolom = {}          # (stasiun, kolom) -> list [(start, end, panjang), ...]

    # ---- 3.1-3.2: deteksi outlier (domain + isoforest) & klasifikasi gap ----
    for stasiun, df in dataset.items():
        kolom_ada = [k for k in KOLOM_NUMERIK if k in df.columns]
        for col in kolom_ada:
            batas_bawah, batas_atas = BATAS_DOMAIN[col]
            outlier_domain = (df[col] < batas_bawah) | (df[col] > batas_atas)
            outlier_iso = _deteksi_outlier_isoforest(df, col)
            outlier_final = outlier_domain | outlier_iso

            df.loc[outlier_final, col] = np.nan

            single, sequential = _klasifikasi_gap(df[col])
            runs = _cari_run_berurutan(sequential)
            info_run_perkolom[(stasiun, col)] = runs
            semua_panjang_gap.extend(panjang for (_, _, panjang) in runs)

            rekap_deteksi.append({
                'stasiun': stasiun, 'kolom': col,
                'outlier_terdeteksi': int(outlier_final.sum()),
                'single_gap': int(single.sum()),
                'sequential_gap': int(sequential.sum()),
            })
        dataset[stasiun] = df

    # ---- 3.3: ambang IQR untuk PANJANG gap (dipakai untuk eksklusi imputasi) ----
    if semua_panjang_gap:
        panjang_series = pd.Series(semua_panjang_gap)
        q1_gap = panjang_series.quantile(0.25)
        q3_gap = panjang_series.quantile(0.75)
        batas_atas_gap = q3_gap + 1.5 * (q3_gap - q1_gap)
    else:
        batas_atas_gap = 0

    # ---- 3.4: imputasi ulang (linear + PCHIP), gap terlalu panjang dikecualikan ----
    rekap_gap_dikecualikan = {}
    for stasiun, df in dataset.items():
        df['_gap_dikecualikan'] = False
        kolom_ada = [k for k in KOLOM_NUMERIK if k in df.columns]
        for col in kolom_ada:
            runs = info_run_perkolom.get((stasiun, col), [])
            idx_excl = []
            for (start, end, panjang) in runs:
                if panjang > batas_atas_gap:
                    idx_excl.extend(df.index[start:end + 1])
            if idx_excl:
                df.loc[idx_excl, '_gap_dikecualikan'] = True
            rekap_gap_dikecualikan[(stasiun, col)] = len(idx_excl)
            df[col] = _imputasi_dengan_pengecualian(df, col, idx_excl)
        dataset[stasiun] = df

    for baris in rekap_deteksi:
        baris['gap_dikecualikan'] = rekap_gap_dikecualikan.get((baris['stasiun'], baris['kolom']), 0)
    tabel_deteksi = pd.DataFrame(rekap_deteksi)

    # ---- 3.5: validasi statistik (delta < 5%), baris dikecualikan tidak ikut dibandingkan ----
    rekap_validasi = []
    for stasiun, df in dataset.items():
        mask_excl = df['_gap_dikecualikan']
        kolom_ada = [k for k in KOLOM_NUMERIK if k in df.columns]
        for col in kolom_ada:
            batas_bawah, batas_atas = BATAS_DOMAIN[col]
            awal = dataset_missing_ok[stasiun].loc[~mask_excl.values, col]
            awal_valid = awal[(awal >= batas_bawah) & (awal <= batas_atas)].dropna()
            akhir = df.loc[~mask_excl, col].dropna()

            def delta(a, b):
                return abs(b - a) / a * 100 if a not in (0, None) and not pd.isna(a) and a != 0 else np.nan

            d_mean = delta(awal_valid.mean(), akhir.mean())
            d_var = delta(awal_valid.var(), akhir.var())
            d_med = delta(awal_valid.median(), akhir.median())
            nilai_delta = [x for x in [d_mean, d_var, d_med] if pd.notna(x)]
            status = 'VALID (< 5%)' if (len(nilai_delta) == 0 or max(nilai_delta) < 5) else 'PERLU DICEK'

            rekap_validasi.append({
                'stasiun': stasiun, 'kolom': col,
                'delta_mean_%': round(d_mean, 2) if pd.notna(d_mean) else None,
                'delta_variance_%': round(d_var, 2) if pd.notna(d_var) else None,
                'delta_median_%': round(d_med, 2) if pd.notna(d_med) else None,
                'status': status
            })

    tabel_validasi = pd.DataFrame(rekap_validasi)

    # ---- 3.6: baris yang masih NaN (gap dikecualikan) di-drop dari dataset akhir ----
    for stasiun, df in dataset.items():
        kolom_ada = [k for k in KOLOM_NUMERIK if k in df.columns]
        dataset[stasiun] = (
            df.dropna(subset=kolom_ada)
              .drop(columns=['_gap_dikecualikan'])
              .reset_index(drop=True)
        )

    return dataset, tabel_deteksi, tabel_validasi


# ============================================================
# TAHAP 4 — TRANSFORMASI / DISKRITISASI
# ============================================================

BATAS_ISPU = {
    'pm_sepuluh': [50, 100, 200, 300],
    'pm_duakomalima': [15.5, 55.4, 150.4, 250.4],
    'sulfur_dioksida': [42, 180, 400, 800],
    'karbon_monoksida': [4000, 8000, 15000, 30000],
    'ozon': [120, 235, 400, 800],
    'nitrogen_dioksida': [80, 200, 1130, 2260],
}


def kategori_ispu(nilai, parameter):
    b = BATAS_ISPU[parameter]
    if nilai <= b[0]:
        return 'Baik'
    elif nilai <= b[1]:
        return 'Sedang'
    elif nilai <= b[2]:
        return 'Tidak Sehat'
    elif nilai <= b[3]:
        return 'Sangat Tidak Sehat'
    else:
        return 'Berbahaya'


def kategori_suhu(nilai):
    if nilai < 25:
        return 'Sejuk'
    elif nilai <= 27:
        return 'Normal'
    else:
        return 'Panas'


def kategori_kelembaban(nilai):
    if nilai < 70:
        return 'Kering'
    elif nilai <= 75:
        return 'Agak Kering'
    elif nilai <= 80:
        return 'Sedang'
    elif nilai <= 85:
        return 'Lembab'
    else:
        return 'Basah'


def kategori_angin(nilai):
    skala = [
        (0.3, 'Sedikit Tenang'), (1.5, 'Sedikit Hembusan'), (3.3, 'Angin Hembusan'),
        (5.5, 'Angin Pelan Hembusan'), (8.0, 'Angin Sedang Sejuk'), (10.8, 'Hembusan Angin'),
        (13.9, 'Kuat'), (17.2, 'Mendekati Kencang'), (20.7, 'Kencang'),
        (24.5, 'Kencang Sekali'), (28.4, 'Badai'), (32.6, 'Badai Dahsyat')
    ]
    for batas_atas, deskripsi in skala:
        if nilai <= batas_atas:
            return deskripsi
    return "Badai Topan"


def kategori_hujan(nilai):
    if nilai == 0:
        return 'Tidak Hujan'
    elif nilai <= 20:
        return 'Hujan Ringan'
    elif nilai <= 50:
        return 'Hujan Sedang'
    elif nilai <= 100:
        return 'Hujan Lebat'
    else:
        return 'Hujan Sangat Lebat'


def transformasi_diskritisasi(dataset_bersih):
    dataset = {s: df.copy() for s, df in dataset_bersih.items()}
    for stasiun, df in dataset.items():
        for param in PARAMETER_ISPU:
            if param in df.columns:
                df[f'kategori_{param}'] = df[param].apply(lambda x: kategori_ispu(x, param))
        if 'temp_air' in df.columns:
            df['kategori_suhu'] = df['temp_air'].apply(kategori_suhu)
        if 'hum_air' in df.columns:
            df['kategori_kelembaban'] = df['hum_air'].apply(kategori_kelembaban)
        if 'ff' in df.columns:
            df['kategori_angin'] = df['ff'].apply(kategori_angin)
        if 'rain' in df.columns:
            df['kategori_hujan'] = df['rain'].apply(kategori_hujan)
        dataset[stasiun] = df
    return dataset


# ============================================================
# TAHAP 5 — FP-GROWTH
# ============================================================

def jalankan_fpgrowth(dataset_transformasi, daftar_stasiun):
    """
    Jalankan seluruh pipeline FP-Growth (4a-4f) dan kembalikan dict berisi
    semua tabel perantara supaya bisa ditampilkan step-by-step di halaman Analisis.
    """
    hasil = {}

    # ---- 4a: transaksi ----
    transaksi_perstasiun = {}
    rekap_transaksi = []
    for stasiun in daftar_stasiun:
        df_g = dataset_transformasi[stasiun]
        if KOLOM_CONSEQUENT not in df_g.columns:
            transaksi_perstasiun[stasiun] = []
            rekap_transaksi.append({'stasiun': stasiun, 'jumlah_transaksi': 0, 'baris_awal': len(df_g)})
            continue
        df_valid = df_g[df_g[KOLOM_CONSEQUENT].notna()].copy().reset_index(drop=True)
        transaksi = []
        kolom_ada = [c for c in KOLOM_ANTECEDENT if c in df_valid.columns]
        for _, row in df_valid.iterrows():
            item_baris = [f"{PREFIX_KOLOM[col]}={row[col]}" for col in kolom_ada]
            item_baris.append(f"ispu={row[KOLOM_CONSEQUENT]}")
            transaksi.append(item_baris)
        transaksi_perstasiun[stasiun] = transaksi
        rekap_transaksi.append({'stasiun': stasiun, 'jumlah_transaksi': len(transaksi), 'baris_awal': len(df_g)})
    hasil['tabel_transaksi'] = pd.DataFrame(rekap_transaksi)

    # ---- 4b: frekuensi & support item ----
    encoded_perstasiun = {}
    tabel_frekuensi_semua = []
    for stasiun in daftar_stasiun:
        transaksi = transaksi_perstasiun[stasiun]
        if len(transaksi) == 0:
            encoded_perstasiun[stasiun] = pd.DataFrame()
            continue
        te = TransactionEncoder()
        te_array = te.fit(transaksi).transform(transaksi)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        encoded_perstasiun[stasiun] = df_encoded

        frekuensi_item = df_encoded.sum()
        support_item = df_encoded.mean()
        tabel_frekuensi_semua.append(pd.DataFrame({
            'stasiun': stasiun, 'item': frekuensi_item.index,
            'support_count': frekuensi_item.values, 'support': support_item.values
        }))
    tabel_frekuensi_gabungan = (
        pd.concat(tabel_frekuensi_semua, ignore_index=True)
        .sort_values(['stasiun', 'support_count'], ascending=[True, False])
        .reset_index(drop=True)
        if tabel_frekuensi_semua else pd.DataFrame()
    )
    hasil['tabel_frekuensi'] = tabel_frekuensi_gabungan
    hasil['top5_perstasiun'] = (
        tabel_frekuensi_gabungan.groupby('stasiun').head(5)
        if not tabel_frekuensi_gabungan.empty else pd.DataFrame()
    )

    # ---- 4c: frequent itemset per kombinasi support ----
    itemset_perstasiun_support = {}
    rekap_itemset = []
    for stasiun in daftar_stasiun:
        df_encoded = encoded_perstasiun[stasiun]
        for sup in SKENARIO_SUPPORT:
            if df_encoded.empty:
                fi = pd.DataFrame()
            else:
                fi = fpgrowth(df_encoded, min_support=sup, use_colnames=True)
                if len(fi) > 0:
                    fi['jumlah_item'] = fi['itemsets'].apply(len)
                    fi = fi.sort_values('support', ascending=False).reset_index(drop=True)
            itemset_perstasiun_support[(stasiun, sup)] = fi
            rekap_itemset.append({
                'stasiun': stasiun, 'min_support': sup,
                'jumlah_itemset': len(fi),
                'itemset_terpanjang': int(fi['jumlah_item'].max()) if len(fi) > 0 else 0
            })
    hasil['tabel_itemset'] = pd.DataFrame(rekap_itemset)

    # ---- 4d: rules mentah ----
    rules_mentah_semua = {}
    rekap_rules_mentah = []
    for stasiun in daftar_stasiun:
        for sup in SKENARIO_SUPPORT:
            fi = itemset_perstasiun_support[(stasiun, sup)]
            for conf in VARIASI_CONFIDENCE:
                if len(fi) == 0:
                    rules_mentah_semua[(stasiun, sup, conf)] = pd.DataFrame()
                    rekap_rules_mentah.append({'stasiun': stasiun, 'min_support': sup,
                                                'min_confidence': conf, 'jumlah_rules_mentah': 0})
                    continue
                rules = association_rules(fi, metric='confidence', min_threshold=conf)
                rules_mentah_semua[(stasiun, sup, conf)] = rules
                rekap_rules_mentah.append({'stasiun': stasiun, 'min_support': sup,
                                            'min_confidence': conf, 'jumlah_rules_mentah': len(rules)})
    hasil['tabel_rules_mentah'] = pd.DataFrame(rekap_rules_mentah)

    # ---- 4e: filter consequent/antecedent/lift ----
    rules_final_semua = {}
    rekap_rules_final = []
    for stasiun in daftar_stasiun:
        for sup in SKENARIO_SUPPORT:
            for conf in VARIASI_CONFIDENCE:
                rm = rules_mentah_semua[(stasiun, sup, conf)]
                if rm.empty:
                    rules_final_semua[(stasiun, sup, conf)] = pd.DataFrame()
                    rekap_rules_final.append({
                        'stasiun': stasiun, 'min_support': sup, 'min_confidence': conf,
                        'jumlah_rules_mentah': 0, 'jumlah_setelah_filter_ca': 0,
                        'jumlah_final_lift_lebih_1': 0
                    })
                    continue
                f1 = rm[rm['consequents'].apply(lambda x: any(i.startswith('ispu=') for i in x))]
                f2 = f1[f1['antecedents'].apply(lambda x: not any(i.startswith('ispu=') for i in x))]
                jml_sebelum_lift = len(f2)
                f3 = f2[f2['lift'] > 1] if not f2.empty else f2
                final = f3.sort_values(['confidence', 'lift'], ascending=False).reset_index(drop=True) if not f3.empty else pd.DataFrame()
                rules_final_semua[(stasiun, sup, conf)] = final
                rekap_rules_final.append({
                    'stasiun': stasiun, 'min_support': sup, 'min_confidence': conf,
                    'jumlah_rules_mentah': len(rm), 'jumlah_setelah_filter_ca': jml_sebelum_lift,
                    'jumlah_final_lift_lebih_1': len(final)
                })
    tabel_rules_final = pd.DataFrame(rekap_rules_final)
    hasil['tabel_rules_final'] = tabel_rules_final

    # ---- 4f: pivot per stasiun + tentukan kombinasi terbaik ----
    pivot_perstasiun = {}
    for stasiun in daftar_stasiun:
        t = tabel_rules_final[tabel_rules_final['stasiun'] == stasiun][
            ['min_support', 'min_confidence', 'jumlah_final_lift_lebih_1']
        ]
        if not t.empty:
            pivot_perstasiun[stasiun] = t.pivot(
                index='min_support', columns='min_confidence', values='jumlah_final_lift_lebih_1'
            )
    hasil['pivot_perstasiun'] = pivot_perstasiun

    cek = tabel_rules_final.groupby(['min_support', 'min_confidence'])['jumlah_final_lift_lebih_1'].agg(
        minimum_antar_stasiun='min', rata_rata_antar_stasiun='mean',
        jumlah_stasiun_ada_rule=lambda x: (x > 0).sum()
    ).reset_index()
    hasil['tabel_cek_kombinasi'] = cek

    kandidat = cek[cek['jumlah_stasiun_ada_rule'] == len(daftar_stasiun)]
    support_final, confidence_final = None, None
    rules_terpilih = {}
    if not kandidat.empty:
        terpilih = kandidat.sort_values(['min_support', 'min_confidence'], ascending=False).iloc[0]
        support_final = float(terpilih['min_support'])
        confidence_final = float(terpilih['min_confidence'])
        for stasiun in daftar_stasiun:
            rf = rules_final_semua[(stasiun, support_final, confidence_final)]
            # Sesuai notebook 4f: consequent harus TEPAT satu item dan item
            # itu adalah kategori ISPU (bukan gabungan ispu + item lain).
            rf = rf[rf['consequents'].apply(
                lambda s: len(s) == 1 and str(list(s)[0]).startswith('ispu=')
            )]
            rules_terpilih[stasiun] = rf.sort_values('confidence', ascending=False).reset_index(drop=True)

    hasil['support_final'] = support_final
    hasil['confidence_final'] = confidence_final
    hasil['rules_terpilih'] = rules_terpilih
    hasil['rules_final_semua'] = rules_final_semua

    return hasil


# ============================================================
# PLOTTING — untuk halaman EDA (interaktif, Plotly)
# ============================================================
# Grafik-grafik di bawah ini sengaja dibuat interaktif (bukan gambar statis)
# dan mengikuti kaidah visualisasi data yang baik:
#   1. Akurat        -> sumbu nilai selalu mulai dari 0 untuk variabel yang
#                        memang tidak bisa negatif (konsentrasi, curah hujan, dst).
#   2. Jelas          -> setiap chart selalu ada judul, label sumbu + satuan,
#                        dan legenda; satu chart cuma bawa satu pesan utama.
#   3. Tipe tepat     -> tren waktu = line chart, distribusi = boxplot,
#                        korelasi = heatmap.
#   4. Minim chart junk -> tanpa efek 3D/gradient, gridline tipis, tanpa
#                        dekorasi yang tidak menambah informasi.
#   5. Warna bermakna -> kategori ISPU pakai warna yang SAMA persis dengan
#                        badge di halaman Upload (lihat WARNA_STATUS_ISPU),
#                        dan skala korelasi pakai diverging merah-biru yang
#                        aman untuk buta warna merah-hijau.
#   6. Konteks        -> tren PM2.5 diberi pita ambang batas kategori ISPU
#                        sebagai baseline, plus penanda titik lonjakan.
#   7. Interaktif     -> tooltip hover nilai presisi, bisa zoom/pan.

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

KOLOM_LABEL = {
    'pm_sepuluh':        'PM10 (µg/m³)',
    'pm_duakomalima':    'PM2.5 (µg/m³)',
    'sulfur_dioksida':   'SO2 (µg/m³)',
    'karbon_monoksida':  'CO (µg/m³)',
    'ozon':              'O3 (µg/m³)',
    'nitrogen_dioksida': 'NO2 (µg/m³)',
    'temp_air':          'Suhu Udara (°C)',
    'hum_air':           'Kelembaban (%)',
    'ff':                'Kecepatan Angin (m/s)',
    'rain':              'Curah Hujan (mm)',
}

# Sama persis dengan --status-baik dkk di static/css/style.css supaya
# warna kategori ISPU konsisten di seluruh halaman.
WARNA_STATUS_ISPU = {
    'Baik':                '#22c55e',
    'Sedang':              '#eab308',
    'Tidak Sehat':         '#f97316',
    'Sangat Tidak Sehat':  '#ef4444',
    'Berbahaya':           '#7f1d1d',
}

PLOTLY_FONT = dict(family="Inter, sans-serif", size=12, color="#33453f")
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def _layout_dasar(judul, tinggi=380, showlegend=False):
    return dict(
        title=dict(
            text=judul,
            font=dict(family="Space Grotesk, sans-serif", size=15, color="#0f1e1c"),
            x=0.01, xanchor="left",
        ),
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=tinggi,
        margin=dict(l=55, r=30, t=55, b=45),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
    )


def _fig_to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)


def _deteksi_lonjakan_global(s):
    """Titik di atas Q3 + 1.5xIQR (ambang IQR global, bukan rolling) — dipakai
    untuk menandai lonjakan pada grafik tren, bukan untuk mengubah data."""
    valid = s.dropna()
    if len(valid) < 4:
        return pd.Series(False, index=s.index)
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    batas_atas = q3 + 1.5 * iqr
    return (s > batas_atas).fillna(False)


def plot_tren_pm25(df, stasiun):
    """Line chart tren PM2.5 harian, dengan pita ambang batas kategori ISPU
    sebagai konteks/baseline, dan penanda titik lonjakan (outlier IQR)."""
    if 'pm_duakomalima' not in df.columns:
        return None

    s = df['pm_duakomalima']
    lonjakan = _deteksi_lonjakan_global(s)
    batas_kategori = [15.5, 55.4, 150.4, 250.4]
    label_kategori = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    y_maks = max(float(s.max(skipna=True) or 0) * 1.08, batas_kategori[-1] + 40)

    fig = go.Figure()

    # Pita ambang batas kategori ISPU sebagai konteks (band + legenda).
    # PENTING: layer="below" wajib di-set eksplisit, karena default Plotly
    # untuk add_hrect adalah "above" (di ATAS trace data) — kalau tidak
    # di-set, pita warna akan menutupi/menyamarkan garis tren di baliknya.
    batas_bawah = 0
    for i, nama in enumerate(label_kategori):
        atas = batas_kategori[i] if i < len(batas_kategori) else y_maks
        fig.add_hrect(
            y0=batas_bawah, y1=atas,
            fillcolor=WARNA_STATUS_ISPU[nama], opacity=0.10, line_width=0,
            layer="below",
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=WARNA_STATUS_ISPU[nama], symbol="square"),
            name=nama, hoverinfo="skip",
        ))
        batas_bawah = atas

    # Garis tren utama — dibuat tegas & gelap supaya kontras dengan pita
    # warna kategori di belakangnya, dan ditaruh paling akhir supaya
    # tergambar paling atas.
    fig.add_trace(go.Scatter(
        x=df['periode'], y=s, mode="lines", name="PM2.5 harian",
        line=dict(color="#0b3d33", width=2.2, shape="linear"),
        connectgaps=False,
        hovertemplate="%{x|%d %b %Y}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        showlegend=True,
    ))

    # Penanda lonjakan
    if lonjakan.any():
        fig.add_trace(go.Scatter(
            x=df.loc[lonjakan, 'periode'], y=s[lonjakan], mode="markers",
            name="Lonjakan (> Q3 + 1.5×IQR)",
            marker=dict(symbol="triangle-up", size=9, color="#ef4444",
                        line=dict(width=1, color="#7f1d1d")),
            hovertemplate="Lonjakan<br>%{x|%d %b %Y}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        ))

    layout = _layout_dasar(f"Tren Harian PM2.5 — {stasiun}", tinggi=380, showlegend=True)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero", range=[0, y_maks],
                      gridcolor="#eaf3ef", zeroline=False)
    fig.update_xaxes(title_text="Periode", gridcolor="#eaf3ef", showgrid=False)
    return _fig_to_div(fig)


def plot_tren_variabel_stasiun(df, stasiun):
    """Versi multi-variabel dari tren PM2.5 satu stasiun: satu garis, tapi
    variabelnya bisa diganti lewat dropdown (polutan maupun cuaca) — dipakai
    di halaman detail per stasiun supaya satu grafik saja bisa menjawab pola
    dari waktu ke waktu untuk variabel apa pun, bukan cuma PM2.5."""
    variabel_urut = ['pm_duakomalima', 'pm_sepuluh', 'sulfur_dioksida',
                      'karbon_monoksida', 'ozon', 'nitrogen_dioksida',
                      'temp_air', 'hum_air', 'ff', 'rain']
    variabel_urut = [v for v in variabel_urut if v in df.columns and not df[v].dropna().empty]
    if not variabel_urut:
        return None

    fig = go.Figure()
    for idx, var in enumerate(variabel_urut):
        fig.add_trace(go.Scatter(
            x=df['periode'], y=df[var], mode="lines",
            name=KOLOM_LABEL[var].split(' (')[0],
            line=dict(color="#0e6fc9", width=2.0),
            visible=(idx == 0), showlegend=True,
            hovertemplate="%{x|%d %b %Y}<br>" + KOLOM_LABEL[var] + ": %{y:.1f}<extra></extra>",
        ))

    buttons = []
    for idx, var in enumerate(variabel_urut):
        vis = [i == idx for i in range(len(variabel_urut))]
        buttons.append(dict(
            label=KOLOM_LABEL[var].split(' (')[0], method="update",
            args=[{"visible": vis}, {"yaxis.title.text": KOLOM_LABEL[var]}],
        ))

    layout = _layout_dasar(f"Tren Harian per Variabel — {stasiun}", tinggi=380, showlegend=True)
    layout['updatemenus'] = [dict(
        type="dropdown", direction="down", x=1.0, xanchor="right",
        y=1.22, yanchor="top", buttons=buttons, showactive=True,
        bgcolor="#ffffff", bordercolor="#cfe3dc",
    )]
    fig.update_layout(**layout)
    fig.update_yaxes(title_text=KOLOM_LABEL[variabel_urut[0]], rangemode="tozero", gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Periode", gridcolor="#eaf3ef", showgrid=False)
    fig.add_annotation(
        text="Pilih variabel ▲", x=1.0, xanchor="right", y=1.30, yanchor="top",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)


def _bungkus_kartu_variabel(potongan_html):
    """Bungkus daftar div grafik (satu per variabel) jadi grid kartu
    responsif — tiap variabel dapat kotaknya sendiri (border + padding)
    supaya batas antar-parameter jelas dan tidak bikin bingung."""
    kartu = "".join(
        f'<div style="background:#fff; border:1px solid #e4e8f0; border-radius:12px; '
        f'padding:8px 6px 2px; overflow:hidden;">{html}</div>'
        for html in potongan_html
    )
    return (
        '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); '
        f'gap:16px;">{kartu}</div>'
    )


def plot_missing_value(df, stasiun):
    """Satu kartu grafik mandiri per kolom yang punya missing value, ditata
    dalam grid responsif (per variabel, per kontainer). Tiap kartu punya
    legend sendiri di sisi kanan; garis putus merah menandai posisi waktu
    yang datanya hilang."""
    kolom_missing = [c for c in KOLOM_NUMERIK if c in df.columns and df[c].isnull().any()]
    if not kolom_missing:
        return None

    potongan = []
    for col in kolom_missing:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['periode'], y=df[col], mode="lines",
            line=dict(color="#0e6fc9", width=1.3), name="Data",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
        ))

        posisi_missing = df.loc[df[col].isna(), 'periode']
        for x in posisi_missing:
            fig.add_vline(x=x, line=dict(color="#ef4444", dash="dot", width=1), opacity=0.45)

        # Trace "hantu" (tidak terlihat, garis kosong) khusus supaya garis
        # putus merah (dibuat lewat add_vline, yang tidak otomatis masuk
        # legend) tetap muncul di legend sebagai penjelasan warna.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#ef4444", dash="dot", width=1.5),
            name="Titik data hilang", showlegend=True,
        ))

        layout = _layout_dasar(KOLOM_LABEL.get(col, col), tinggi=300, showlegend=True)
        # Legend ditaruh vertikal di sisi KANAN kartu (bukan horizontal di
        # atas), supaya tidak menutupi judul/grafik dan konsisten tiap kartu
        # selalu punya legend sendiri (tidak berbagi lewat legendgroup).
        layout["legend"] = dict(orientation="v", yanchor="middle", y=0.5,
                                 xanchor="left", x=1.02, font=dict(size=11))
        layout["margin"] = dict(l=55, r=135, t=45, b=45)
        fig.update_layout(**layout)
        fig.update_yaxes(gridcolor="#eaf3ef", zeroline=False,
                          showline=True, linewidth=1, linecolor="#d8dee7", mirror=True)
        fig.update_xaxes(title_text="Periode", tickangle=-45, gridcolor="#eaf3ef",
                          showline=True, linewidth=1, linecolor="#d8dee7", mirror=True)

        potongan.append(_fig_to_div(fig))

    return _bungkus_kartu_variabel(potongan)


def plot_screening_outlier(df, stasiun):
    """Satu kartu grafik mandiri per parameter untuk screening cepat nilai
    di luar ambang domain (batas fisis), ditata dalam grid responsif (per
    variabel, per kontainer). Titik yang melanggar ambang domain ditandai
    merah, garis nilai normal biru — tiap kartu punya legend sendiri di
    sisi kanan yang menjelaskan kedua warna itu."""
    kolom_ada = [c for c in KOLOM_NUMERIK if c in df.columns]

    potongan = []
    for col in kolom_ada:
        valid = df[col].notna()
        batas_bawah, batas_atas = BATAS_DOMAIN.get(col, (-np.inf, np.inf))
        is_outlier = valid & ((df[col] < batas_bawah) | (df[col] > batas_atas))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.loc[valid, 'periode'], y=df.loc[valid, col], mode="lines",
            line=dict(color="#0e6fc9", width=1), opacity=0.6,
            name="Nilai normal",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df.loc[is_outlier, 'periode'], y=df.loc[is_outlier, col], mode="markers",
            marker=dict(color="#ef4444", size=6, symbol="circle",
                        line=dict(color="white", width=0.5)),
            name="Di luar ambang domain",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
        ))

        rangemode = "tozero" if col != "temp_air" else "normal"
        layout = _layout_dasar(KOLOM_LABEL.get(col, col), tinggi=270, showlegend=True)
        # Legend vertikal di sisi KANAN, bukan horizontal di atas grid subplot
        # — sekarang tiap kartu figure sendiri jadi legend "Nilai normal" dan
        # "Di luar ambang domain" (bulatan merah) selalu tampil per kartu.
        layout["legend"] = dict(orientation="v", yanchor="middle", y=0.5,
                                 xanchor="left", x=1.02, font=dict(size=11))
        layout["margin"] = dict(l=50, r=150, t=45, b=35)
        fig.update_layout(**layout)
        fig.update_yaxes(rangemode=rangemode, gridcolor="#eaf3ef")
        fig.update_xaxes(showticklabels=False, gridcolor="#eaf3ef")

        potongan.append(_fig_to_div(fig))

    return _bungkus_kartu_variabel(potongan)


def plot_korelasi(df, stasiun):
    """Heatmap korelasi Pearson antar parameter numerik. Skala diverging
    merah-biru (bukan merah-hijau) supaya tetap terbaca oleh pembaca dengan
    buta warna merah-hijau, jenis paling umum."""
    kolom_ada = [c for c in KOLOM_NUMERIK if c in df.columns]
    corr = df[kolom_ada].corr()
    label = [KOLOM_LABEL.get(c, c) for c in kolom_ada]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=label, y=label,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title=dict(text="r", side="right"), thickness=14),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
    ))

    layout = _layout_dasar(f"Korelasi Antar Variabel — {stasiun}", tinggi=520)
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-40, gridcolor="#eaf3ef")
    fig.update_yaxes(autorange="reversed", gridcolor="#eaf3ef")
    return _fig_to_div(fig)


# ============================================================
# PLOTTING — RINGKASAN SELURUH STASIUN DKI JAKARTA
# ============================================================
# Bagian ini khusus untuk kebutuhan "orang awam": ringkas, satu pesan per
# grafik, dan pakai warna kategori ISPU yang sama dengan seluruh halaman
# lain supaya tidak perlu menebak artinya.

# Palet warna per-stasiun (dipakai konsisten di bar & line overview) —
# sengaja BUKAN warna kategori ISPU (hijau/kuning/oranye/merah) supaya
# tidak tertukar maknanya dengan warna status; ini murni pembeda stasiun.
WARNA_STASIUN = ["#0e6fc9", "#17a673", "#f7931e", "#a855f7", "#ef4444"]


def _hex_ke_rgba(hex_color, alpha=0.18):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _hex_ke_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_ke_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, round(c))) for c in rgb])


# Gradasi warna RELATIF (bukan ambang absolut ISPU) — dipakai untuk kartu
# ranking stasiun. Kenapa relatif? Karena ambang absolut ISPU rentangnya
# jauh (mis. kategori "Tidak Sehat" itu 55.4–150.4), sedangkan nilai
# rata-rata antar 5 stasiun biasanya berdekatan (mis. 67–76) — kalau
# dipetakan ke skala absolut, warnanya jadi keliatan nyaris sama semua.
# Dengan skala relatif (dipetakan dari nilai TERENDAH ke TERTINGGI di antara
# stasiun yang sedang dibandingkan), stasiun paling bersih SELALU jadi warna
# paling terang dan yang paling buruk SELALU paling gelap, apa pun kategori
# ISPU-nya — jadi urutan ranking langsung kebaca dari gelap-terangnya kartu.
_GRADASI_RELATIF_STOPS = [
    (0.0, "#fef9c3"),  # paling bersih di antara yang dibandingkan
    (0.5, "#f97316"),
    (1.0, "#7f1d1d"),  # paling buruk di antara yang dibandingkan
]


def _warna_gradasi_relatif(t):
    """t = posisi nilai antara 0 (paling rendah/bersih) dan 1 (paling
    tinggi/buruk) di antara sekumpulan nilai yang dibandingkan."""
    stops = _GRADASI_RELATIF_STOPS
    t = max(0.0, min(1.0, t))
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x0 <= t <= x1:
            frac = (t - x0) / (x1 - x0) if x1 > x0 else 0.0
            rgb0, rgb1 = _hex_ke_rgb(c0), _hex_ke_rgb(c1)
            rgb = [rgb0[i] + (rgb1[i] - rgb0[i]) * frac for i in range(3)]
            return _rgb_ke_hex(rgb)
    return stops[-1][1]


IKON_PARAMETER = {
    'pm_sepuluh':        '🌫️',
    'pm_duakomalima':    '🌫️',
    'sulfur_dioksida':   '🏭',
    'karbon_monoksida':  '🚗',
    'ozon':              '☀️',
    'nitrogen_dioksida': '🚦',
    'temp_air':          '🌡️',
    'hum_air':           '💧',
    'ff':                '💨',
    'rain':              '🌧️',
}


def hitung_ringkasan_semua_parameter(gabungan_perstasiun, daftar_stasiun):
    """Ringkasan gabungan (rata-rata dari semua stasiun, curah hujan
    dijumlah) untuk SEMUA parameter — PM10, SO2, CO, O3, NO2, suhu,
    kelembaban, angin, hujan — bukan cuma PM2.5. Dipakai supaya overview
    tidak melulu menonjolkan PM2.5; polutan lain & cuaca juga langsung
    kelihatan tanpa harus buka dropdown di grafik lain."""
    hasil = []
    for kolom in KOLOM_NUMERIK:
        if kolom == 'pm_duakomalima':
            continue  # sudah ditonjolkan lewat KPI & ranking di atas
        nilai_semua = []
        for stasiun in daftar_stasiun:
            df = gabungan_perstasiun[stasiun]
            if kolom not in df.columns or df[kolom].dropna().empty:
                continue
            if kolom == 'rain':
                nilai_semua.append(float(df[kolom].sum(skipna=True)))
            else:
                nilai_semua.append(float(df[kolom].mean(skipna=True)))
        if not nilai_semua:
            continue
        agregat = sum(nilai_semua) / len(nilai_semua)
        label = KOLOM_LABEL.get(kolom, kolom)
        if '(' in label:
            nama, satuan = label.split(' (', 1)
            satuan = satuan.rstrip(')')
        else:
            nama, satuan = label, ''
        hasil.append({
            'ikon': IKON_PARAMETER.get(kolom, '📊'),
            'label': nama,
            'nilai': round(agregat, 1),
            'satuan': satuan,
            'jenis': 'Polutan' if kolom in PARAMETER_ISPU else 'Cuaca',
        })
    return hasil


BADGE_CLASS_KATEGORI = {
    'Baik':               'badge-baik',
    'Sedang':             'badge-sedang',
    'Tidak Sehat':        'badge-tidaksehat',
    'Sangat Tidak Sehat': 'badge-sangattidaksehat',
    'Berbahaya':          'badge-berbahaya',
}


def ringkasan_kategori_semua_stasiun(gabungan_perstasiun, daftar_stasiun):
    """Ringkasan super-sederhana per stasiun (untuk kartu, bukan grafik):
    rata-rata PM2.5 + kategori ISPU-nya, supaya orang awam langsung paham
    'stasiun ini kondisinya begini' tanpa perlu baca grafik dulu."""
    hasil = []
    for stasiun in daftar_stasiun:
        df = gabungan_perstasiun[stasiun]
        if 'pm_duakomalima' not in df.columns or df['pm_duakomalima'].dropna().empty:
            hasil.append({
                'stasiun': stasiun, 'rata_rata': None, 'kategori': '-',
                'badge_class': 'badge-default', 'warna': '#9db3ab',
            })
            continue
        rata = float(df['pm_duakomalima'].mean(skipna=True))
        kategori = kategori_ispu(rata, 'pm_duakomalima')
        hasil.append({
            'stasiun': stasiun,
            'rata_rata': round(rata, 1),
            'kategori': kategori,
            'badge_class': BADGE_CLASS_KATEGORI.get(kategori, 'badge-default'),
            'warna': WARNA_STATUS_ISPU.get(kategori, '#9db3ab'),
        })
    return hasil


def _kategori_bersih(df):
    """Ambil kolom 'kategori' dan normalisasi ke label baku ('Baik', 'Sedang', dst),
    supaya tahan terhadap variasi format dari CSV asli (huruf besar semua, spasi
    ekstra, dll). Nilai yang tidak dikenali (mis. 'TIDAK ADA DATA') dibuang, bukan
    bikin error."""
    if 'kategori' not in df.columns:
        return pd.Series(dtype='object')
    peta = {
        'BAIK': 'Baik',
        'SEDANG': 'Sedang',
        'TIDAK SEHAT': 'Tidak Sehat',
        'SANGAT TIDAK SEHAT': 'Sangat Tidak Sehat',
        'BERBAHAYA': 'Berbahaya',
    }
    bersih = df['kategori'].dropna().astype(str).str.strip().str.upper().map(peta)
    return bersih.dropna()


def hitung_kpi_keseluruhan(gabungan_perstasiun, daftar_stasiun):
    """KPI ringkas gabungan seluruh stasiun (untuk kartu paling atas halaman
    overview). Sengaja dibatasi ke beberapa angka paling penting saja —
    jangan kebanyakan KPI, nanti orang awam malah bingung."""
    semua_periode, semua_kategori = [], []
    rasio_parameter = {p: [] for p in PARAMETER_ISPU}

    for stasiun in daftar_stasiun:
        df = gabungan_perstasiun[stasiun]
        if 'periode' in df.columns:
            semua_periode.append(df['periode'])
        if 'kategori' in df.columns:
            semua_kategori.append(_kategori_bersih(df))
        for p in PARAMETER_ISPU:
            if p in df.columns and not df[p].dropna().empty:
                rata = float(df[p].mean(skipna=True))
                rasio_parameter[p].append(rata / BATAS_ISPU[p][0])

    periode_awal = periode_akhir = None
    if semua_periode:
        gab = pd.concat(semua_periode)
        periode_awal = gab.min()
        periode_akhir = gab.max()

    pct_kategori = {}
    kategori_dominan, pct_dominan = '-', 0
    if semua_kategori:
        gab_kat = pd.concat(semua_kategori)
        total = len(gab_kat)
        if total > 0:
            counts = gab_kat.value_counts()
            for k in WARNA_STATUS_ISPU:
                pct_kategori[k] = round(counts.get(k, 0) / total * 100, 1)
            kategori_dominan = counts.idxmax()
            pct_dominan = round(counts.max() / total * 100, 1)

    # "Polutan dominan" dihitung dari rata-rata rasio (nilai rata-rata dibagi
    # ambang batas kategori Baik-nya) — supaya adil membandingkan parameter
    # dengan satuan & skala yang jauh berbeda (µg/m³ vs mg/m³, dst).
    polutan_dominan, rasio_maks = '-', 0
    for p, vals in rasio_parameter.items():
        if vals:
            rata_rasio = sum(vals) / len(vals)
            if rata_rasio > rasio_maks:
                rasio_maks = rata_rasio
                polutan_dominan = KOLOM_LABEL[p].split(' (')[0]

    return {
        'total_stasiun': len(daftar_stasiun),
        'periode_awal': periode_awal.strftime('%b %Y') if periode_awal is not None else '-',
        'periode_akhir': periode_akhir.strftime('%b %Y') if periode_akhir is not None else '-',
        'polutan_dominan': polutan_dominan,
        'kategori_dominan': kategori_dominan,
        'pct_dominan': pct_dominan,
        'badge_dominan': BADGE_CLASS_KATEGORI.get(kategori_dominan, 'badge-default'),
        'warna_dominan': WARNA_STATUS_ISPU.get(kategori_dominan, '#9db3ab'),
        'pct_kategori': pct_kategori,
    }


def ranking_stasiun(gabungan_perstasiun, daftar_stasiun):
    """Ranking sederhana stasiun berdasarkan rata-rata PM2.5 (parameter yang
    paling umum jadi acuan). Rank 1 = kualitas udara relatif paling buruk
    (rata-rata PM2.5 tertinggi), diberi medali untuk 3 besar."""
    medali = {1: '🥇', 2: '🥈', 3: '🥉'}
    ringkasan = ringkasan_kategori_semua_stasiun(gabungan_perstasiun, daftar_stasiun)
    ringkasan = [r for r in ringkasan if r['rata_rata'] is not None]
    ringkasan.sort(key=lambda r: r['rata_rata'], reverse=True)

    nilai_semua = [r['rata_rata'] for r in ringkasan]
    v_min, v_max = min(nilai_semua), max(nilai_semua)
    rentang = v_max - v_min

    for i, r in enumerate(ringkasan, start=1):
        r['peringkat'] = i
        r['medali'] = medali.get(i, '')
        # Kartu ranking pakai gradasi warna RELATIF: dibandingkan cuma
        # antar stasiun yang tampil di ranking ini, bukan skala absolut
        # ISPU — supaya makin gelap kartunya = makin buruk *dibanding
        # stasiun lain*, dan makin terang = makin bersih, meski semuanya
        # kebetulan masuk kategori ISPU yang sama. Label kategori (badge)
        # tetap sama seperti biasa.
        t = (r['rata_rata'] - v_min) / rentang if rentang > 0 else 0.5
        r['warna'] = _warna_gradasi_relatif(t)
    return ringkasan


def plot_perbandingan_indikator_stasiun(gabungan_perstasiun, daftar_stasiun):
    """Bar chart rata-rata per stasiun, BISA GANTI indikator (PM2.5, PM10,
    SO2, CO, O3, NO2) lewat dropdown di dalam grafik — jadi satu grafik
    interaktif, bukan 6 grafik terpisah yang bikin halaman penuh."""
    label_singkat = [s.split(' ', 1)[-1].title() for s in daftar_stasiun]
    fig = go.Figure()

    for idx, param in enumerate(PARAMETER_ISPU):
        nilai, warna, teks = [], [], []
        for stasiun in daftar_stasiun:
            df = gabungan_perstasiun[stasiun]
            if param in df.columns and not df[param].dropna().empty:
                rata = float(df[param].mean(skipna=True))
                kat = kategori_ispu(rata, param)
                nilai.append(rata)
                warna.append(WARNA_STATUS_ISPU[kat])
                teks.append(f"{rata:.1f}")
            else:
                nilai.append(0)
                warna.append('#cbd5d1')
                teks.append('-')

        fig.add_trace(go.Bar(
            x=label_singkat, y=nilai, marker_color=warna,
            text=teks, textposition="outside",
            visible=(idx == 0), showlegend=False,
            hovertemplate="%{x}<br>" + KOLOM_LABEL[param] + ": %{y:.1f}<extra></extra>",
        ))

    n_param = len(PARAMETER_ISPU)

    # Legenda warna kategori ISPU (dummy trace, tidak menggambar apa pun di
    # plot). WAJIB ada karena warna bar di grafik ini mengikuti kategori
    # ISPU (Baik/Sedang/dst), sedangkan sumbu-X-nya nama stasiun — tanpa
    # legenda ini pembaca tidak akan tahu warna itu artinya apa. Selalu
    # tampil apa pun parameter yang dipilih lewat dropdown, karena arti
    # warnanya sama persis di semua parameter.
    urutan_legenda = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    for kat in urutan_legenda:
        fig.add_trace(go.Bar(
            x=[None], y=[None], marker_color=WARNA_STATUS_ISPU[kat],
            name=kat, showlegend=True, hoverinfo="skip", visible=True,
        ))

    total_trace = n_param + len(urutan_legenda)
    buttons = []
    for idx, param in enumerate(PARAMETER_ISPU):
        vis = [False] * total_trace
        vis[idx] = True
        for j in range(n_param, total_trace):
            vis[j] = True  # trace legenda selalu tampil
        buttons.append(dict(
            label=KOLOM_LABEL[param].split(' (')[0], method="update",
            args=[{"visible": vis},
                  {"yaxis.title.text": KOLOM_LABEL[param]}],
        ))

    layout = _layout_dasar("Rata-rata Polutan per Stasiun — DKI Jakarta", tinggi=400, showlegend=True)
    layout['updatemenus'] = [dict(
        type="dropdown", direction="down", x=1.0, xanchor="right",
        y=1.22, yanchor="top", buttons=buttons, showactive=True,
        bgcolor="#ffffff", bordercolor="#cfe3dc",
    )]
    fig.update_layout(**layout)
    fig.update_yaxes(title_text=KOLOM_LABEL[PARAMETER_ISPU[0]], rangemode="tozero", gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Stasiun", gridcolor="#eaf3ef", showgrid=False)
    fig.add_annotation(
        text="Pilih parameter ▲", x=1.0, xanchor="right", y=1.30, yanchor="top",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)


def plot_distribusi_kategori_keseluruhan(gabungan_perstasiun, daftar_stasiun):
    """Bar chart: dari SEMUA hari & SEMUA stasiun, berapa persen yang masuk
    tiap kategori ISPU? Ini jawaban paling gampang dipahami orang awam
    untuk pertanyaan 'secara umum udaranya gimana sih'."""
    semua_kategori = []
    for stasiun in daftar_stasiun:
        df = gabungan_perstasiun[stasiun]
        if 'kategori' in df.columns:
            semua_kategori.append(_kategori_bersih(df))
    if not semua_kategori:
        return None
    gab = pd.concat(semua_kategori)
    total = len(gab)
    if total == 0:
        return None
    counts = gab.value_counts()

    urutan = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    urutan = [k for k in urutan if k in counts.index]
    if not urutan:
        return None
    persen = [round(counts[k] / total * 100, 1) for k in urutan]
    warna = [WARNA_STATUS_ISPU[k] for k in urutan]
    teks = [f"{p:.1f}%" for p in persen]

    fig = go.Figure(go.Bar(
        x=urutan, y=persen, marker_color=warna, text=teks, textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}%% dari seluruh hari<extra></extra>",
    ))
    layout = _layout_dasar("Sebaran Kategori Kualitas Udara — Seluruh Stasiun", tinggi=360)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Persentase hari (%)", rangemode="tozero",
                      range=[0, max(persen) * 1.2], gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Kategori ISPU", gridcolor="#eaf3ef", showgrid=False)
    return _fig_to_div(fig)


def tabel_top_hari_terburuk(gabungan_perstasiun, daftar_stasiun, n=10):
    """Top-N hari dengan PM2.5 tertinggi, lintas semua stasiun — lebih
    'nyantol' buat orang awam daripada angka rata-rata/median/std doang."""
    baris = []
    for stasiun in daftar_stasiun:
        df = gabungan_perstasiun[stasiun]
        if 'pm_duakomalima' not in df.columns or 'periode' not in df.columns:
            continue
        tmp = df[['periode', 'pm_duakomalima']].dropna()
        for _, row in tmp.iterrows():
            kat = kategori_ispu(row['pm_duakomalima'], 'pm_duakomalima')
            baris.append({
                'tanggal': row['periode'].strftime('%d %b %Y'),
                'stasiun': stasiun,
                'nilai': round(float(row['pm_duakomalima']), 1),
                'kategori': kat,
                'badge_class': BADGE_CLASS_KATEGORI.get(kat, 'badge-default'),
            })
    if not baris:
        return []
    baris.sort(key=lambda r: r['nilai'], reverse=True)
    return baris[:n]


def plot_tren_semua_stasiun(gabungan_perstasiun, daftar_stasiun):
    """Line chart tren PM2.5 harian, satu garis per stasiun, dalam satu
    grafik — supaya pembaca bisa langsung melihat stasiun mana yang
    polanya beda/lebih tinggi dari yang lain sepanjang waktu."""
    fig = go.Figure()
    y_maks = 0
    ada_data = False

    for i, stasiun in enumerate(daftar_stasiun):
        df = gabungan_perstasiun[stasiun]
        if 'pm_duakomalima' not in df.columns:
            continue
        s = df['pm_duakomalima']
        if s.dropna().empty:
            continue
        ada_data = True
        y_maks = max(y_maks, float(s.max(skipna=True) or 0))
        warna = WARNA_STASIUN[i % len(WARNA_STASIUN)]
        label = stasiun.split(' ', 1)[-1].title()
        fig.add_trace(go.Scatter(
            x=df['periode'], y=s, mode="lines", name=label,
            line=dict(color=warna, width=1.7),
            hovertemplate=f"{label}<br>" + "%{x|%d %b %Y}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        ))

    if not ada_data:
        return None

    layout = _layout_dasar("Tren PM2.5 — Perbandingan Semua Stasiun DKI Jakarta",
                            tinggi=420, showlegend=True)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero",
                      range=[0, y_maks * 1.08], gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Periode", gridcolor="#eaf3ef", showgrid=False)
    return _fig_to_div(fig)


def plot_tren_semua_variabel_stasiun(gabungan_perstasiun, daftar_stasiun):
    """Versi multi-variabel dari tren PM2.5 semua stasiun: satu garis per
    stasiun, tapi variabelnya bisa diganti lewat dropdown (PM2.5, PM10, SO2,
    CO, O3, NO2, suhu, kelembaban, angin, hujan) — supaya perbandingan antar
    stasiun tidak cuma terbatas ke PM2.5."""
    variabel_urut = ['pm_duakomalima', 'pm_sepuluh', 'sulfur_dioksida',
                      'karbon_monoksida', 'ozon', 'nitrogen_dioksida',
                      'temp_air', 'hum_air', 'ff', 'rain']
    variabel_urut = [v for v in variabel_urut if v in KOLOM_NUMERIK]
    if not variabel_urut:
        return None

    fig = go.Figure()
    trace_ranges = []
    ada_data = False

    for v_idx, var in enumerate(variabel_urut):
        start = len(fig.data)
        for i, stasiun in enumerate(daftar_stasiun):
            df = gabungan_perstasiun[stasiun]
            label = stasiun.split(' ', 1)[-1].title()
            warna = WARNA_STASIUN[i % len(WARNA_STASIUN)]
            vis0 = (v_idx == 0)
            if var in df.columns and not df[var].dropna().empty:
                ada_data = True
                fig.add_trace(go.Scatter(
                    x=df['periode'], y=df[var], mode="lines", name=label,
                    line=dict(color=warna, width=1.7),
                    visible=vis0, showlegend=vis0,
                    hovertemplate=f"{label}<br>" + "%{x|%d %b %Y}<br>"
                                  + KOLOM_LABEL.get(var, var) + ": %{y:.1f}<extra></extra>",
                ))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=label,
                                          line=dict(color=warna, width=1.7),
                                          visible=vis0, showlegend=vis0))
        trace_ranges.append((start, len(fig.data)))

    if not ada_data:
        return None

    total = len(fig.data)
    buttons = []
    for idx, var in enumerate(variabel_urut):
        start, end = trace_ranges[idx]
        vis = [False] * total
        for t in range(start, end):
            vis[t] = True
        buttons.append(dict(
            label=KOLOM_LABEL[var].split(' (')[0], method="update",
            args=[{"visible": vis, "showlegend": vis},
                  {"yaxis.title.text": KOLOM_LABEL[var]}],
        ))

    layout = _layout_dasar("Tren Harian — Perbandingan Semua Stasiun", tinggi=420, showlegend=True)
    layout['updatemenus'] = [dict(
        type="dropdown", direction="down", x=1.0, xanchor="right",
        y=1.22, yanchor="top", buttons=buttons, showactive=True,
        bgcolor="#ffffff", bordercolor="#cfe3dc",
    )]
    fig.update_layout(**layout)
    fig.update_yaxes(title_text=KOLOM_LABEL[variabel_urut[0]], rangemode="tozero", gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Periode", gridcolor="#eaf3ef", showgrid=False)
    fig.add_annotation(
        text="Pilih variabel ▲", x=1.0, xanchor="right", y=1.30, yanchor="top",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)


def plot_boxplot_semua_stasiun(gabungan_perstasiun, daftar_stasiun):
    """Boxplot PM2.5, satu kotak per stasiun, berdampingan dalam satu
    grafik — untuk membandingkan sebaran (bukan cuma rata-rata) tiap
    stasiun: mana yang datanya lebih stabil dan mana yang lebih liar."""
    fig = go.Figure()
    ada_data = False
    for i, stasiun in enumerate(daftar_stasiun):
        df = gabungan_perstasiun[stasiun]
        if 'pm_duakomalima' not in df.columns or df['pm_duakomalima'].dropna().empty:
            continue
        ada_data = True
        warna = WARNA_STASIUN[i % len(WARNA_STASIUN)]
        label = stasiun.split(' ', 1)[-1].title()
        fig.add_trace(go.Box(
            y=df['pm_duakomalima'].dropna(), name=label,
            marker_color=warna, line_color=warna,
            fillcolor=_hex_ke_rgba(warna, 0.18),
            boxpoints="outliers", marker=dict(size=3, opacity=0.6),
            showlegend=False,
            hovertemplate=f"{label}<br>Q3: " + "%{q3:.1f}<br>Median: %{median:.1f}<br>Q1: %{q1:.1f}<extra></extra>",
        ))

    if not ada_data:
        return None

    layout = _layout_dasar("Sebaran PM2.5 per Stasiun — DKI Jakarta", tinggi=380)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero", gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Stasiun", gridcolor="#eaf3ef", showgrid=False)
    return _fig_to_div(fig)


# ============================================================
# PLOTTING — ANALISIS PER STASIUN (halaman detail, 1 stasiun terpilih)
# ============================================================

def hitung_kpi_stasiun(df, stasiun):
    """KPI ringkas untuk satu stasiun terpilih: rata-rata PM2.5 & PM10, plus
    persentase hari Baik vs hari yang perlu diwaspadai (Tidak Sehat ke atas)."""
    kpi = {'rata_pm25': None, 'rata_pm10': None, 'pct_baik': None, 'pct_waspada': None}
    if 'pm_duakomalima' in df.columns and not df['pm_duakomalima'].dropna().empty:
        kpi['rata_pm25'] = round(float(df['pm_duakomalima'].mean(skipna=True)), 1)
    if 'pm_sepuluh' in df.columns and not df['pm_sepuluh'].dropna().empty:
        kpi['rata_pm10'] = round(float(df['pm_sepuluh'].mean(skipna=True)), 1)
    kat = _kategori_bersih(df)
    if not kat.empty:
        total = len(kat)
        kpi['pct_baik'] = round((kat == 'Baik').sum() / total * 100, 1)
        waspada = kat.isin(['Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']).sum()
        kpi['pct_waspada'] = round(waspada / total * 100, 1)
    return kpi


def plot_distribusi_kategori_stasiun(df, stasiun):
    """Sama seperti komposisi kategori keseluruhan, tapi untuk satu stasiun
    saja — 'bagaimana kondisi udara di stasiun ini selama periode data?'
    Dipakai JUMLAH HARI (bukan persentase) supaya lebih gampang dibayangkan
    orang awam — mis. '547 hari' lebih konkret dibanding '78.3%'."""
    kat = _kategori_bersih(df)
    if kat.empty:
        return None
    counts = kat.value_counts()
    urutan = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    urutan = [k for k in urutan if k in counts.index]
    if not urutan:
        return None
    jumlah = [int(counts[k]) for k in urutan]
    warna = [WARNA_STATUS_ISPU[k] for k in urutan]
    teks = [f"{j} hari" for j in jumlah]

    fig = go.Figure(go.Bar(
        x=urutan, y=jumlah, marker_color=warna, text=teks, textposition="outside",
        showlegend=False,
        hovertemplate="%{x}<br>%{y} hari di stasiun ini<extra></extra>",
    ))
    # Legenda warna kategori (dummy trace per kategori yang muncul), biar
    # warnanya dijelaskan eksplisit, bukan cuma mengandalkan label sumbu-X.
    for k in urutan:
        fig.add_trace(go.Bar(
            x=[None], y=[None], marker_color=WARNA_STATUS_ISPU[k],
            name=k, showlegend=True, hoverinfo="skip",
        ))
    layout = _layout_dasar(f"Sebaran Kategori Kualitas Udara — {stasiun}", tinggi=340, showlegend=True)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Jumlah hari", rangemode="tozero",
                      range=[0, max(jumlah) * 1.2], gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Kategori ISPU", gridcolor="#eaf3ef", showgrid=False)
    return _fig_to_div(fig)


def hitung_profil_polutan_stasiun(df, stasiun):
    """Ringkasan rata-rata tiap polutan dalam SATUAN ASLINYA (µg/m³), bukan
    rasio 'kali lipat ambang batas' — supaya orang awam tinggal baca angka
    dan label kategorinya (Baik/Sedang/dst) langsung, tanpa perlu memahami
    konsep normalisasi/rasio yang membingungkan."""
    hasil = []
    for p in PARAMETER_ISPU:
        if p not in df.columns or df[p].dropna().empty:
            continue
        rata = float(df[p].mean(skipna=True))
        kat = kategori_ispu(rata, p)
        satuan = KOLOM_LABEL[p].split('(')[-1].rstrip(')')
        hasil.append({
            'label': KOLOM_LABEL[p].split(' (')[0],
            'nilai': f"{rata:.1f}",
            'satuan': satuan,
            'kategori': kat,
            'warna': WARNA_STATUS_ISPU[kat],
            'batas_baik': f"{BATAS_ISPU[p][0]:g}",
        })
    return hasil


LABEL_CUACA = {
    # (ikon, label, satuan, warna) — warna dipakai buat --station-color di
    # kartu KPI. Sengaja BUKAN warna kategori ISPU (hijau/kuning/oranye/
    # merah), murni pembeda antar parameter cuaca, tiap parameter beda warna
    # supaya kartunya gampang dibedain sekilas (dulu semua kartu di-hardcode
    # biru, jadi keempatnya keliatan identik).
    'temp_air': ('🌡️', 'Suhu rata-rata', '°C', '#f97316'),
    'hum_air':  ('💧', 'Kelembapan rata-rata', '%', '#0ea5e9'),
    'ff':       ('💨', 'Kecepatan angin rata-rata', 'm/s', '#64748b'),
    'rain':     ('🌧️', 'Total curah hujan', 'mm', '#2563eb'),
}


def hitung_kpi_cuaca_stasiun(df, stasiun):
    """KPI cuaca ringkas untuk satu stasiun — suhu & kelembapan dirata-rata,
    curah hujan dijumlah (karena representasi 'total hujan' lebih bermakna
    daripada rata-rata harian yang kebanyakan nol). Tiap KPI dikasih
    keterangan kualitatif singkat (mis. 'Panas', 'Lembab') supaya angkanya
    tidak berdiri sendiri tanpa konteks bagi pembaca awam."""
    hasil = []
    for kolom, (ikon, label, satuan, warna) in LABEL_CUACA.items():
        if kolom not in df.columns or df[kolom].dropna().empty:
            continue
        if kolom == 'rain':
            nilai = df[kolom].sum(skipna=True)
            n_hari = int(df[kolom].notna().sum())
            rata_harian = nilai / n_hari if n_hari else 0
            # Diberi label kualitatif (sama seperti suhu/kelembapan/angin) supaya
            # tidak berupa teks statistik polos saja — pakai rata-rata harian
            # sebagai acuan kategori, bukan totalnya (karena skala kategori
            # kategori_hujan() memang untuk curah hujan HARIAN).
            keterangan = f"{kategori_hujan(rata_harian)} — rata-rata {rata_harian:.1f} mm/hari selama {n_hari} hari data"
        else:
            nilai = df[kolom].mean(skipna=True)
            if kolom == 'temp_air':
                keterangan = kategori_suhu(nilai)
            elif kolom == 'hum_air':
                keterangan = kategori_kelembaban(nilai)
            elif kolom == 'ff':
                keterangan = kategori_angin(nilai)
            else:
                keterangan = ""
        hasil.append({'ikon': ikon, 'label': label, 'nilai': round(float(nilai), 1),
                      'satuan': satuan, 'keterangan': keterangan, 'warna': warna})
    return hasil


def plot_perbandingan_tahunan_stasiun(df, stasiun):
    """Bar chart perbandingan rata-rata tiap variabel (cuaca maupun polutan)
    antar tahun yang ada di data — mis. 2024 vs 2025. Variabelnya bisa
    diganti lewat dropdown. Kalau data cuma mencakup satu tahun, grafik ini
    disembunyikan karena tidak ada apa pun untuk dibandingkan."""
    if 'periode' not in df.columns:
        return None
    tahun = df['periode'].dt.year
    tahun_unik = sorted(tahun.dropna().unique())
    if len(tahun_unik) < 2:
        return None

    variabel_urut = ['pm_duakomalima', 'pm_sepuluh', 'sulfur_dioksida',
                      'karbon_monoksida', 'ozon', 'nitrogen_dioksida',
                      'temp_air', 'hum_air', 'ff', 'rain']
    variabel_urut = [v for v in variabel_urut if v in df.columns and not df[v].dropna().empty]
    if not variabel_urut:
        return None

    palet_tahun = ['#0e6fc9', '#17a673', '#f7931e', '#7c3aed', '#ef4444']
    warna_tahun = {t: palet_tahun[i % len(palet_tahun)] for i, t in enumerate(tahun_unik)}
    label_tahun = [str(int(t)) for t in tahun_unik]

    fig = go.Figure()
    range_per_var = {}
    for idx, var in enumerate(variabel_urut):
        # Curah hujan dijumlah per tahun (total hujan setahun lebih relevan
        # daripada rata-rata harian); variabel lain pakai rata-rata.
        agregasi = 'sum' if var == 'rain' else 'mean'
        nilai_per_tahun = df.groupby(tahun)[var].agg(agregasi).reindex(tahun_unik)
        # Beri headroom 22% di atas nilai tertinggi supaya label angka di atas
        # bar (text="outside") tidak kepotong oleh batas atas grafik.
        nilai_maks = float(nilai_per_tahun.max(skipna=True) or 0)
        range_per_var[var] = [0, nilai_maks * 1.22 if nilai_maks > 0 else 1]
        fig.add_trace(go.Bar(
            x=label_tahun, y=nilai_per_tahun.values,
            marker=dict(color=[warna_tahun[t] for t in tahun_unik]),
            text=[f"{v:.1f}" if pd.notna(v) else "-" for v in nilai_per_tahun.values],
            textposition="outside", cliponaxis=False,
            visible=(idx == 0), showlegend=False,
            hovertemplate="%{x}<br>" + KOLOM_LABEL[var] + ": %{y:.1f}<extra></extra>",
        ))

    # Legenda warna tahun (trace "hantu" terpisah, SELALU tampil apa pun
    # variabel yang lagi dipilih di dropdown) -- supaya warna tiap tahun
    # tetap dijelaskan, tidak ikut hilang saat variabel diganti. Ditaruh di
    # kiri atas supaya tidak tumpang tindih dengan dropdown pemilih variabel
    # yang ada di kanan atas.
    for t in tahun_unik:
        fig.add_trace(go.Bar(
            x=[None], y=[None], marker_color=warna_tahun[t],
            name=str(int(t)), showlegend=True, hoverinfo="skip",
        ))

    n_legenda = len(tahun_unik)
    buttons = []
    for idx, var in enumerate(variabel_urut):
        vis = [i == idx for i in range(len(variabel_urut))] + [True] * n_legenda
        satuan_ket = " (dijumlah setahun)" if var == 'rain' else " (rata-rata)"
        buttons.append(dict(
            label=KOLOM_LABEL[var].split(' (')[0], method="update",
            args=[{"visible": vis},
                  {"yaxis.title.text": KOLOM_LABEL[var] + satuan_ket,
                   "yaxis.range": range_per_var[var]}],
        ))

    layout = _layout_dasar(f"Perbandingan Rata-rata per Tahun — {stasiun}", tinggi=480, showlegend=True)
    # Tinggi total 480, margin atas 150 -> area plot 285px. Semua elemen yang
    # ditumpuk DI ATAS area plot (judul, legenda, label, dropdown) dihitung
    # eksplisit dalam pixel supaya tidak ada yang tumpang tindih ATAU
    # terdorong ke luar batas gambar (di luar 0..tinggi_total jadi kepotong):
    #   judul (title, koordinat "container")  -> ±18px dari atas
    #   legenda tahun                          -> ±55px dari atas
    #   label "Pilih variabel ▲"               -> ±85px dari atas
    #   dropdown                               -> mulai ±105px dari atas
    #   area plot mulai persis di 150px dari atas
    layout['margin'] = dict(l=55, r=30, t=150, b=45)
    layout['title']['y'] = 0.97
    layout['title']['yanchor'] = 'top'
    layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.333, xanchor="left", x=0, font=dict(size=12))
    layout['updatemenus'] = [dict(
        type="dropdown", direction="down", x=1.0, xanchor="right",
        y=1.158, yanchor="top", buttons=buttons, showactive=True,
        bgcolor="#ffffff", bordercolor="#cfe3dc",
    )]
    fig.update_layout(**layout)
    satuan_awal = " (dijumlah setahun)" if variabel_urut[0] == 'rain' else " (rata-rata)"
    fig.update_yaxes(title_text=KOLOM_LABEL[variabel_urut[0]] + satuan_awal,
                      range=range_per_var[variabel_urut[0]], gridcolor="#eaf3ef")
    fig.update_xaxes(title_text="Tahun", gridcolor="#eaf3ef", showgrid=False)
    fig.add_annotation(
        text="Pilih variabel ▲", x=1.0, xanchor="right", y=1.228, yanchor="top",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)


def plot_hubungan_cuaca_polutan_stasiun(df, stasiun):
    """Scatter polutan vs variabel cuaca, dengan DUA dropdown: pilih polutan
    (PM2.5 sebagai default — partikel paling halus & paling umum jadi acuan
    utama kualitas udara di berbagai studi, tapi polutan lain tetap bisa
    dipilih) dan pilih variabel cuaca (curah hujan, suhu, kelembapan,
    kecepatan angin). Dipakai sebagai penutup EDA per stasiun: memberi
    gambaran visual awal apakah kondisi meteorologi tampak berkaitan dengan
    tingkat polusi — sebelum pola asosiasinya digali lebih formal lewat
    FP-Growth di halaman Analisis. Tiap kali salah satu dropdown diganti,
    keterangan kecenderungan & korelasi (r) ikut diperbarui (kecenderungan
    visual, bukan hubungan sebab-akibat).

    Catatan teknis: dropdown polutan & dropdown cuaca berjalan independen
    lewat Plotly native (tanpa JS tambahan) — supaya keduanya bisa saling
    kombinasi tanpa perlu tahu status dropdown yang lain, mengganti SALAH
    SATU dropdown akan mereset dropdown yang lain ke pilihan pertamanya
    (PM2.5 / Curah Hujan). Ini trade-off yang disengaja demi kesederhanaan
    & keandalan (murni pakai fitur bawaan Plotly).
    """
    variabel_cuaca = ['rain', 'temp_air', 'hum_air', 'ff']
    daftar_polutan = ['pm_duakomalima', 'pm_sepuluh', 'sulfur_dioksida',
                       'karbon_monoksida', 'ozon', 'nitrogen_dioksida']

    polutan_valid = [p for p in daftar_polutan if p in df.columns and not df[p].dropna().empty]
    cuaca_valid = [v for v in variabel_cuaca if v in df.columns and not df[v].dropna().empty]
    if not polutan_valid or not cuaca_valid:
        return None

    def teks_keterangan(r, label_polutan):
        if r is None or math.isnan(r):
            arah = "data belum cukup untuk melihat kecenderungan"
        elif abs(r) < 0.2:
            arah = "kecenderungannya sangat lemah / hampir tidak terlihat"
        elif r > 0:
            arah = f"cenderung naik bersama — semakin tinggi nilainya, {label_polutan} juga cenderung ikut naik"
        else:
            arah = f"cenderung berlawanan — semakin tinggi nilainya, {label_polutan} cenderung turun"
        return f"Korelasi (r) = {r:.2f} — {arah}. Ini kecenderungan, bukan hubungan sebab-akibat."

    # Satu trace untuk tiap kombinasi (polutan, cuaca) -- disimpan dalam
    # urutan baris-per-baris (polutan luar, cuaca dalam) supaya indexnya
    # gampang dihitung ulang saat membuat tombol dropdown.
    fig = go.Figure()
    korelasi = {}  # (pol_idx, cuaca_idx) -> r
    for pi, pol in enumerate(polutan_valid):
        label_pol = KOLOM_LABEL.get(pol, pol).split(' (')[0]
        for ci, cuaca in enumerate(cuaca_valid):
            tmp = df[[cuaca, pol]].dropna()
            r = float(tmp[cuaca].corr(tmp[pol])) if len(tmp) >= 5 else float('nan')
            korelasi[(pi, ci)] = r
            label_cuaca = KOLOM_LABEL.get(cuaca, cuaca)
            fig.add_trace(go.Scatter(
                x=tmp[cuaca], y=tmp[pol], mode="markers",
                marker=dict(size=6, color="#0e6fc9", opacity=0.55, line=dict(width=0)),
                showlegend=False, visible=(pi == 0 and ci == 0),
                hovertemplate=f"{label_cuaca}: " + "%{x:.1f}<br>" + f"{label_pol}: " + "%{y:.1f}<extra></extra>",
            ))

    n_pol, n_cuaca = len(polutan_valid), len(cuaca_valid)

    def visible_mask(pol_idx, cuaca_idx):
        mask = []
        for pi in range(n_pol):
            for ci in range(n_cuaca):
                mask.append(pi == pol_idx and ci == cuaca_idx)
        return mask

    # Dropdown 1: pilih polutan (mereset cuaca ke pilihan pertama / cuaca_valid[0])
    tombol_polutan = []
    for pi, pol in enumerate(polutan_valid):
        label_pol_full = KOLOM_LABEL.get(pol, pol)
        label_pol = label_pol_full.split(' (')[0]
        r = korelasi[(pi, 0)]
        tombol_polutan.append(dict(
            label=label_pol, method="update",
            args=[{"visible": visible_mask(pi, 0)},
                  {"yaxis.title.text": label_pol_full,
                   "xaxis.title.text": KOLOM_LABEL.get(cuaca_valid[0], cuaca_valid[0]),
                   "annotations[0].text": teks_keterangan(r, label_pol)}],
        ))

    # Dropdown 2: pilih variabel cuaca (mereset polutan ke pilihan pertama / PM2.5)
    label_pol0 = KOLOM_LABEL.get(polutan_valid[0], polutan_valid[0]).split(' (')[0]
    tombol_cuaca = []
    for ci, cuaca in enumerate(cuaca_valid):
        r = korelasi[(0, ci)]
        tombol_cuaca.append(dict(
            label=KOLOM_LABEL.get(cuaca, cuaca).split(' (')[0], method="update",
            args=[{"visible": visible_mask(0, ci)},
                  {"xaxis.title.text": KOLOM_LABEL.get(cuaca, cuaca),
                   "yaxis.title.text": KOLOM_LABEL.get(polutan_valid[0], polutan_valid[0]),
                   "annotations[0].text": teks_keterangan(r, label_pol0)}],
        ))

    layout = _layout_dasar(f"Hubungan Cuaca & Polusi — {stasiun}", tinggi=480, showlegend=False)
    # Skema jarak sama seperti grafik "Perbandingan Antar Tahun": tinggi total
    # 480, margin atas 150 -> area plot 285px, dan tiap elemen di atas area
    # plot dihitung eksplisit dalam pixel dari tepi atas gambar:
    #   judul (title, koordinat "container")     -> ±18px
    #   teks korelasi (annotations[0])            -> ±50px
    #   label "Cuaca ▲" / "Polutan ▲"             -> ±85px
    #   dua dropdown                              -> mulai ±105px
    #   area plot mulai persis di 150px
    layout['margin'] = dict(l=55, r=30, t=150, b=45)
    layout['title']['y'] = 0.97
    layout['title']['yanchor'] = 'top'
    layout['updatemenus'] = [
        dict(
            type="dropdown", direction="down", x=1.0, xanchor="right",
            y=1.158, yanchor="top", buttons=tombol_polutan, showactive=True,
            bgcolor="#ffffff", bordercolor="#cfe3dc",
        ),
        dict(
            type="dropdown", direction="down", x=0.62, xanchor="right",
            y=1.158, yanchor="top", buttons=tombol_cuaca, showactive=True,
            bgcolor="#ffffff", bordercolor="#cfe3dc",
        ),
    ]
    fig.update_layout(**layout)
    fig.update_yaxes(title_text=KOLOM_LABEL.get(polutan_valid[0], polutan_valid[0]),
                      rangemode="tozero", gridcolor="#eaf3ef")
    fig.update_xaxes(title_text=KOLOM_LABEL.get(cuaca_valid[0], cuaca_valid[0]),
                      rangemode="tozero", gridcolor="#eaf3ef")
    fig.add_annotation(
        text=teks_keterangan(korelasi[(0, 0)], label_pol0),
        x=0.5, y=1.351, xref="paper", yref="paper", showarrow=False,
        font=dict(size=10.5, color="#5c7269"),
    )
    fig.add_annotation(
        text="Polutan ▲", x=1.0, xanchor="right", y=1.228, yanchor="top",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    fig.add_annotation(
        text="Cuaca ▲", x=0.62, xanchor="right", y=1.228, yanchor="top",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)




def plot_scatter_cuaca_polusi(df, stasiun, kolom_cuaca='rain'):
    """Scatter plot PM2.5 vs satu variabel cuaca — untuk melihat KECENDERUNGAN
    (bukan sebab-akibat, karena ini data observasional) antara cuaca &
    polusi. Default: curah hujan, karena paling intuitif untuk orang awam
    ('pas hujan udaranya lebih bersih gak sih?')."""
    if kolom_cuaca not in df.columns or 'pm_duakomalima' not in df.columns:
        return None
    tmp = df[[kolom_cuaca, 'pm_duakomalima']].dropna()
    if len(tmp) < 5:
        return None

    label_cuaca = KOLOM_LABEL.get(kolom_cuaca, kolom_cuaca)
    korelasi = tmp[kolom_cuaca].corr(tmp['pm_duakomalima'])

    fig = go.Figure(go.Scatter(
        x=tmp[kolom_cuaca], y=tmp['pm_duakomalima'], mode="markers",
        marker=dict(size=6, color="#0e6fc9", opacity=0.55, line=dict(width=0)),
        hovertemplate=f"{label_cuaca}: " + "%{x:.1f}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))
    layout = _layout_dasar(f"{label_cuaca} vs PM2.5 — {stasiun}", tinggi=380)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero", gridcolor="#eaf3ef")
    fig.update_xaxes(title_text=label_cuaca, rangemode="tozero", gridcolor="#eaf3ef")
    fig.add_annotation(
        text=f"Korelasi (r) = {korelasi:.2f} — nilai mendekati 0 berarti hubungannya lemah, "
             f"bukan berarti tidak ada pengaruh sama sekali",
        x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False,
        font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)


def plot_outlier_before_after(df_sebelum, df_sesudah, stasiun, kolom):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.3))
    axes[0].scatter(df_sebelum['periode'], df_sebelum[kolom], s=8, alpha=0.6, color='#d9534f')
    axes[0].set_title(f"Sebelum — {kolom}", fontsize=10)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(df_sesudah['periode'], df_sesudah[kolom], s=8, alpha=0.6, color='#3ba272')
    axes[1].set_title(f"Sesudah — {kolom}", fontsize=10)
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Outlier {kolom} — {stasiun}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# PLOTTING — HALAMAN OUTPUT (menjawab rumusan masalah)
# ============================================================
# Tiga grafik di bawah ini dirancang khusus untuk menjawab dua rumusan
# masalah skripsi secara visual:
#   1) "Bagaimana pola asosiasi antara kondisi meteorologi dan parameter
#      polutan terhadap kualitas udara?" -> plot_frekuensi_item_rules
#      (kondisi apa yang paling sering muncul di rule) dan
#      plot_scatter_support_confidence (sebaran kekuatan pola secara umum).
#   2) "Apa aturan asosiasi terkuat berdasarkan Support, Confidence, Lift?"
#      -> plot_top_rules_lift (ranking eksplisit berdasarkan Lift).

PREFIX_POLUTAN = {'pm10', 'pm25', 'so2', 'co', 'o3', 'no2'}
LABEL_PREFIX_ITEM = {
    'pm10': 'PM10', 'pm25': 'PM2.5', 'so2': 'SO2', 'co': 'CO', 'o3': 'O3', 'no2': 'NO2',
    'suhu': 'Suhu', 'lembab': 'Kelembaban', 'angin': 'Angin', 'hujan': 'Hujan',
}


def _format_rule_text(antecedents, consequents):
    ante = " & ".join(sorted(antecedents))
    kons = ", ".join(sorted(consequents))
    return f"{ante} → {kons}"


def plot_frekuensi_item_rules(rules_terpilih, daftar_stasiun):
    """Seberapa sering tiap kondisi meteorologi/polutan muncul sebagai
    antecedent pada rule final, digabung dari semua stasiun — jawaban
    visual untuk pola asosiasi apa yang paling sering terbentuk."""
    hitung = {}
    for stasiun in daftar_stasiun:
        r = rules_terpilih.get(stasiun)
        if r is None or r.empty:
            continue
        for antecedents in r['antecedents']:
            for item in antecedents:
                hitung[item] = hitung.get(item, 0) + 1
    if not hitung:
        return None

    items = sorted(hitung.items(), key=lambda x: x[1], reverse=True)
    label, nilai, warna = [], [], []
    for item, cnt in items:
        prefix, _, nilai_kategori = item.partition('=')
        label.append(f"{LABEL_PREFIX_ITEM.get(prefix, prefix)}: {nilai_kategori}")
        nilai.append(cnt)
        warna.append('#0e6fc9' if prefix in PREFIX_POLUTAN else '#17a673')

    fig = go.Figure(go.Bar(
        x=nilai, y=label, orientation='h', marker_color=warna,
        hovertemplate="%{y}<br>Muncul di %{x} rule<extra></extra>",
    ))
    layout = _layout_dasar("Kondisi yang Paling Sering Muncul pada Aturan Asosiasi",
                            tinggi=max(320, 30 * len(label)))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Jumlah rule", gridcolor="#eaf3ef", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.add_annotation(
        text="🔵 Parameter Polutan &nbsp;&nbsp; 🟢 Kondisi Meteorologi",
        x=1.0, xanchor="right", y=1.08, yanchor="bottom",
        xref="paper", yref="paper", showarrow=False, font=dict(size=10.5, color="#5c7269"),
    )
    return _fig_to_div(fig)


def plot_scatter_support_confidence(rules_terpilih, daftar_stasiun):
    """Sebaran seluruh rule final pada bidang Support x Confidence, ukuran
    titik = Lift — gambaran umum seberapa kuat & seberapa sering pola
    asosiasi yang ditemukan, dibedakan warnanya per stasiun."""
    fig = go.Figure()
    ada_data = False
    for i, stasiun in enumerate(daftar_stasiun):
        r = rules_terpilih.get(stasiun)
        if r is None or r.empty:
            continue
        ada_data = True
        warna = WARNA_STASIUN[i % len(WARNA_STASIUN)]
        label = stasiun.split(' ', 1)[-1].title()
        teks_rule = [_format_rule_text(a, c) for a, c in zip(r['antecedents'], r['consequents'])]
        ukuran = (r['lift'].clip(upper=5) * 7 + 6).tolist()
        fig.add_trace(go.Scatter(
            x=r['support'], y=r['confidence'], mode="markers", name=label,
            marker=dict(size=ukuran, color=warna, opacity=0.65, line=dict(width=1, color="white")),
            text=teks_rule, customdata=r['lift'],
            hovertemplate="%{text}<br>Support: %{x:.2f}<br>Confidence: %{y:.2f}"
                          "<br>Lift: %{customdata:.2f}<br>" + label + "<extra></extra>",
        ))
    if not ada_data:
        return None
    layout = _layout_dasar("Peta Kekuatan Aturan Asosiasi — Support vs Confidence (ukuran titik = Lift)",
                            tinggi=440, showlegend=True)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Support", gridcolor="#eaf3ef")
    fig.update_yaxes(title_text="Confidence", gridcolor="#eaf3ef")
    return _fig_to_div(fig)


def plot_top_rules_lift(rules_terpilih, daftar_stasiun, n=10):
    """Top-N aturan asosiasi dengan Lift tertinggi, digabung dari semua
    stasiun — jawaban visual langsung untuk 'aturan asosiasi terkuat'."""
    baris = []
    for stasiun in daftar_stasiun:
        r = rules_terpilih.get(stasiun)
        if r is None or r.empty:
            continue
        for _, row in r.iterrows():
            baris.append({
                'stasiun': stasiun.split(' ', 1)[-1].title(),
                'teks': _format_rule_text(row['antecedents'], row['consequents']),
                'support': row['support'], 'confidence': row['confidence'], 'lift': row['lift'],
            })
    if not baris:
        return None

    tabel = pd.DataFrame(baris).sort_values('lift', ascending=False).head(n).iloc[::-1]
    label = [f"{t}  ({s})" for t, s in zip(tabel['teks'], tabel['stasiun'])]

    fig = go.Figure(go.Bar(
        x=tabel['lift'], y=label, orientation='h', marker_color="#f7931e",
        text=[f"{v:.2f}" for v in tabel['lift']], textposition="outside",
        customdata=tabel[['support', 'confidence']].values,
        hovertemplate="%{y}<br>Lift: %{x:.2f}<br>Support: %{customdata[0]:.2f}"
                      "<br>Confidence: %{customdata[1]:.2f}<extra></extra>",
    ))
    layout = _layout_dasar(f"Top {n} Aturan Asosiasi Terkuat (berdasarkan Lift)",
                            tinggi=max(340, 38 * len(label)))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Lift", gridcolor="#eaf3ef", rangemode="tozero")
    fig.update_yaxes(automargin=True)
    return _fig_to_div(fig)