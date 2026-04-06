import cv2
import numpy as np
import os
import shutil
from pathlib import Path
 
 
# ──────────────────────────────────────────
#  KONFIGURASI PATH
# ──────────────────────────────────────────
SOURCE_IMAGES = "dataset/test/images"   # Ganti sesuai path dataset Anda
SOURCE_LABELS = "dataset/test/labels"   # Ganti sesuai path dataset Anda
OUTPUT_DIR    = "dataset_lighting"      # Folder output skenario
 
 
# ──────────────────────────────────────────
#  DEFINISI SKENARIO PENCAHAYAAN
# ──────────────────────────────────────────
SCENARIOS = {
    "S0":  {"name": "Normal (Baseline)",        "type": "baseline"},
    "S1":  {"name": "Underexposure Ringan",     "type": "gamma",      "gamma": 1.5},
    "S2":  {"name": "Underexposure Berat",      "type": "gamma",      "gamma": 2.5},
    "S3":  {"name": "Overexposure Ringan",      "type": "gamma",      "gamma": 0.6},
    "S4":  {"name": "Overexposure Berat",       "type": "gamma",      "gamma": 0.3},
    "S5":  {"name": "Bayangan Parsial",         "type": "shadow",     "ratio": 0.5, "darken": 0.4},
    "S6":  {"name": "Brightness Rendah",        "type": "brightness", "beta": -40},
    "S7":  {"name": "Brightness Tinggi",        "type": "brightness", "beta":  40},
    "S8":  {"name": "Saturation Rendah",        "type": "saturation", "delta": -0.4},
    "S9":  {"name": "Saturation Tinggi",        "type": "saturation", "delta":  0.4},
    "S10": {"name": "Exposure Rendah",          "type": "exposure",   "alpha": 0.7},
    "S11": {"name": "Exposure Tinggi",          "type": "exposure",   "alpha": 1.3},
}
 
 
# ──────────────────────────────────────────
#  FUNGSI TRANSFORMASI
# ──────────────────────────────────────────
 
def apply_baseline(img: np.ndarray) -> np.ndarray:
    """S0 - Tanpa transformasi."""
    return img.copy()
 
 
def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    S1-S4: Gamma Correction.
    γ > 1  → gambar lebih gelap  (underexposure)
    γ < 1  → gambar lebih terang (overexposure)
    """
    inv_gamma = 1.0 / gamma
    # Buat lookup table agar proses cepat
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, table)
 
 
def apply_shadow(img: np.ndarray, ratio: float = 0.5, darken: float = 0.4) -> np.ndarray:
    """
    S5: Bayangan Parsial.
    Membagi citra menjadi dua bagian horizontal;
    bagian bawah (ratio * tinggi) digelapkan dengan faktor darken.
    """
    result = img.copy()
    h = img.shape[0]
    split_y = int(h * ratio)
 
    # Konversi area shadow ke HSV lalu kurangi Value (kecerahan)
    shadow_region = result[split_y:, :].copy()
    hsv = cv2.cvtColor(shadow_region, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 - darken), 0, 255)
    result[split_y:, :] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result
 
 
def apply_brightness(img: np.ndarray, beta: int) -> np.ndarray:
    """
    S6-S7: Penyesuaian Brightness Linear.
    beta negatif → gelap, positif → terang.
    """
    # alpha=1 (kontras tetap), beta ubah kecerahan
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
 
 
def apply_saturation(img: np.ndarray, delta: float) -> np.ndarray:
    """
    S8-S9: Penyesuaian Saturasi melalui ruang warna HSV.
    delta = -0.4 → kurangi 40%,  delta = +0.4 → tambah 40%.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + delta), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
 
 
def apply_exposure(img: np.ndarray, alpha: float) -> np.ndarray:
    """
    S10-S11: Scaling Intensitas Linear (Exposure).
    alpha < 1 → kurangi paparan,  alpha > 1 → tambah paparan.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
 
 
def transform_image(img: np.ndarray, scenario_cfg: dict) -> np.ndarray:
    """Dispatcher: pilih fungsi transformasi berdasarkan tipe skenario."""
    t = scenario_cfg["type"]
    if t == "baseline":
        return apply_baseline(img)
    elif t == "gamma":
        return apply_gamma(img, scenario_cfg["gamma"])
    elif t == "shadow":
        return apply_shadow(img, scenario_cfg["ratio"], scenario_cfg["darken"])
    elif t == "brightness":
        return apply_brightness(img, scenario_cfg["beta"])
    elif t == "saturation":
        return apply_saturation(img, scenario_cfg["delta"])
    elif t == "exposure":
        return apply_exposure(img, scenario_cfg["alpha"])
    else:
        raise ValueError(f"Tipe skenario tidak dikenal: {t}")
 
 
# ──────────────────────────────────────────
#  FUNGSI UTAMA
# ──────────────────────────────────────────
 
def generate_lighting_dataset(
    src_images: str,
    src_labels: str,
    output_dir: str,
    scenarios: dict
):
    src_images = Path(src_images)
    src_labels = Path(src_labels)
    output_dir = Path(output_dir)
 
    # Ekstensi gambar yang didukung
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in src_images.iterdir() if f.suffix.lower() in img_exts]
 
    if not image_files:
        print(f"[ERROR] Tidak ada gambar ditemukan di: {src_images}")
        return
 
    print(f"\n{'='*55}")
    print(f"  Dataset Test  : {src_images}")
    print(f"  Total Gambar  : {len(image_files)}")
    print(f"  Output Dir    : {output_dir}")
    print(f"  Total Skenario: {len(scenarios)}")
    print(f"{'='*55}\n")
 
    for code, cfg in scenarios.items():
        out_img_dir = output_dir / code / "images"
        out_lbl_dir = output_dir / code / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
 
        print(f"[{code}] Memproses: {cfg['name']} ...")
 
        ok = 0
        for img_path in image_files:
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Gagal membaca: {img_path.name}")
                continue
 
            # Terapkan transformasi
            transformed = transform_image(img, cfg)
 
            # Simpan gambar hasil
            out_img_path = out_img_dir / img_path.name
            cv2.imwrite(str(out_img_path), transformed)
 
            # Salin label (anotasi tidak berubah)
            lbl_path = src_labels / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(str(lbl_path), str(out_lbl_dir / lbl_path.name))
 
            ok += 1
 
        print(f"  ✓ Selesai: {ok}/{len(image_files)} gambar\n")
 
    print("="*55)
    print("  Semua skenario pencahayaan berhasil dibuat!")
    print(f"  Output tersimpan di: {output_dir.resolve()}")
    print("="*55)
 
 
# ──────────────────────────────────────────
#  PREVIEW (OPSIONAL) - Simpan grid preview
# ──────────────────────────────────────────
 
def save_preview_grid(sample_image_path: str, output_dir: str, scenarios: dict):
    """
    Buat satu gambar grid yang menampilkan semua skenario
    untuk satu sampel citra. Berguna untuk verifikasi visual.
    """
    img = cv2.imread(sample_image_path)
    if img is None:
        print(f"[WARN] Gambar preview tidak ditemukan: {sample_image_path}")
        return
 
    img_resized = cv2.resize(img, (320, 320))
    results = []
 
    for code, cfg in scenarios.items():
        transformed = transform_image(img_resized, cfg)
        # Tambahkan label teks pada gambar
        label_img = transformed.copy()
        cv2.putText(
            label_img, f"{code}: {cfg['name']}",
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 255), 1, cv2.LINE_AA
        )
        results.append(label_img)
 
    # Susun grid 4 kolom
    cols = 4
    rows = (len(results) + cols - 1) // cols
    # Pad agar jumlah gambar genap
    blank = np.zeros_like(results[0])
    while len(results) % cols != 0:
        results.append(blank)
 
    row_imgs = [
        np.hstack(results[i*cols:(i+1)*cols])
        for i in range(rows)
    ]
    grid = np.vstack(row_imgs)
 
    out_path = Path(output_dir) / "preview_lighting_grid.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"\n[Preview] Grid disimpan: {out_path.resolve()}")
 
 
# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
 
if __name__ == "__main__":
    # 1. Generate seluruh skenario pencahayaan
    generate_lighting_dataset(
        src_images=SOURCE_IMAGES,
        src_labels=SOURCE_LABELS,
        output_dir=OUTPUT_DIR,
        scenarios=SCENARIOS,
    )
 
    # 2. (Opsional) Simpan preview grid dari satu sampel gambar
    #    Ganti path di bawah dengan salah satu gambar test Anda
    SAMPLE_IMG = "dataset/test/images/sample.jpg"
    if Path(SAMPLE_IMG).exists():
        save_preview_grid(SAMPLE_IMG, OUTPUT_DIR, SCENARIOS)
    else:
        print("\n[INFO] Untuk melihat preview grid, ubah SAMPLE_IMG")
        print("       ke salah satu path gambar test Anda.")