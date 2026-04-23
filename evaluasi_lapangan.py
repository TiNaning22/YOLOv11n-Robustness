import os
import csv
import yaml
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from ultralytics import YOLO
 
 
# ──────────────────────────────────────────
#  KONFIGURASI — SESUAIKAN PATH INI
# ──────────────────────────────────────────
# DRIVE_ROOT        = "/content/drive/MyDrive"
MODEL_PATH        = "yolov11n_baseline2/weights/best.pt"
FIELD_ANNOT_DIR   = "dataset_lapangan"
SYNTHETIC_CSV     = "hasil_evaluasi/robustness_results_v2.csv"
OUTPUT_DIR        = "hasil_evaluasi/lapangan/field_comparison_v2"
IMG_SIZE          = 640
 
# Confidence threshold optimal dari F1 curve terbaru
CONF_THRESHOLD    = 0.01
# CONF_THRESHOLD    = 0.5
 
# Baseline referensi dari training terbaru
BASELINE_MAP50    = 0.94
 
CLASS_NAMES = [
    "Healthy",
    "Brown Spot",
    "Leaf Blast",
    "Leaf Blight",
    "Leaf Scald",
    "Narrow Brown Spot",
]
 
# AP per kelas baseline (dari PR Curve training terbaru)
BASELINE_AP_PER_CLASS = {
    "Healthy":          0.973,
    "Brown Spot":       0.856,
    "Leaf Blast":       0.863,
    "Leaf Blight":      0.995,
    "Leaf Scald":       0.995,
    "Narrow Brown Spot":0.994,
}
 
# Definisi kondisi lapangan + skenario sintetis padanannya
FIELD_CONDITIONS = {
    "L1": {
        "name":          "Pagi Hari (07.00–08.00)",
        "analogy":       ["S1", "S6"],
        "analogy_label": "S1 (Underexposure Ringan) + S6 (Brightness Rendah)",
        "karakteristik": "Cahaya rendah-sedang, suhu warna dingin, sudut rendah",
    },
    "L2": {
        "name":          "Siang Hari (11.00–13.00)",
        "analogy":       ["S3", "S7"],
        "analogy_label": "S3 (Overexposure Ringan) + S7 (Brightness Tinggi)",
        "karakteristik": "Cahaya tinggi, bayangan minimal, kontras tinggi",
    },
    "L3": {
        "name":          "Sore Hari (15.00–17.00)",
        "analogy":       ["S1", "S6"],
        "analogy_label": "S1 (Underexposure Ringan) + S6 (Brightness Rendah)",
        "karakteristik": "Cahaya rendah-sedang, suhu warna hangat, sudut rendah",
    },
    "L4": {
        "name":          "Area Bayangan Tanaman",
        "analogy":       ["S5", "S2"],
        "analogy_label": "S5 (Bayangan Parsial) + S2 (Underexposure Berat)",
        "karakteristik": "Pencahayaan tidak merata akibat naungan kanopi",
    },
}
 
FIELD_ORDER = ["L1", "L2", "L3", "L4"]
 
 
# ──────────────────────────────────────────
#  BUAT data.yaml SEMENTARA PER KONDISI
# ──────────────────────────────────────────
 
def create_field_yaml(condition_code: str, field_annot_dir: str) -> str:
    cond_path = Path(field_annot_dir) / condition_code
    yaml_path = cond_path / "data.yaml"
    config = {
        "path":  str(cond_path.resolve()),
        "train": "images",
        "val":   "images",
        "test":  "images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return str(yaml_path)
 
 
# ──────────────────────────────────────────
#  CEK KELENGKAPAN DATA LAPANGAN
# ──────────────────────────────────────────
 
def check_field_data(field_annot_dir: str) -> dict:
    """
    Cek kelengkapan folder dan label untuk setiap kondisi lapangan.
    Kembalikan dict status per kondisi.
    """
    status = {}
    print(f"\n{'='*55}")
    print(f"  Cek Kelengkapan Data Lapangan")
    print(f"{'='*55}")
 
    for code in FIELD_ORDER:
        img_dir = Path(field_annot_dir) / code / "images"
        lbl_dir = Path(field_annot_dir) / code / "labels"
 
        img_exts  = {".jpg", ".jpeg", ".png", ".bmp"}
        n_images  = len([f for f in img_dir.iterdir()
                         if f.suffix.lower() in img_exts]) if img_dir.exists() else 0
        n_labels  = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        n_empty   = 0
 
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                content = lbl.read_text().strip()
                if not content or all(l.startswith("#") for l in content.splitlines()):
                    n_empty += 1
 
        ok = img_dir.exists() and lbl_dir.exists() and n_images > 0 and n_labels > 0
 
        cond_name = FIELD_CONDITIONS[code]["name"]
        print(f"  [{code}] {cond_name}")
        print(f"       Gambar : {n_images}  |  Label : {n_labels}  |  "
              f"Label kosong : {n_empty}  |  Status : {'✓ Siap' if ok else '✗ Belum siap'}")
 
        status[code] = {
            "ok": ok, "n_images": n_images,
            "n_labels": n_labels, "n_empty": n_empty,
        }
 
    print(f"{'='*55}\n")
    return status
 
 
# ──────────────────────────────────────────
#  EVALUASI SATU KONDISI LAPANGAN
# ──────────────────────────────────────────
 
def evaluate_field_condition(
    model: YOLO,
    condition_code: str,
    field_annot_dir: str,
    conf: float = CONF_THRESHOLD,
    img_size: int = IMG_SIZE,
) -> dict | None:
    img_dir = Path(field_annot_dir) / condition_code / "images"
    lbl_dir = Path(field_annot_dir) / condition_code / "labels"
 
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"  [SKIP] {condition_code}: folder tidak ditemukan")
        return None
 
    label_files = list(lbl_dir.glob("*.txt"))
    if not label_files:
        print(f"  [SKIP] {condition_code}: tidak ada label")
        print(f"         Jalankan annotate_field.py dan koreksi manual!")
        return None
 
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    yaml_path = create_field_yaml(condition_code, field_annot_dir)
 
    metrics = model.val(
        data    = yaml_path,
        split   = "test",
        imgsz   = img_size,
        conf    = conf,
        device  = device,
        verbose = False,
    )
 
    p  = metrics.box.mp
    r  = metrics.box.mr
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
 
    # AP per kelas jika tersedia
    ap_per_class = {}
    if hasattr(metrics.box, "ap_class_index") and hasattr(metrics.box, "ap"):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            ap_per_class[name] = round(float(metrics.box.ap[i]), 4)
 
    cond_info = FIELD_CONDITIONS.get(condition_code, {})
    return {
        "code":         condition_code,
        "name":         cond_info.get("name", condition_code),
        "karakteristik":cond_info.get("karakteristik", ""),
        "precision":    round(p,  4),
        "recall":       round(r,  4),
        "f1":           round(f1, 4),
        "map50":        round(metrics.box.map50, 4),
        "map5095":      round(metrics.box.map,   4),
        "ap_per_class": ap_per_class,
    }
 
 
# ──────────────────────────────────────────
#  BACA HASIL SINTETIS DARI CSV
# ──────────────────────────────────────────
 
def load_synthetic_results(csv_path: str) -> dict:
    results = {}
    if not Path(csv_path).exists():
        print(f"[WARN] CSV sintetis tidak ditemukan: {csv_path}")
        print(f"       Jalankan robustness_evaluation_v2.py terlebih dahulu!")
        return results
 
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"]
            results[code] = {
                "code":      code,
                "name":      row["name"],
                "precision": float(row["precision"]),
                "recall":    float(row["recall"]),
                "f1":        float(row["f1"]),
                "map50":     float(row["map50"]),
                "map5095":   float(row["map5095"]),
                "drop_pct":  float(row["drop_pct"]),
                "robustness":row.get("robustness", ""),
            }
    print(f"[INFO] Data sintetis dimuat: {len(results)} skenario dari {csv_path}")
    return results
 
 
# ──────────────────────────────────────────
#  HITUNG GAP LAPANGAN vs SINTETIS
# ──────────────────────────────────────────
 
def compute_comparison(
    field_results: list,
    synthetic_results: dict,
    baseline_map50: float,
) -> list:
    """
    Untuk setiap kondisi lapangan Lx hitung:
    - Drop vs Baseline (%)
    - Rata-rata mAP50 skenario sintetis padanan
    - Gap (%) antara lapangan dan sintetis padanan
    - Interpretasi keterwakilan sintetis
    """
    comparison = []
 
    for fr in field_results:
        code      = fr["code"]
        cond_info = FIELD_CONDITIONS.get(code, {})
        analogies = cond_info.get("analogy", [])
 
        # Rata-rata mAP50 skenario sintetis padanan
        analogy_maps = [
            synthetic_results[s]["map50"]
            for s in analogies if s in synthetic_results
        ]
        avg_synth_map50 = round(np.mean(analogy_maps), 4) if analogy_maps else None
 
        # Drop lapangan vs baseline
        field_drop = round(
            ((baseline_map50 - fr["map50"]) / baseline_map50 * 100)
            if baseline_map50 > 0 else 0.0, 2
        )
 
        # Drop sintetis vs baseline
        synth_drop = round(
            ((baseline_map50 - avg_synth_map50) / baseline_map50 * 100)
            if avg_synth_map50 and baseline_map50 > 0 else 0.0, 2
        )
 
        # Gap lapangan vs sintetis
        # positif = lapangan lebih buruk dari sintetis
        # negatif = lapangan lebih baik dari sintetis
        gap_pct = round(
            ((avg_synth_map50 - fr["map50"]) / avg_synth_map50 * 100)
            if avg_synth_map50 else 0.0, 2
        )
 
        # Kategori ketahanan lapangan
        if abs(field_drop) <= 5:
            robustness_field = "Sangat Tahan"
        elif abs(field_drop) <= 15:
            robustness_field = "Tahan"
        elif abs(field_drop) <= 30:
            robustness_field = "Cukup Tahan"
        else:
            robustness_field = "Tidak Tahan"
 
        # Keterwakilan sintetis terhadap lapangan
        if abs(gap_pct) <= 5:
            representasi = "Baik (gap ≤5%)"
        elif abs(gap_pct) <= 15:
            representasi = "Cukup (gap 5–15%)"
        else:
            representasi = "Kurang (gap >15%)"
 
        comparison.append({
            **fr,
            "drop_vs_baseline":  field_drop,
            "robustness_field":  robustness_field,
            "analogy_codes":     "+".join(analogies),
            "analogy_label":     cond_info.get("analogy_label", ""),
            "synth_map50":       avg_synth_map50,
            "synth_drop_pct":    synth_drop,
            "gap_pct":           gap_pct,
            "representasi":      representasi,
        })
 
    return comparison
 
 
# ──────────────────────────────────────────
#  PRINT TABEL HASIL
# ──────────────────────────────────────────
 
def print_field_table(field_results: list, baseline_map50: float):
    sep = "─" * 78
    print(f"\n{'='*78}")
    print(f"  METRIK PER KONDISI LAPANGAN (conf={CONF_THRESHOLD})")
    print(f"  Baseline mAP@0.5: {baseline_map50:.4f}")
    print(f"{'='*78}")
    print(f"  {'Kode':<5} {'Kondisi':<26} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP5095':>8} {'Drop%':>7}")
    print(sep)
    for r in field_results:
        drop = ((baseline_map50 - r["map50"]) / baseline_map50 * 100
                if baseline_map50 > 0 else 0.0)
        print(
            f"  {r['code']:<5} {r['name']:<26} "
            f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f} "
            f"{r['map50']:>7.4f} {r['map5095']:>8.4f} {drop:>6.2f}%"
        )
    print(sep)
 
 
def print_comparison_table(comparison: list):
    print(f"\n{'='*90}")
    print(f"  PERBANDINGAN LAPANGAN vs SINTETIS (Rumusan Masalah ke-4)")
    print(f"{'='*90}")
    print(f"  {'Kode':<5} {'Kondisi':<24} {'mAP-L':>7} {'Padanan':>12} {'mAP-S':>7} "
          f"{'Gap%':>7} {'Ketahanan':<16} {'Representasi'}")
    print("─" * 90)
    for r in comparison:
        synth = f"{r['synth_map50']:.4f}" if r["synth_map50"] else "N/A"
        print(
            f"  {r['code']:<5} {r['name']:<24} "
            f"{r['map50']:>7.4f} {r['analogy_codes']:>12} "
            f"{synth:>7} {r['gap_pct']:>7.2f}%  "
            f"{r['robustness_field']:<16} {r['representasi']}"
        )
    print("─" * 90)
    print("\n  Interpretasi Gap (keterwakilan sintetis terhadap lapangan nyata):")
    print("    ≤  5% → sintetis BAIK  merepresentasikan lapangan")
    print("    5–15% → sintetis CUKUP merepresentasikan lapangan")
    print("    > 15% → sintetis KURANG merepresentasikan lapangan")
 
 
# ──────────────────────────────────────────
#  SIMPAN CSV
# ──────────────────────────────────────────
 
def save_comparison_csv(data: list, output_path: str):
    fieldnames = [
        "code", "name", "karakteristik",
        "precision", "recall", "f1", "map50", "map5095",
        "drop_vs_baseline", "robustness_field",
        "analogy_codes", "analogy_label",
        "synth_map50", "synth_drop_pct",
        "gap_pct", "representasi",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    print(f"[INFO] CSV perbandingan disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 1: mAP50 Lapangan vs Sintetis per Kondisi
# ──────────────────────────────────────────
 
def plot_field_vs_synthetic(
    field_results: list,
    comparison: list,
    synthetic_results: dict,
    baseline_map50: float,
    output_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Perbandingan mAP@0.5: Kondisi Lapangan vs Skenario Sintetis Padanan\n"
        f"YOLOv11n — Baseline mAP@0.5 = {baseline_map50:.4f}  |  conf = {CONF_THRESHOLD}",
        fontsize=12, fontweight="bold"
    )
 
    for idx, (fr, cmp) in enumerate(zip(field_results, comparison)):
        ax   = axes[idx // 2][idx % 2]
        code = fr["code"]
        analogies = FIELD_CONDITIONS[code]["analogy"]
 
        labels, maps, colors, hatches = [], [], [], []
 
        # S0 baseline
        if "S0" in synthetic_results:
            labels.append(f"S0\nBaseline")
            maps.append(synthetic_results["S0"]["map50"])
            colors.append("#2ecc71")
            hatches.append("")
 
        # Skenario sintetis padanan
        for s in analogies:
            if s in synthetic_results:
                labels.append(f"{s}\nSintetis")
                maps.append(synthetic_results[s]["map50"])
                colors.append("#3498db")
                hatches.append("//")
 
        # Kondisi lapangan
        labels.append(f"{code}\nLapangan")
        maps.append(fr["map50"])
        colors.append("#e74c3c")
        hatches.append("xx")
 
        bars = ax.bar(labels, maps, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.55)
 
        ax.set_title(
            f"{code}: {fr['name']}\n"
            f"Padanan: {cmp['analogy_label'][:40]}",
            fontsize=9, fontweight="bold"
        )
        ax.set_ylabel("mAP@0.5", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.axhline(y=baseline_map50, color="gray", linestyle="--",
                   linewidth=0.9, alpha=0.7)
 
        # Anotasi nilai
        for bar, val in zip(bars, maps):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5
            )
 
        # Label gap & representasi
        gap_color = (
            "#27ae60" if abs(cmp["gap_pct"]) <= 5 else
            "#f39c12" if abs(cmp["gap_pct"]) <= 15 else "#e74c3c"
        )
        ax.text(
            0.97, 0.06,
            f"Gap: {cmp['gap_pct']:+.2f}%\n{cmp['representasi']}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=gap_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=gap_color, alpha=0.85)
        )
 
        # Label drop lapangan
        ax.text(
            0.03, 0.06,
            f"Drop L: {cmp['drop_vs_baseline']:.1f}%\n{cmp['robustness_field']}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.85)
        )
 
    # Legend global
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="S0 Baseline"),
        mpatches.Patch(color="#3498db", label="Sintetis Padanan"),
        mpatches.Patch(color="#e74c3c", label="Kondisi Lapangan"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.01))
 
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 1 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 2: Grouped Bar Drop% Lapangan vs Sintetis
# ──────────────────────────────────────────
 
def plot_drop_comparison(comparison: list, baseline_map50: float, output_path: str):
    x      = np.arange(len(comparison))
    width  = 0.35
    xlabels = [f"{c['code']}\n{c['name']}" for c in comparison]
    synth_drops = [c["synth_drop_pct"]    for c in comparison]
    field_drops = [c["drop_vs_baseline"]  for c in comparison]
 
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        "Perbandingan Drop mAP@0.5 (%) — Lapangan vs Sintetis Padanan\n"
        f"YOLOv11n — Baseline = {baseline_map50:.4f}  |  conf = {CONF_THRESHOLD}",
        fontsize=12, fontweight="bold"
    )
 
    bars_s = ax.bar(x - width/2, synth_drops, width,
                    label="Sintetis Padanan (rata-rata)",
                    color="#3498db", edgecolor="white", alpha=0.85)
    bars_f = ax.bar(x + width/2, field_drops, width,
                    label="Kondisi Lapangan",
                    color="#e74c3c", edgecolor="white", alpha=0.85)
 
    # Garis threshold
    ax.axhline(y=5,  color="#3498db", linestyle=":", linewidth=1,
               alpha=0.6, label="Sangat Tahan ≤5%")
    ax.axhline(y=15, color="#f39c12", linestyle=":", linewidth=1,
               alpha=0.6, label="Tahan ≤15%")
    ax.axhline(y=30, color="#e74c3c", linestyle=":", linewidth=1,
               alpha=0.6, label="Tidak Tahan >30%")
 
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Drop mAP@0.5 (%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(y=0, color="gray", linewidth=0.8)
 
    for bar in list(bars_s) + list(bars_f):
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 2 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 3: Radar Chart Metrik per Kondisi
# ──────────────────────────────────────────
 
def plot_radar_chart(
    field_results: list,
    comparison: list,
    synthetic_results: dict,
    output_path: str,
):
    metrics_keys   = ["precision", "recall", "f1", "map50", "map5095"]
    metrics_labels = ["Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.5:0.95"]
    N      = len(metrics_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             subplot_kw=dict(polar=True))
    fig.suptitle(
        "Radar Chart Metrik — Kondisi Lapangan vs Sintetis Padanan\n"
        f"YOLOv11n | conf = {CONF_THRESHOLD}",
        fontsize=13, fontweight="bold"
    )
 
    lcolors = {"L1": "#e74c3c", "L2": "#e67e22", "L3": "#9b59b6", "L4": "#1abc9c"}
 
    for idx, (fr, cmp) in enumerate(zip(field_results, comparison)):
        ax   = axes[idx // 2][idx % 2]
        code = fr["code"]
 
        # Data lapangan
        vals_field = [fr[k] for k in metrics_keys] + [fr[metrics_keys[0]]]
 
        # Data sintetis padanan (rata-rata)
        analogies     = FIELD_CONDITIONS[code]["analogy"]
        analogy_data  = [synthetic_results[s] for s in analogies
                         if s in synthetic_results]
        if analogy_data:
            vals_synth = [np.mean([d[k] for d in analogy_data])
                          for k in metrics_keys]
            vals_synth += [vals_synth[0]]
        else:
            vals_synth = None
 
        # Data baseline S0
        if "S0" in synthetic_results:
            vals_base = [synthetic_results["S0"][k] for k in metrics_keys]
            vals_base += [vals_base[0]]
            ax.plot(angles, vals_base, "o--", color="#2ecc71",
                    linewidth=1.2, alpha=0.6, label="S0 Baseline")
            ax.fill(angles, vals_base, alpha=0.05, color="#2ecc71")
 
        # Plot sintetis
        if vals_synth:
            ax.plot(angles, vals_synth, "s--", color="#3498db",
                    linewidth=1.5, label="Sintetis Padanan")
            ax.fill(angles, vals_synth, alpha=0.1, color="#3498db")
 
        # Plot lapangan
        c = lcolors.get(code, "#e74c3c")
        ax.plot(angles, vals_field, "o-", color=c,
                linewidth=2, label=f"{code} Lapangan")
        ax.fill(angles, vals_field, alpha=0.15, color=c)
 
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
        ax.set_title(
            f"{code}: {fr['name']}\nGap={cmp['gap_pct']:+.2f}% | {cmp['representasi']}",
            size=9, fontweight="bold", pad=14
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=7.5)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 3 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 4: AP per Kelas — Lapangan vs Baseline
# ──────────────────────────────────────────
 
def plot_ap_per_class_field(field_results: list, output_path: str):
    """
    Grouped bar: AP tiap kelas di kondisi lapangan L1-L4
    dibandingkan AP baseline.
    """
    # Cek apakah ada data AP per kelas
    has_data = any(
        r.get("ap_per_class") and any(v > 0 for v in r["ap_per_class"].values())
        for r in field_results
    )
    if not has_data:
        print("[INFO] Data AP per kelas lapangan tidak tersedia, plot 4 dilewati.")
        return
 
    x      = np.arange(len(CLASS_NAMES))
    width  = 0.15
    lcolors_map = {
        "L1": "#e74c3c", "L2": "#e67e22",
        "L3": "#9b59b6", "L4": "#1abc9c"
    }
 
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(
        "AP per Kelas: Kondisi Lapangan vs Baseline\n"
        "YOLOv11n — Deteksi Penyakit Daun Padi",
        fontsize=12, fontweight="bold"
    )
 
    # Baseline AP per kelas
    baseline_vals = [BASELINE_AP_PER_CLASS.get(n, 0) for n in CLASS_NAMES]
    offset_base   = -(len(field_results)/2) * width
    ax.bar(x + offset_base, baseline_vals, width,
           label="S0 Baseline", color="#2ecc71",
           edgecolor="white", linewidth=0.5, alpha=0.85)
 
    # AP per kondisi lapangan
    for i, fr in enumerate(field_results):
        ap_dict = fr.get("ap_per_class", {})
        vals    = [ap_dict.get(n, 0) for n in CLASS_NAMES]
        offset  = (-(len(field_results)/2) + i + 1) * width
        ax.bar(x + offset, vals, width,
               label=f"{fr['code']} {fr['name'][:12]}",
               color=lcolors_map.get(fr["code"], "#888"),
               edgecolor="white", linewidth=0.5, alpha=0.85)
 
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Average Precision (AP@0.5)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    ax.axhline(y=0, color="gray", linewidth=0.5)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 4 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  SIMPAN RINGKASAN TEKS
# ──────────────────────────────────────────
 
def save_summary_txt(comparison: list, baseline_map50: float, output_path: str):
    lines = [
        "=" * 65,
        "  RINGKASAN EVALUASI DATA LAPANGAN",
        "  Proposal Skripsi: Rangga Putra Sopyan (221210029)",
        "  Universitas Mercu Buana Yogyakarta — 2026",
        "=" * 65,
        "",
        f"  Baseline mAP@0.5 (S0) : {baseline_map50:.4f}",
        f"  Confidence threshold   : {CONF_THRESHOLD}",
        "",
        "  AP per kelas baseline (S0):",
    ]
    for name, ap in BASELINE_AP_PER_CLASS.items():
        lines.append(f"    {name:<22}: {ap:.3f}")
 
    lines += [
        "",
        "  Hasil kondisi lapangan:",
        "  " + "─" * 60,
    ]
    for r in comparison:
        lines.append(
            f"  {r['code']:<4} {r['name']:<26} "
            f"mAP50={r['map50']:.4f}  "
            f"Drop={r['drop_vs_baseline']:>6.2f}%  "
            f"[{r['robustness_field']}]"
        )
 
    lines += [
        "",
        "  Perbandingan vs Sintetis:",
        "  " + "─" * 60,
    ]
    for r in comparison:
        lines.append(
            f"  {r['code']:<4} Padanan: {r['analogy_codes']:<8} "
            f"mAP-Sintetis={r['synth_map50']:.4f}  "
            f"Gap={r['gap_pct']:>+.2f}%  "
            f"[{r['representasi']}]"
        )
 
    lines += ["", "=" * 65]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Ringkasan disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi & perbandingan data lapangan vs sintetis v2"
    )
    parser.add_argument("--model",     default=MODEL_PATH,      help="Path ke best.pt")
    parser.add_argument("--field",     default=FIELD_ANNOT_DIR, help="Folder data lapangan")
    parser.add_argument("--synth_csv", default=SYNTHETIC_CSV,   help="CSV hasil sintetis v2")
    parser.add_argument("--output",    default=OUTPUT_DIR,      help="Folder output")
    parser.add_argument("--conf",      default=CONF_THRESHOLD,  type=float,
                        help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    args = parser.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output).mkdir(parents=True, exist_ok=True)
 
    print(f"\n{'='*60}")
    print(f"  Evaluasi Data Lapangan vs Sintetis — v2")
    print(f"  Model      : {args.model}")
    print(f"  Conf thr   : {args.conf}")
    print(f"  Device     : {device.upper()}")
    print(f"  Output     : {args.output}")
    print(f"{'='*60}")
 
    # ── Cek kelengkapan data lapangan ───────────────────
    status = check_field_data(args.field)
 
    # ── Muat model ───────────────────────────────────────
    model = YOLO(args.model)
    print(f"[INFO] Model dimuat: {list(model.names.values())}\n")
 
    # ── Muat hasil sintetis ──────────────────────────────
    synthetic_results = load_synthetic_results(args.synth_csv)
    baseline_map50    = (synthetic_results["S0"]["map50"]
                         if "S0" in synthetic_results else BASELINE_MAP50)
    print(f"[INFO] Baseline mAP@0.5: {baseline_map50:.4f}\n")
 
    # ── Evaluasi kondisi lapangan ────────────────────────
    field_results = []
    for code in FIELD_ORDER:
        if not status.get(code, {}).get("ok", False):
            print(f"  [{code}] SKIP — data belum siap")
            continue
 
        cond_name = FIELD_CONDITIONS[code]["name"]
        print(f"  [{code}] {cond_name} ...", end=" ", flush=True)
 
        result = evaluate_field_condition(
            model           = model,
            condition_code  = code,
            field_annot_dir = args.field,
            conf            = args.conf,
            img_size        = IMG_SIZE,
        )
        if result:
            drop = ((baseline_map50 - result["map50"]) / baseline_map50 * 100
                    if baseline_map50 > 0 else 0.0)
            result["drop_vs_baseline"] = round(drop, 2)
            field_results.append(result)
            print(f"mAP50={result['map50']:.4f}  F1={result['f1']:.4f}  "
                  f"Drop={drop:.2f}%  ✓")
        else:
            print("SKIP")
 
    if not field_results:
        print("\n[ERROR] Tidak ada kondisi lapangan yang berhasil dievaluasi.")
        print("        Pastikan annotate_field.py sudah dijalankan dan")
        print("        anotasi sudah dikoreksi manual.")
        return
 
    # ── Hitung perbandingan ──────────────────────────────
    comparison = compute_comparison(field_results, synthetic_results, baseline_map50)
 
    # ── Tampilkan tabel ──────────────────────────────────
    print_field_table(field_results, baseline_map50)
    print_comparison_table(comparison)
 
    # ── Simpan output ────────────────────────────────────
    out = Path(args.output)
 
    save_comparison_csv(comparison, str(out / "field_vs_synthetic_v2.csv"))
    save_summary_txt(comparison, baseline_map50, str(out / "field_summary_v2.txt"))
 
    plot_field_vs_synthetic(
        field_results, comparison, synthetic_results,
        baseline_map50, str(out / "plot1_field_vs_synthetic.png")
    )
    plot_drop_comparison(
        comparison, baseline_map50, str(out / "plot2_drop_comparison.png")
    )
    plot_radar_chart(
        field_results, comparison, synthetic_results,
        str(out / "plot3_radar_chart.png")
    )
    plot_ap_per_class_field(
        field_results, str(out / "plot4_ap_per_class_field.png")
    )
 
    print(f"\n{'='*60}")
    print(f"  SELESAI — Output tersimpan di:")
    print(f"  {out.resolve()}")
    print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    main()