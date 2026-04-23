import os
import csv
import yaml
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path
from ultralytics import YOLO
 
 
# ──────────────────────────────────────────
#  KONFIGURASI — SESUAIKAN PATH INI
# ──────────────────────────────────────────
# DRIVE_ROOT    = "/content/drive/MyDrive"
MODEL_PATH    = "yolov11n_baseline2/weights/best.pt"
LIGHTING_DIR  = "dataset_lighting"
RESULTS_DIR   = "hasil_evaluasi"
IMG_SIZE      = 640
 
# Confidence threshold optimal dari F1 curve terbaru
CONF_THRESHOLD = 0.489
CLASS_NAMES = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown']

 
# AP per kelas dari PR Curve (baseline referensi untuk analisis)
BASELINE_AP_PER_CLASS = {
    "Healthy":          0.973,
    "Brown Spot":       0.856,
    "Leaf Blast":       0.863,
    "Leaf Blight":      0.995,
    "Leaf Scald":       0.995,
    "Narrow Brown Spot":0.994,
}
 
SCENARIO_NAMES = {
    "S0":  "Normal (Baseline)",
    "S1":  "Overexposure Ringan (γ=1.5)",
    "S2":  "Overexposure Berat (γ=2.5)",
    "S3":  "Underexposure Ringan (γ=0.6)",
    "S4":  "Underexposure Berat (γ=0.3)",
    "S5":  "Bayangan Parsial",
    "S6":  "Brightness Rendah (β=-40)",
    "S7":  "Brightness Tinggi (β=+40)",
    "S8":  "Saturation Rendah (-40%)",
    "S9":  "Saturation Tinggi (+40%)",
    "S10": "Exposure Rendah (α=0.7)",
    "S11": "Exposure Tinggi (α=1.3)",
}
 
SCENARIO_ORDER = ["S0","S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11"]
 
# Pengelompokan skenario untuk analisis
SCENARIO_GROUPS = {
    "Overexposure": ["S1", "S2"],
    "Underexposure":  ["S3", "S4"],
    "Bayangan":      ["S5"],
    "Brightness":    ["S6", "S7"],
    "Saturation":    ["S8", "S9"],
    "Exposure":      ["S10", "S11"],
}
 
 
# ──────────────────────────────────────────
#  BUAT data.yaml SEMENTARA PER SKENARIO
# ──────────────────────────────────────────
 
def create_scenario_yaml(scenario_code: str, lighting_dir: str) -> str:
    scenario_path = Path(lighting_dir) / scenario_code
    yaml_path     = scenario_path / "data.yaml"
    config = {
        "path":  str(scenario_path.resolve()),
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
#  EVALUASI SATU SKENARIO
# ──────────────────────────────────────────
 
def evaluate_scenario(
    model: YOLO,
    scenario_code: str,
    lighting_dir: str,
    img_size: int = 640,
    conf: float = CONF_THRESHOLD,
) -> dict | None:
    img_dir = Path(lighting_dir) / scenario_code / "images"
    if not img_dir.exists():
        print(f"  [SKIP] {scenario_code}: folder tidak ditemukan")
        return None
 
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    yaml_path = create_scenario_yaml(scenario_code, lighting_dir)
 
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
 
    return {
        "code":         scenario_code,
        "name":         SCENARIO_NAMES.get(scenario_code, scenario_code),
        "precision":    round(p,  4),
        "recall":       round(r,  4),
        "f1":           round(f1, 4),
        "map50":        round(metrics.box.map50, 4),
        "map5095":      round(metrics.box.map,   4),
        "ap_per_class": ap_per_class,
    }
 
 
# ──────────────────────────────────────────
#  HITUNG PENURUNAN PERFORMA
# ──────────────────────────────────────────
 
def compute_drop(results: list) -> list:
    baseline = next((r for r in results if r["code"] == "S0"), None)
    if not baseline:
        raise ValueError("Hasil S0 tidak ditemukan!")
 
    b_map50 = baseline["map50"]
    out = []
    for r in results:
        delta    = round(b_map50 - r["map50"], 4)
        drop_pct = round((delta / b_map50) * 100, 2) if b_map50 > 0 else 0.0
 
        # Kategori ketahanan
        if abs(drop_pct) <= 5:
            robustness = "Sangat Tahan"
        elif abs(drop_pct) <= 15:
            robustness = "Tahan"
        elif abs(drop_pct) <= 30:
            robustness = "Cukup Tahan"
        else:
            robustness = "Tidak Tahan"
 
        out.append({**r, "delta_map50": delta,
                    "drop_pct": drop_pct, "robustness": robustness})
    return out
 

def save_csv(data: list, output_path: str):
    fieldnames = [
        "code", "name", "precision", "recall", "f1",
        "map50", "map5095", "delta_map50", "drop_pct", "robustness"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    print(f"[INFO] CSV disimpan: {output_path}")
 

def print_results_table(data: list):
    header = (
        f"{'Kode':<5} {'Skenario':<35} "
        f"{'P':>6} {'R':>6} {'F1':>6} "
        f"{'mAP50':>7} {'mAP5095':>8} "
        f"{'ΔmAP':>7} {'Drop%':>7} {'Ketahanan'}"
    )
    sep = "─" * 105
    print(f"\n{sep}")
    print(header)
    print(sep)
 
    for r in data:
        flag = " ◀ baseline" if r["code"] == "S0" else ""
        print(
            f"{r['code']:<5} {r['name']:<35} "
            f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f} "
            f"{r['map50']:>7.4f} {r['map5095']:>8.4f} "
            f"{r['delta_map50']:>7.4f} {r['drop_pct']:>6.2f}%"
            f"  {r.get('robustness','')}{flag}"
        )
    print(sep)
 
    # Ringkasan
    non_baseline = [r for r in data if r["code"] != "S0"]
    if non_baseline:
        worst = max(non_baseline, key=lambda x: x["drop_pct"])
        best  = min(non_baseline, key=lambda x: x["drop_pct"])
        avg_drop = np.mean([r["drop_pct"] for r in non_baseline])
        print(f"\n  Baseline mAP@0.5      : {data[0]['map50']:.4f}")
        print(f"  Rata-rata Drop        : {avg_drop:.2f}%")
        print(f"  Penurunan terbesar    : {worst['code']} — {worst['name']} ({worst['drop_pct']:.2f}%)")
        print(f"  Penurunan terkecil    : {best['code']}  — {best['name']}  ({best['drop_pct']:.2f}%)")
 
 
def plot_map50_per_scenario(data: list, output_path: str):
    codes    = [r["code"]  for r in data]
    map50    = [r["map50"] for r in data]
    baseline = map50[0]
 
    colors = []
    for r in data:
        if r["code"] == "S0":
            colors.append("#2ecc71")
        elif r["drop_pct"] <= 5:
            colors.append("#3498db")
        elif r["drop_pct"] <= 15:
            colors.append("#f39c12")
        elif r["drop_pct"] <= 30:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")
 
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(codes, map50, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
 
    ax.axhline(y=baseline, color="#e74c3c", linestyle="--", linewidth=1.3,
               label=f"Baseline S0 = {baseline:.4f}")
    ax.set_title(
        "mAP@0.5 per Skenario Pencahayaan — YOLOv11n (setelah oversampling)\n"
        "Deteksi Penyakit Daun Padi",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Kode Skenario")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)
 
    for bar, val in zip(bars, map50):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
 
    # Legend warna ketahanan
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Baseline"),
        mpatches.Patch(color="#3498db", label="Sangat Tahan (drop ≤5%)"),
        mpatches.Patch(color="#f39c12", label="Tahan (drop 5–15%)"),
        mpatches.Patch(color="#e67e22", label="Cukup Tahan (drop 15–30%)"),
        mpatches.Patch(color="#e74c3c", label="Tidak Tahan (drop >30%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 1 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 2: Drop% per Skenario
# ──────────────────────────────────────────
 
def plot_drop_pct(data: list, output_path: str):
    non_baseline = [r for r in data if r["code"] != "S0"]
    codes    = [r["code"]     for r in non_baseline]
    drops    = [r["drop_pct"] for r in non_baseline]
    names    = [r["name"]     for r in non_baseline]
 
    colors = [
        "#3498db" if d <= 5 else
        "#f39c12" if d <= 15 else
        "#e67e22" if d <= 30 else "#e74c3c"
        for d in drops
    ]
 
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(codes, drops, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
 
    # Garis threshold ketahanan
    ax.axhline(y=5,  color="#3498db", linestyle=":", linewidth=1, alpha=0.7, label="Sangat Tahan ≤5%")
    ax.axhline(y=15, color="#f39c12", linestyle=":", linewidth=1, alpha=0.7, label="Tahan ≤15%")
    ax.axhline(y=30, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.7, label="Tidak Tahan >30%")
 
    ax.set_title(
        "Persentase Penurunan mAP@0.5 terhadap Baseline (Drop%)\n"
        "YOLOv11n — Deteksi Penyakit Daun Padi (setelah oversampling)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Kode Skenario")
    ax.set_ylabel("Drop mAP@0.5 (%)")
    ax.set_ylim(0, max(drops) * 1.25 + 2)
    ax.legend(fontsize=8, loc="upper right")
 
    for bar, val, name in zip(bars, drops, names):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
 
    # Label nama skenario di bawah (miring)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 2 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 3: Heatmap Metrik per Skenario
# ──────────────────────────────────────────
 
def plot_metrics_heatmap(data: list, output_path: str):
    metrics_keys   = ["precision", "recall", "f1", "map50", "map5095"]
    metrics_labels = ["Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.5:0.95"]
 
    codes  = [r["code"] for r in data]
    matrix = np.array([[r[k] for k in metrics_keys] for r in data])
 
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
 
    ax.set_xticks(range(len(metrics_labels)))
    ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(
        [f"{r['code']} — {r['name'][:28]}" for r in data],
        fontsize=8
    )
 
    # Anotasi nilai di setiap sel
    for i in range(len(codes)):
        for j in range(len(metrics_keys)):
            val   = matrix[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)
 
    ax.set_title(
        "Heatmap Metrik Evaluasi per Skenario Pencahayaan\n"
        "YOLOv11n — Deteksi Penyakit Daun Padi (setelah oversampling)",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Nilai Metrik")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 3 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 4: AP per Kelas per Skenario
# ──────────────────────────────────────────
 
def plot_ap_per_class(data: list, output_path: str):
    """Grouped bar: AP tiap kelas di setiap skenario."""
    # Kumpulkan AP per kelas per skenario
    class_ap = {name: [] for name in CLASS_NAMES}
    codes     = [r["code"] for r in data]
 
    for r in data:
        ap_dict = r.get("ap_per_class", {})
        for name in CLASS_NAMES:
            class_ap[name].append(ap_dict.get(name, 0.0))
 
    # Cek apakah ada data AP per kelas
    has_data = any(any(v > 0 for v in vals) for vals in class_ap.values())
    if not has_data:
        print("[INFO] Data AP per kelas tidak tersedia, grafik 4 dilewati.")
        return
 
    x      = np.arange(len(codes))
    width  = 0.13
    colors = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
 
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        offset = (i - len(CLASS_NAMES)/2 + 0.5) * width
        ax.bar(x + offset, class_ap[name], width, label=name,
               color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
 
    ax.set_title(
        "AP per Kelas per Skenario Pencahayaan\n"
        "YOLOv11n — Deteksi Penyakit Daun Padi",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Kode Skenario")
    ax.set_ylabel("Average Precision (AP@0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(codes, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    ax.axhline(y=0.717, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.6, label="mAP baseline")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 4 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  SIMPAN RINGKASAN TEKS
# ──────────────────────────────────────────
 
def save_summary_txt(data: list, output_path: str):
    lines = [
        f"  Model        : YOLOv11n",
        f"  Baseline mAP@0.5 (S0) : {data[0]['map50']:.4f}",
        f"  Conf threshold        : {CONF_THRESHOLD}",
        "",
        "  AP per kelas (baseline S0):",
    ]
    for name, ap in BASELINE_AP_PER_CLASS.items():
        lines.append(f"    {name:<22}: {ap:.3f}")
 
    lines += ["", "  Hasil per skenario:", "  " + "─" * 60]
    for r in data:
        lines.append(
            f"  {r['code']:<4} {r['name']:<35} "
            f"mAP50={r['map50']:.4f}  Drop={r['drop_pct']:>6.2f}%  [{r.get('robustness','')}]"
        )
 
    non_baseline = [r for r in data if r["code"] != "S0"]
    if non_baseline:
        worst    = max(non_baseline, key=lambda x: x["drop_pct"])
        best     = min(non_baseline, key=lambda x: x["drop_pct"])
        avg_drop = np.mean([r["drop_pct"] for r in non_baseline])
        lines += [
            "",
            f"  Rata-rata Drop    : {avg_drop:.2f}%",
            f"  Drop terbesar     : {worst['code']} ({worst['name']}) — {worst['drop_pct']:.2f}%",
            f"  Drop terkecil     : {best['code']} ({best['name']}) — {best['drop_pct']:.2f}%",
        ]
 
    lines += ["", "=" * 65]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Ringkasan disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi ketahanan YOLOv11n v2 (post-oversampling)"
    )
    parser.add_argument("--model",    default=MODEL_PATH,   help="Path ke best.pt")
    parser.add_argument("--lighting", default=LIGHTING_DIR, help="Folder dataset_lighting")
    parser.add_argument("--output",   default=RESULTS_DIR,  help="Folder output")
    parser.add_argument("--conf",     default=CONF_THRESHOLD, type=float,
                        help="Confidence threshold (default: 0.382)")
    args = parser.parse_args()
 
    Path(args.output).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    print(f"\n{'='*60}")
    print(f"  Evaluasi Ketahanan YOLOv11n v2 — Post Oversampling")
    print(f"  Model      : {args.model}")
    print(f"  Conf thr   : {args.conf}")
    print(f"  Device     : {device.upper()}")
    print(f"  Output     : {args.output}")
    print(f"{'='*60}\n")
 
    # Muat model
    model = YOLO(args.model)
    print(f"[INFO] Model dimuat. Kelas: {list(model.names.values())}\n")
 
    # ── Evaluasi semua skenario ──────────────────────────
    raw_results = []
    for code in SCENARIO_ORDER:
        print(f"  [{code}] {SCENARIO_NAMES.get(code, '')} ...", end=" ", flush=True)
        result = evaluate_scenario(
            model         = model,
            scenario_code = code,
            lighting_dir  = args.lighting,
            img_size      = IMG_SIZE,
            conf          = args.conf,
        )
        if result:
            raw_results.append(result)
            print(f"mAP50={result['map50']:.4f}  F1={result['f1']:.4f}  ✓")
        else:
            print("SKIP")
 
    if not raw_results:
        print("\n[ERROR] Tidak ada skenario yang berhasil dievaluasi.")
        print("        Pastikan lighting_simulation.py sudah dijalankan.")
        return
 
    # ── Hitung drop ──────────────────────────────────────
    final_results = compute_drop(raw_results)
 
    # ── Tampilkan tabel ──────────────────────────────────
    print_results_table(final_results)
 
    # ── Simpan output ────────────────────────────────────
    out = Path(args.output)
 
    save_csv(
        final_results,
        str(out / "robustness_results_v2.csv")
    )
    save_summary_txt(
        final_results,
        str(out / "robustness_summary_v2.txt")
    )
    plot_map50_per_scenario(
        final_results,
        str(out / "plot1_map50_per_scenario.png")
    )
    plot_drop_pct(
        final_results,
        str(out / "plot2_drop_pct.png")
    )
    plot_metrics_heatmap(
        final_results,
        str(out / "plot3_heatmap_metrics.png")
    )
    plot_ap_per_class(
        final_results,
        str(out / "plot4_ap_per_class.png")
    )
 
    print(f"\n{'='*60}")
    print(f"  SELESAI — Semua output tersimpan di:")
    print(f"  {out.resolve()}")
    print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    main()