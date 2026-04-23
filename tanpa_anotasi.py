import os
import csv
import cv2
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO
 
 
# ──────────────────────────────────────────
#  KONFIGURASI — SESUAIKAN PATH INI
# ──────────────────────────────────────────
# DRIVE_ROOT    = "/content/drive/MyDrive"
MODEL_PATH    = "yolov11n_baseline2/weights/best.pt"
FIELD_DIR     = "lapangan"   # folder foto lapangan
OUTPUT_DIR    = "hasil_evaluasi/field_qualitative"
IMG_SIZE      = 640
CONF_THRESHOLD = 0.2  # optimal dari F1 curve terbaru
 
CLASS_NAMES = [
    "healthy",
    "Brown Spot",
    "Leaf Blast",
    "Leaf Blight",
    "Leaf Scald",
    "Narrow Brown Spot",
]
 
FIELD_CONDITIONS = {
    "L1": {
        "name":          "Pagi Hari (07.00–08.00)",
        "analogy":       ["S1", "S6"],
        "karakteristik": "Cahaya rendah-sedang, suhu warna dingin, sudut rendah",
    },
    "L2": {
        "name":          "Siang Hari (11.00–13.00)",
        "analogy":       ["S3", "S7"],
        "karakteristik": "Cahaya tinggi, bayangan minimal, kontras tinggi",
    },
    "L3": {
        "name":          "Sore Hari (15.00–17.00)",
        "analogy":       ["S1", "S6"],
        "karakteristik": "Cahaya rendah-sedang, suhu warna hangat, sudut rendah",
    },
    "L4": {
        "name":          "Area Bayangan Tanaman",
        "analogy":       ["S5", "S2"],
        "karakteristik": "Pencahayaan tidak merata akibat naungan kanopi",
    },
}
 
FIELD_ORDER = ["L1", "L2", "L3", "L4"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}
 
# Warna bounding box per kelas (BGR untuk OpenCV)
CLASS_COLORS_BGR = {
    "Healthy":           (50,  205,  50),
    "Brown Spot":        (30,  144, 255),
    "Leaf Blast":        (0,   200, 200),
    "Leaf Blight":       (0,   100, 220),
    "Leaf Scald":        (180,  50, 220),
    "Narrow Brown Spot": (30,  180, 180),
}
 
 
# ──────────────────────────────────────────
#  INFERENSI SATU KONDISI
# ──────────────────────────────────────────
 
def run_inference_condition(
    model: YOLO,
    condition_code: str,
    field_dir: str,
    output_dir: str,
    conf: float,
    img_size: int,
) -> list:
    """
    Jalankan inferensi pada semua gambar satu kondisi.
    Simpan visualisasi bounding box.
    Kembalikan list dict hasil deteksi per gambar.
    """
    src_dir   = Path(field_dir) / condition_code
    out_dir   = Path(output_dir) / condition_code / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if not src_dir.exists():
        print(f"  [SKIP] Folder tidak ditemukan: {src_dir}")
        return []
 
    img_files = [f for f in src_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    if not img_files:
        print(f"  [SKIP] Tidak ada gambar di: {src_dir}")
        return []
 
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    per_image_results = []
 
    for img_path in sorted(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
 
        h, w = img.shape[:2]
 
        # Inferensi
        results = model.predict(
            source  = str(img_path),
            imgsz   = img_size,
            conf    = conf,
            device  = device,
            verbose = False,
        )[0]
 
        boxes     = results.boxes
        n_detect  = len(boxes) if boxes is not None else 0
 
        # Kumpulkan info deteksi
        detections  = []
        class_counts = Counter()
 
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id   = int(box.cls.item())
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                conf_val = round(float(box.conf.item()), 4)
                x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
                class_counts[cls_name] += 1
 
        # ── Gambar visualisasi ──────────────────────────
        viz = img.copy()
        for det in detections:
            color = CLASS_COLORS_BGR.get(det["class_name"], (128, 128, 128))
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            label_txt = f"{det['class_name']} {det['confidence']:.2f}"
            (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(viz, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(viz, label_txt, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
 
        # Watermark kondisi & jumlah deteksi
        cond_name = FIELD_CONDITIONS.get(condition_code, {}).get("name", condition_code)
        wm_text   = f"{condition_code}: {cond_name} | Deteksi: {n_detect}"
        cv2.rectangle(viz, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(viz, wm_text, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
 
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), viz)
 
        # Hitung rata-rata confidence
        avg_conf = round(np.mean([d["confidence"] for d in detections]), 4) \
                   if detections else 0.0
 
        per_image_results.append({
            "condition":    condition_code,
            "image":        img_path.name,
            "n_detections": n_detect,
            "avg_confidence": avg_conf,
            "detected":     1 if n_detect > 0 else 0,
            "class_counts": dict(class_counts),
            "detections":   detections,
        })
 
    return per_image_results
 
 
# ──────────────────────────────────────────
#  HITUNG STATISTIK PER KONDISI
# ──────────────────────────────────────────
 
def compute_condition_stats(per_image: list, condition_code: str) -> dict:
    if not per_image:
        return {}
 
    n_images      = len(per_image)
    n_detected    = sum(r["detected"] for r in per_image)
    detection_rate = round(n_detected / n_images * 100, 2)
    total_detect  = sum(r["n_detections"] for r in per_image)
    avg_detect    = round(total_detect / n_images, 2)
 
    conf_vals = [r["avg_confidence"] for r in per_image if r["avg_confidence"] > 0]
    avg_conf  = round(np.mean(conf_vals), 4) if conf_vals else 0.0
    min_conf  = round(min(conf_vals), 4)     if conf_vals else 0.0
    max_conf  = round(max(conf_vals), 4)     if conf_vals else 0.0
 
    # Distribusi kelas
    total_class = Counter()
    for r in per_image:
        total_class.update(r["class_counts"])
 
    cond_info = FIELD_CONDITIONS.get(condition_code, {})
    return {
        "condition":       condition_code,
        "name":            cond_info.get("name", condition_code),
        "karakteristik":   cond_info.get("karakteristik", ""),
        "analogy":         "+".join(cond_info.get("analogy", [])),
        "n_images":        n_images,
        "n_detected":      n_detected,
        "detection_rate":  detection_rate,
        "total_detections":total_detect,
        "avg_detections":  avg_detect,
        "avg_confidence":  avg_conf,
        "min_confidence":  min_conf,
        "max_confidence":  max_conf,
        "class_distribution": dict(total_class),
    }
 
 
# ──────────────────────────────────────────
#  PRINT TABEL RINGKASAN
# ──────────────────────────────────────────
 
def print_summary_table(stats_list: list):
    sep = "─" * 85
    print(f"\n{'='*85}")
    print(f"  RINGKASAN EVALUASI KUALITATIF DATA LAPANGAN")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"{'='*85}")
    print(f"  {'Kode':<5} {'Kondisi':<26} {'Gambar':>7} {'Terdet.':>7} "
          f"{'Rate%':>7} {'Avg Det':>8} {'Avg Conf':>9}")
    print(sep)
    for s in stats_list:
        print(
            f"  {s['condition']:<5} {s['name']:<26} "
            f"{s['n_images']:>7} {s['n_detected']:>7} "
            f"{s['detection_rate']:>6.1f}% "
            f"{s['avg_detections']:>8.2f} "
            f"{s['avg_confidence']:>9.4f}"
        )
    print(sep)
 
    # Distribusi kelas per kondisi
    print(f"\n  Distribusi Kelas Terdeteksi:")
    print(f"  {'Kondisi':<28}", end="")
    for cls in CLASS_NAMES:
        print(f"  {cls[:8]:>8}", end="")
    print()
    print(f"  {'─'*75}")
    for s in stats_list:
        print(f"  {s['name']:<28}", end="")
        for cls in CLASS_NAMES:
            cnt = s["class_distribution"].get(cls, 0)
            print(f"  {cnt:>8}", end="")
        print()
    print()
 
 
# ──────────────────────────────────────────
#  SIMPAN CSV
# ──────────────────────────────────────────
 
def save_summary_csv(stats_list: list, output_path: str):
    fieldnames = [
        "condition", "name", "karakteristik", "analogy",
        "n_images", "n_detected", "detection_rate",
        "total_detections", "avg_detections",
        "avg_confidence", "min_confidence", "max_confidence",
    ] + [f"cls_{n.replace(' ','_')}" for n in CLASS_NAMES]
 
    rows = []
    for s in stats_list:
        row = {k: s[k] for k in fieldnames
               if k in s and not k.startswith("cls_")}
        for cls in CLASS_NAMES:
            row[f"cls_{cls.replace(' ','_')}"] = \
                s["class_distribution"].get(cls, 0)
        rows.append(row)
 
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV kondisi disimpan: {output_path}")
 
 
def save_per_image_csv(all_results: list, output_path: str):
    fieldnames = [
        "condition", "image", "n_detections",
        "avg_confidence", "detected",
    ] + [f"cls_{n.replace(' ','_')}" for n in CLASS_NAMES]
 
    rows = []
    for r in all_results:
        row = {
            "condition":      r["condition"],
            "image":          r["image"],
            "n_detections":   r["n_detections"],
            "avg_confidence": r["avg_confidence"],
            "detected":       r["detected"],
        }
        for cls in CLASS_NAMES:
            row[f"cls_{cls.replace(' ','_')}"] = r["class_counts"].get(cls, 0)
        rows.append(row)
 
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV per gambar disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 1: Detection Rate per Kondisi
# ──────────────────────────────────────────
 
def plot_detection_rate(stats_list: list, output_path: str):
    codes  = [s["condition"]     for s in stats_list]
    names  = [s["name"]          for s in stats_list]
    rates  = [s["detection_rate"]for s in stats_list]
    colors = ["#e74c3c","#e67e22","#9b59b6","#1abc9c"]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(codes, rates, color=colors[:len(codes)],
                  edgecolor="white", linewidth=0.8, width=0.55)
 
    ax.set_title(
        "Detection Rate per Kondisi Lapangan\n"
        f"YOLOv11n | conf = {CONF_THRESHOLD}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
 
    for bar, val, name in zip(bars, rates, names):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height()/2,
                name, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
                wrap=True)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 1 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 2: Confidence per Kondisi (Box Plot)
# ──────────────────────────────────────────
 
def plot_confidence_boxplot(all_results: list, output_path: str):
    # Kumpulkan confidence per kondisi
    conf_per_cond = defaultdict(list)
    for r in all_results:
        for det in r["detections"]:
            conf_per_cond[r["condition"]].append(det["confidence"])
 
    codes_avail = [c for c in FIELD_ORDER if conf_per_cond[c]]
    data        = [conf_per_cond[c] for c in codes_avail]
    names       = [FIELD_CONDITIONS[c]["name"] for c in codes_avail]
    colors      = {"L1":"#e74c3c","L2":"#e67e22","L3":"#9b59b6","L4":"#1abc9c"}
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
 
    for patch, code in zip(bp["boxes"], codes_avail):
        patch.set_facecolor(colors.get(code, "#888"))
        patch.set_alpha(0.8)
 
    ax.set_title(
        "Distribusi Confidence Score per Kondisi Lapangan\n"
        f"YOLOv11n | conf = {CONF_THRESHOLD}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(range(1, len(codes_avail)+1))
    ax.set_xticklabels(
        [f"{c}\n{FIELD_CONDITIONS[c]['name'][:18]}" for c in codes_avail],
        fontsize=8.5
    )
    ax.set_ylabel("Confidence Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=CONF_THRESHOLD, color="#e74c3c", linestyle="--",
               linewidth=1, label=f"Threshold = {CONF_THRESHOLD}")
    ax.legend(fontsize=9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 2 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 3: Distribusi Kelas per Kondisi
# ──────────────────────────────────────────
 
def plot_class_distribution(stats_list: list, output_path: str):
    x      = np.arange(len(CLASS_NAMES))
    width  = 0.2
    lcolors = {"L1":"#e74c3c","L2":"#e67e22","L3":"#9b59b6","L4":"#1abc9c"}
 
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title(
        "Distribusi Kelas Terdeteksi per Kondisi Lapangan\n"
        f"YOLOv11n | conf = {CONF_THRESHOLD}",
        fontsize=12, fontweight="bold"
    )
 
    for i, s in enumerate(stats_list):
        code   = s["condition"]
        counts = [s["class_distribution"].get(cls, 0) for cls in CLASS_NAMES]
        offset = (i - len(stats_list)/2 + 0.5) * width
        bars   = ax.bar(x + offset, counts, width,
                        label=f"{code}: {s['name'][:14]}",
                        color=lcolors.get(code, "#888"),
                        edgecolor="white", linewidth=0.5, alpha=0.85)
 
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Jumlah Deteksi")
    ax.legend(fontsize=8, loc="upper right")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 3 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  GRAFIK 4: Deteksi per Gambar (Scatter)
# ──────────────────────────────────────────
 
def plot_detections_per_image(all_results: list, output_path: str):
    lcolors = {"L1":"#e74c3c","L2":"#e67e22","L3":"#9b59b6","L4":"#1abc9c"}
 
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        "Jumlah Deteksi per Gambar per Kondisi Lapangan\n"
        f"YOLOv11n | conf = {CONF_THRESHOLD}",
        fontsize=12, fontweight="bold"
    )
 
    offset_map = {"L1": -0.3, "L2": -0.1, "L3": 0.1, "L4": 0.3}
    legend_patches = []
 
    for code in FIELD_ORDER:
        subset = [r for r in all_results if r["condition"] == code]
        if not subset:
            continue
 
        x_vals = [i + offset_map.get(code, 0) for i in range(len(subset))]
        y_vals = [r["n_detections"] for r in subset]
        color  = lcolors.get(code, "#888")
 
        ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=50, zorder=3)
        ax.plot(x_vals, y_vals, color=color, alpha=0.4, linewidth=1)
        legend_patches.append(
            mpatches.Patch(color=color,
                           label=f"{code}: {FIELD_CONDITIONS[code]['name']}")
        )
 
    ax.set_xlabel("Indeks Gambar")
    ax.set_ylabel("Jumlah Deteksi")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot 4 disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  SIMPAN LAPORAN TEKS
# ──────────────────────────────────────────
 
def save_qualitative_report(stats_list: list, output_path: str):
    lines = [
        "=" * 65,
        "  LAPORAN EVALUASI KUALITATIF DATA LAPANGAN",
        "  Proposal Skripsi: Rangga Putra Sopyan (221210029)",
        "  Universitas Mercu Buana Yogyakarta — 2026",
        "=" * 65,
        "",
        f"  Model              : YOLOv11n",
        f"  Confidence threshold: {CONF_THRESHOLD}",
        f"  Metode evaluasi    : Kualitatif (tanpa ground truth)",
        "",
        "  CATATAN METODOLOGI:",
        "  Evaluasi ini tidak menggunakan anotasi ground truth,",
        "  sehingga metrik kuantitatif (mAP, Precision, Recall)",
        "  tidak tersedia. Hasil berupa analisis deskriptif",
        "  yang dapat melengkapi evaluasi kuantitatif S0-S11.",
        "",
        "─" * 65,
    ]
 
    for s in stats_list:
        lines += [
            "",
            f"  [{s['condition']}] {s['name']}",
            f"  Karakteristik : {s['karakteristik']}",
            f"  Skenario analog: {s['analogy']}",
            f"  Jumlah gambar  : {s['n_images']}",
            f"  Gambar terdeteksi: {s['n_detected']} ({s['detection_rate']:.1f}%)",
            f"  Total deteksi  : {s['total_detections']}",
            f"  Rata-rata deteksi/gambar: {s['avg_detections']:.2f}",
            f"  Confidence — Rata: {s['avg_confidence']:.4f} | "
            f"Min: {s['min_confidence']:.4f} | Max: {s['max_confidence']:.4f}",
            "  Distribusi kelas:",
        ]
        for cls in CLASS_NAMES:
            cnt = s["class_distribution"].get(cls, 0)
            if cnt > 0:
                lines.append(f"    {cls:<22}: {cnt}")
 
    lines += ["", "=" * 65]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Laporan disimpan: {output_path}")
 
 
# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi kualitatif data lapangan tanpa anotasi"
    )
    parser.add_argument("--model",  default=MODEL_PATH,   help="Path ke best.pt")
    parser.add_argument("--field",  default=FIELD_DIR,    help="Folder foto lapangan")
    parser.add_argument("--output", default=OUTPUT_DIR,   help="Folder output")
    parser.add_argument("--conf",   default=CONF_THRESHOLD, type=float,
                        help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    args = parser.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output).mkdir(parents=True, exist_ok=True)
 
    print(f"\n{'='*60}")
    print(f"  Evaluasi Kualitatif Data Lapangan (Tanpa Anotasi)")
    print(f"  Model    : {args.model}")
    print(f"  Conf thr : {args.conf}")
    print(f"  Device   : {device.upper()}")
    print(f"  Output   : {args.output}")
    print(f"{'='*60}\n")
 
    model = YOLO(args.model)
 
    all_results  = []
    stats_list   = []
 
    for code in FIELD_ORDER:
        src = Path(args.field) / code
        if not src.exists():
            print(f"  [{code}] SKIP — folder tidak ditemukan: {src}")
            continue
 
        n_imgs = len([f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS])
        if n_imgs == 0:
            print(f"  [{code}] SKIP — tidak ada gambar")
            continue
 
        cond_name = FIELD_CONDITIONS[code]["name"]
        print(f"\n[{code}] {cond_name} ({n_imgs} gambar) ...")
 
        per_image = run_inference_condition(
            model          = model,
            condition_code = code,
            field_dir      = args.field,
            output_dir     = args.output,
            conf           = args.conf,
            img_size       = IMG_SIZE,
        )
 
        stats = compute_condition_stats(per_image, code)
        all_results.extend(per_image)
        stats_list.append(stats)
 
        print(f"  ✓ Detection rate : {stats['detection_rate']:.1f}%")
        print(f"  ✓ Avg confidence : {stats['avg_confidence']:.4f}")
        print(f"  ✓ Avg deteksi/img: {stats['avg_detections']:.2f}")
 
    if not stats_list:
        print("\n[ERROR] Tidak ada kondisi lapangan yang berhasil diproses.")
        print(f"        Pastikan folder foto tersedia di: {args.field}")
        return
 
    # Tampilkan tabel
    print_summary_table(stats_list)
 
    # Simpan CSV
    out = Path(args.output)
    save_summary_csv(all_results and stats_list,
                     str(out / "summary_per_condition.csv"))
    save_per_image_csv(all_results, str(out / "summary_per_image.csv"))
 
    # Simpan grafik
    plot_detection_rate(stats_list,
                        str(out / "plot1_detection_rate.png"))
    plot_confidence_boxplot(all_results,
                            str(out / "plot2_confidence_boxplot.png"))
    plot_class_distribution(stats_list,
                             str(out / "plot3_class_distribution.png"))
    plot_detections_per_image(all_results,
                               str(out / "plot4_detections_per_image.png"))
 
    # Simpan laporan teks
    save_qualitative_report(stats_list,
                             str(out / "qualitative_report.txt"))
 
    print(f"\n{'='*60}")
    print(f"  SELESAI — Output tersimpan di:")
    print(f"  {out.resolve()}")
    print(f"\n  File output:")
    print(f"  ├── L1/detections/  ← foto dengan bounding box")
    print(f"  ├── L2/detections/")
    print(f"  ├── L3/detections/")
    print(f"  ├── L4/detections/")
    print(f"  ├── summary_per_condition.csv")
    print(f"  ├── summary_per_image.csv")
    print(f"  ├── plot1_detection_rate.png")
    print(f"  ├── plot2_confidence_boxplot.png")
    print(f"  ├── plot3_class_distribution.png")
    print(f"  ├── plot4_detections_per_image.png")
    print(f"  └── qualitative_report.txt")
    print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    main()