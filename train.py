from ultralytics import YOLO
from pathlib import Path
import torch
import yaml
import os
 
 
# ──────────────────────────────────────────
#  KONFIGURASI UTAMA
# ──────────────────────────────────────────
DATA_YAML    = "data.yaml"          # Path ke file konfigurasi dataset
PROJECT_DIR  = "runs/train"         # Folder output hasil training
EXP_NAME     = "yolov11n_baseline"  # Nama eksperimen
 
# Hyperparameter (sesuai Tabel 6 proposal)
EPOCHS       = 100
BATCH_SIZE   = 16
IMG_SIZE     = 640
LR0          = 0.01        # Learning rate awal
WEIGHT_DECAY = 0.0005
OPTIMIZER    = "AdamW"
 
# Pretrained weight YOLOv11n dari COCO
PRETRAINED   = "yolo11n.pt"        # Akan otomatis diunduh oleh Ultralytics
 
 
# ──────────────────────────────────────────
#  FUNGSI BUAT data.yaml OTOMATIS
#  (Jalankan jika belum punya data.yaml)
# ──────────────────────────────────────────
 
def create_data_yaml(
    dataset_root: str = "./dataset",
    output_path:  str = "data.yaml"
):
    """
    Buat file data.yaml sesuai format Ultralytics YOLO.
    Sesuaikan nama kelas dengan dataset Rice Leaf Disease.
    """
    config = {
        "path":  str(Path(dataset_root).resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc": 6,
        "names": [
            "Healthy",
            "Brown Spot",
            "Leaf Blast",
            "Leaf Blight",
            "Leaf Scald",
            "Narrow Brown Spot"
        ]
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] data.yaml berhasil dibuat: {Path(output_path).resolve()}")
    return output_path
 
 
# ──────────────────────────────────────────
#  FUNGSI PELATIHAN MODEL
# ──────────────────────────────────────────
 
def train_yolov11n(
    data_yaml:   str  = DATA_YAML,
    epochs:      int  = EPOCHS,
    batch:       int  = BATCH_SIZE,
    imgsz:       int  = IMG_SIZE,
    lr0:         float = LR0,
    weight_decay: float = WEIGHT_DECAY,
    optimizer:   str  = OPTIMIZER,
    project:     str  = PROJECT_DIR,
    name:        str  = EXP_NAME,
    pretrained:  str  = PRETRAINED,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  Model         : YOLOv11n")
    print(f"  Device        : {device.upper()}")
    print(f"  Dataset       : {data_yaml}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch Size    : {batch}")
    print(f"  Input Size    : {imgsz}x{imgsz}")
    print(f"  Optimizer     : {optimizer}")
    print(f"  Learning Rate : {lr0}")
    print(f"  Weight Decay  : {weight_decay}")
    print(f"  Output        : {project}/{name}")
    print(f"{'='*55}\n")
 
    # Muat model dengan pretrained weights COCO
    model = YOLO(pretrained)
    print(f"[INFO] Model dimuat: {pretrained}")
 
    # ── Mulai Pelatihan ──────────────────────────────────
    results = model.train(
        data          = data_yaml,
        epochs        = epochs,
        batch         = batch,
        imgsz         = imgsz,
        optimizer     = optimizer,
        lr0           = lr0,
        weight_decay  = weight_decay,
        project       = project,
        name          = name,
        device        = device,
 
        # Augmentasi Geometrik (AKTIF sesuai proposal)
        fliplr        = 0.5,   # Horizontal flip
        mosaic        = 1.0,   # Mosaic augmentation
 
        # Augmentasi Fotometrik (NONAKTIF sesuai proposal)
        # Dinonaktifkan agar evaluasi ketahanan pencahayaan
        # mencerminkan kemampuan generalisasi alami model
        hsv_h         = 0.0,   # Tanpa perubahan Hue
        hsv_s         = 0.0,   # Tanpa perubahan Saturation
        hsv_v         = 0.0,   # Tanpa perubahan Value/Brightness
        auto_augment  = None,  # Nonaktifkan AutoAugment
 
        # Pengaturan tambahan
        patience      = 20,    # Early stopping jika tidak ada peningkatan
        save          = True,  # Simpan model terbaik & terakhir
        save_period   = 10,    # Simpan checkpoint setiap 10 epoch
        verbose       = True,
        seed          = 42,    # Reproducibility
        val           = True,  # Validasi setelah setiap epoch
    )
 
    print(f"\n[INFO] Pelatihan selesai!")
    best_model_path = Path(project) / name / "weights" / "best.pt"
    print(f"[INFO] Model terbaik: {best_model_path.resolve()}")
    return results, str(best_model_path)
 
 
# ──────────────────────────────────────────
#  FUNGSI VALIDASI BASELINE (S0)
# ──────────────────────────────────────────
 
def validate_baseline(model_path: str, data_yaml: str = DATA_YAML):
    """
    Validasi model pada data test S0 (baseline).
    Hasil ini menjadi nilai referensi untuk perhitungan Drop(%).
    """
    print(f"\n{'='*55}")
    print(f"  Validasi Baseline (S0)")
    print(f"  Model: {model_path}")
    print(f"{'='*55}\n")
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_path)
 
    metrics = model.val(
        data   = data_yaml,
        split  = "test",     # Gunakan subset test
        imgsz  = IMG_SIZE,
        device = device,
        verbose= True,
    )
 
    print(f"\n── Metrik Baseline (S0) ──────────────────────")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"─────────────────────────────────────────────\n")
    return metrics
 
 
# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
 
if __name__ == "__main__":
 
    # ── LANGKAH 0: Buat data.yaml (skip jika sudah ada) ──
    if not Path(DATA_YAML).exists():
        print("[INFO] data.yaml tidak ditemukan, membuat otomatis...")
        create_data_yaml(dataset_root="./dataset", output_path=DATA_YAML)
    else:
        print(f"[INFO] Menggunakan data.yaml yang sudah ada: {DATA_YAML}")
 
    # ── LANGKAH 1: Latih model YOLOv11n ──────────────────
    results, best_model = train_yolov11n()
 
    # ── LANGKAH 2: Validasi pada data test baseline (S0) ──
    validate_baseline(model_path=best_model, data_yaml=DATA_YAML)
 
    print("\n[SELESAI] Model siap digunakan untuk pengujian ketahanan (S0-S11).")
    print(f"          Jalankan: robustness_evaluation.py --model {best_model}")