import os
import argparse
import shutil
from pathlib import Path
 
# ── Remap: index_lama → index_baru ──────────────────────────────────────────
CLASS_REMAP = {
    0: 1,   # Brown Spot        → 1
    1: 3,   # Leaf Blight       → 3
    2: 4,   # Leaf Scald        → 4
    3: 2,   # Leaf Blast        → 2
    4: 5,   # Narrow Brown Spot → 5
    5: 0,   # Healthy           → 0
}
 
CLASS_NAMES = {
    0: "Healthy",
    1: "Brown Spot",
    2: "Leaf Blast",
    3: "Leaf Blight",
    4: "Leaf Scald",
    5: "Narrow Brown Spot",
}
 
 
def is_polygon_line(parts: list) -> bool:
    """Polygon jika koordinat > 4 (lebih dari x_c y_c w h)."""
    return (len(parts) - 1) > 4
 
 
def polygon_to_bbox(coords):
    """Konversi koordinat polygon ke bbox YOLO (xc, yc, w, h) normalized."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return xc, yc, w, h
 
 
def clamp(v):
    return max(0.0, min(1.0, v))
 
 
def process_label_file(label_path, backup=True):
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()
 
    if not lines:
        return {"status": "skipped_empty", "path": str(label_path), "unknown_classes": set()}
 
    new_lines = []
    did_remap    = False
    did_convert  = False
    unknown_classes = set()
 
    for line in lines:
        line = line.strip()
        if not line:
            continue
 
        parts = line.split()
        try:
            old_class = int(parts[0])
            coords    = [float(v) for v in parts[1:]]
        except ValueError:
            new_lines.append(line)
            continue
 
        # Remap class
        if old_class not in CLASS_REMAP:
            unknown_classes.add(old_class)
            new_class = old_class
        else:
            new_class = CLASS_REMAP[old_class]
            if new_class != old_class:
                did_remap = True
 
        # Konversi polygon → bbox jika perlu
        if is_polygon_line(parts):
            did_convert = True
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            xc, yc, w, h = polygon_to_bbox(coords)
            xc, yc, w, h = clamp(xc), clamp(yc), clamp(w), clamp(h)
            new_lines.append(f"{new_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        else:
            coord_str = " ".join(parts[1:])
            new_lines.append(f"{new_class} {coord_str}")
 
    if not did_remap and not did_convert:
        status = "skipped_no_change"
    elif did_convert:
        status = "converted"
    else:
        status = "remapped"
 
    if did_remap or did_convert:
        if backup:
            shutil.copy2(label_path, label_path.with_suffix(".txt.bak"))
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
 
    return {
        "status": status,
        "path": str(label_path),
        "unknown_classes": unknown_classes,
    }
 
 
def process_dataset(dataset_dir, splits, backup, dry_run):
    dataset_path = Path(dataset_dir)
    summary = {
        "converted":         [],
        "remapped":          [],
        "skipped_no_change": [],
        "skipped_empty":     [],
        "not_found":         [],
        "errors":            [],
        "unknown_classes":   set(),
    }
 
    for split in splits:
        labels_dir = dataset_path / split / "labels"
 
        if not labels_dir.exists():
            print(f"  [WARN] Folder tidak ditemukan: {labels_dir}")
            summary["not_found"].append(str(labels_dir))
            continue
 
        txt_files = sorted(labels_dir.glob("*.txt"))
        print(f"\n{split}/labels → {len(txt_files)} file ditemukan")
 
        for label_path in txt_files:
            if label_path.suffix == ".bak":
                continue
 
            try:
                if dry_run:
                    with open(label_path, "r") as f:
                        lines = f.read().strip().splitlines()
                    valid_lines = [l for l in lines if l.strip()]
                    has_poly = any(is_polygon_line(l.split()) for l in valid_lines)
                    needs_remap = False
                    for l in valid_lines:
                        parts = l.split()
                        try:
                            old_c = int(parts[0])
                            if old_c in CLASS_REMAP and CLASS_REMAP[old_c] != old_c:
                                needs_remap = True
                                break
                        except:
                            pass
                    tag = []
                    if has_poly:    tag.append("polygon→bbox")
                    if needs_remap: tag.append("remap class")
                    print(f"  [DRY-RUN] {label_path.name}: {', '.join(tag) if tag else 'no change'}")
                    continue
 
                result = process_label_file(label_path, backup=backup)
                status = result["status"]
                summary[status].append(label_path.name)
                summary["unknown_classes"].update(result.get("unknown_classes", set()))
 
                if status == "converted":
                    print(f"  ✅ Converted+Remapped : {label_path.name}")
                elif status == "remapped":
                    print(f"  🔄 Remapped           : {label_path.name}")
 
            except Exception as e:
                print(f"  ❌ ERROR pada {label_path.name}: {e}")
                summary["errors"].append({"file": str(label_path), "error": str(e)})
 
    print("\n" + "=" * 60)
    print("RINGKASAN HASIL")
    print("=" * 60)
    print(f"  Polygon→BBox + Remap class  : {len(summary['converted'])}")
    print(f"  Remap class saja (sudah bbox): {len(summary['remapped'])}")
    print(f"  ⏭Tidak ada perubahan          : {len(summary['skipped_no_change'])}")
    print(f"  File kosong (di-skip)        : {len(summary['skipped_empty'])}")
    print(f"  Folder tidak ditemukan       : {len(summary['not_found'])}")
    print(f"  Error                         : {len(summary['errors'])}")
 
    if summary["unknown_classes"]:
        print(f"\n  Class index tidak dikenal (tidak di-remap): {summary['unknown_classes']}")
 
    if backup and not dry_run and (summary["converted"] or summary["remapped"]):
        print(f"\n  File asli di-backup sebagai .txt.bak")

    print("\n  Target class setelah konversi (sesuai data.yaml):")
    for idx, name in CLASS_NAMES.items():
        print(f"     {idx} = {name}")
    print("=" * 60)
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Konversi polygon→bbox + remap class index untuk dataset YOLO"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path root dataset (berisi train/, valid/, test/)")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Nama split folder (default: train valid test)")
    parser.add_argument("--no_backup", action="store_true",
                        help="Jangan buat backup .txt.bak (tidak disarankan)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Tampilkan rencana tanpa mengubah file")
    args = parser.parse_args()
 
    print("=" * 60)
    print("🌿 YOLO Polygon→BBox + Class Remapper")
    print("=" * 60)
    print(f"  Dataset dir : {args.dataset_dir}")
    print(f"  Splits      : {args.splits}")
    print(f"  Backup      : {'Tidak' if args.no_backup else 'Ya (.txt.bak)'}")
    print(f"  Dry run     : {'Ya' if args.dry_run else 'Tidak'}")
    print("\n  Remap class yang akan dilakukan:")
    for old, new in CLASS_REMAP.items():
        arrow = "→" if old != new else "="
        print(f"     {old} {arrow} {new}  ({CLASS_NAMES[new]})")
    print("=" * 60)
 
    process_dataset(
        dataset_dir=args.dataset_dir,
        splits=args.splits,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )
 
 
if __name__ == "__main__":
    main()