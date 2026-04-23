import os
import cv2
import random
import numpy as np
from collections import defaultdict

BASE_DIR = "dataset_lapangan"
OUT_DIR = "dataset_lapangan_aug"

CONDITIONS = ["L1", "L2", "L3", "L4"]
TARGET_PER_CLASS = 15

# ===============================
# Helper
# ===============================
def read_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append([int(cls), x, y, w, h])
    return boxes

def save_label(path, boxes):
    with open(path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

# ===============================
# Augmentasi
# ===============================
def flip_horizontal(img, boxes):
    img = cv2.flip(img, 1)
    new_boxes = []
    for cls, x, y, w, h in boxes:
        new_boxes.append([cls, 1 - x, y, w, h])
    return img, new_boxes

def rotate(img, boxes, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    new_boxes = []
    for cls, x, y, bw, bh in boxes:
        # convert ke pixel
        px = x * w
        py = y * h

        coords = np.array([px, py, 1])
        new_coords = M @ coords

        x_new = new_coords[0] / w
        y_new = new_coords[1] / h

        new_boxes.append([cls, x_new, y_new, bw, bh])

    return rotated, new_boxes

# ===============================
# Proses per kondisi (L1-L4)
# ===============================
for cond in CONDITIONS:
    print(f"\nProcessing {cond}...")

    img_dir = os.path.join(BASE_DIR, cond, "images")
    lbl_dir = os.path.join(BASE_DIR, cond, "labels")

    out_img_dir = os.path.join(OUT_DIR, cond, "images")
    out_lbl_dir = os.path.join(OUT_DIR, cond, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # mapping class → list file
    class_map = defaultdict(list)

    for file in os.listdir(img_dir):
        if not file.endswith(".jpg"):
            continue

        img_path = os.path.join(img_dir, file)
        lbl_path = os.path.join(lbl_dir, file.replace(".jpg", ".txt"))

        boxes = read_label(lbl_path)
        if len(boxes) == 0:
            continue

        for box in boxes:
            class_map[box[0]].append(file)

    # ===============================
    # augment per class
    # ===============================
    for cls, files in class_map.items():
        print(f" Class {cls} → {len(files)} data")

        count = 0
        while count < TARGET_PER_CLASS:
            file = random.choice(files)

            img_path = os.path.join(img_dir, file)
            lbl_path = os.path.join(lbl_dir, file.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)
            boxes = read_label(lbl_path)

            # pilih augmentasi random
            aug_type = random.choice(["flip", "rotate"])

            if aug_type == "flip":
                img_aug, boxes_aug = flip_horizontal(img, boxes)
                suffix = "flip"

            else:
                angle = random.choice([-15, -10, 10, 15])
                img_aug, boxes_aug = rotate(img, boxes, angle)
                suffix = f"rot{angle}"

            new_name = f"{file.replace('.jpg','')}_{suffix}_{count}.jpg"

            cv2.imwrite(os.path.join(out_img_dir, new_name), img_aug)
            save_label(os.path.join(out_lbl_dir, new_name.replace(".jpg", ".txt")), boxes_aug)

            count += 1

print("\nDONE augmentasi L1-L4 🚀")