import os
import random
import shutil

dataset = "dataset"

valid_img = os.path.join(dataset, "valid/images")
test_img = os.path.join(dataset, "test/images")

valid_lbl = os.path.join(dataset, "valid/labels")
test_lbl = os.path.join(dataset, "test/labels")

# gabungkan semua gambar
images = []

for img in os.listdir(valid_img):
    images.append(("valid", img))

for img in os.listdir(test_img):
    images.append(("test", img))

random.shuffle(images)

split = int(len(images) * 0.5)

new_val = images[:split]
new_test = images[split:]


def move_files(files, target):
    for src_folder, img in files:

        img_name = os.path.splitext(img)[0]

        src_img = os.path.join(dataset, src_folder, "images", img)
        src_lbl = os.path.join(dataset, src_folder, "labels", img_name + ".txt")

        dst_img = os.path.join(dataset, target, "images", img)
        dst_lbl = os.path.join(dataset, target, "labels", img_name + ".txt")

        shutil.move(src_img, dst_img)

        if os.path.exists(src_lbl):   # cek apakah label ada
            shutil.move(src_lbl, dst_lbl)


move_files(new_val, "valid")
move_files(new_test, "test")

print("Resplit selesai")