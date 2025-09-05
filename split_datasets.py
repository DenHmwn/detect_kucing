import os
import shutil
import random

# Path dataset asli (ganti sesuai lokasi dataset kamu)
SOURCE_DIR = r"D:\SP di semester 4\dataset kucing\n"

# Path tujuan di project
DEST_DIR = r"D:\SP di semester 4\projcet komvis (CNN)\datasets"

# Proporsi pembagian (80% train, 20% val)
train_split = 0.8  

# Buat folder tujuan kalau belum ada
os.makedirs(os.path.join(DEST_DIR, "train/non_cat"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "val/non_cat"), exist_ok=True)

# Ambil semua file gambar
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Acak biar random
random.shuffle(all_images)

# Hitung jumlah untuk train dan val
train_count = int(len(all_images) * train_split)

train_files = all_images[:train_count]
val_files = all_images[train_count:]

# Copy file ke train/cat
for f in train_files:
    shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, "train/non_cat", f))

# Copy file ke val/cat
for f in val_files:
    shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, "val/non_cat", f))

print(f"Total gambar: {len(all_images)}")
print(f"Train: {len(train_files)} | Val: {len(val_files)}")
print("✅ Pembagian dataset selesai!")
