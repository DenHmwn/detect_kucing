import cv2
import os

# Path dataset asli (langsung ke folder cat atau non_cat)
SOURCE_DIR = r"D:\SP di semester 4\projcet komvis (CNN)\datasets\train\non_cat"

# Path tujuan dataset hasil resize (langsung ke folder yang sama / bisa beda)
DEST_DIR = r"D:\SP di semester 4\projcet komvis (CNN)\datasets\train\non_cat"

# Ukuran target untuk CNN (ubah sesuai kebutuhan model)
IMG_SIZE = (128, 128)

# Buat folder tujuan kalau belum ada
os.makedirs(DEST_DIR, exist_ok=True)

# Proses resize semua gambar di SOURCE_DIR
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Gagal membaca: {filename}")
            continue

        # Resize gambar
        resized = cv2.resize(img, IMG_SIZE)

        # Simpan hasil resize (overwrite file lama)
        save_path = os.path.join(DEST_DIR, filename)
        cv2.imwrite(save_path, resized)

print("✅ Resize selesai! Semua gambar tersimpan di:", DEST_DIR)
