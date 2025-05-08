import os
import shutil
from PIL import Image

# Path relatif terhadap root: ./sign_language/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ./sign_language/run
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # ./sign_language

SIBI_DIR = os.path.join(BASE_DIR, "sibi_dataset")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "test_split.txt")
DEST_DIR = os.path.join(ROOT_DIR, "dataset")
TEMP_DIR = os.path.join(BASE_DIR, "temp_processed")
IMAGE_SIZE = (224, 224)

# Bersihkan TEMP_DIR jika sudah ada
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR)

# Preprocess semua gambar terlebih dahulu ke TEMP_DIR
for label in os.listdir(SIBI_DIR):
    label_path = os.path.join(SIBI_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file_name in os.listdir(label_path):
        src_path = os.path.join(label_path, file_name)
        dst_path = os.path.join(TEMP_DIR, label, file_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img.save(dst_path)
        except Exception as e:
            print(f"❌ Gagal memproses {src_path}: {e}")

# Baca daftar file test
with open(TEST_SPLIT_PATH, "r") as f:
    test_files = set(line.strip() for line in f)

# Split ke train/test dari TEMP_DIR ke DEST_DIR
for label in os.listdir(TEMP_DIR):
    label_path = os.path.join(TEMP_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file_name in os.listdir(label_path):
        rel_path = f"{label}/{file_name}"
        src_path = os.path.join(label_path, file_name)

        subset = "test" if rel_path in test_files else "train"
        dst_path = os.path.join(DEST_DIR, subset, label, file_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)

# Cleanup
shutil.rmtree(SIBI_DIR)
shutil.rmtree(TEMP_DIR)

print("✅ Semua file berhasil dipreproses, displit, dan folder 'sibi_dataset' & 'temp_processed' telah dihapus.")