import os
import shutil

# Path relatif terhadap root: ./sign_language/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # = ./sign_language/run
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # = ./sign_language

SIBI_DIR = os.path.join(BASE_DIR, "sibi_dataset")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "test_split.txt")
DEST_DIR = os.path.join(ROOT_DIR, "dataset")

# Baca daftar file test
with open(TEST_SPLIT_PATH, "r") as f:
    test_files = set(line.strip() for line in f)

# Proses folder per huruf
for label in os.listdir(SIBI_DIR):
    label_path = os.path.join(SIBI_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file_name in os.listdir(label_path):
        rel_path = f"{label}/{file_name}"
        src_path = os.path.join(label_path, file_name)

        subset = "test" if rel_path in test_files else "train"
        dst_path = os.path.join(DEST_DIR, subset, label, file_name)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)

# Hapus folder sibi_dataset setelah semua file dipindahkan
shutil.rmtree(SIBI_DIR)

print("✅ Semua file berhasil dipindahkan dan folder 'sibi_dataset' telah dihapus.")
