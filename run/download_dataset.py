import kagglehub
import os
import shutil

def download_and_move_kagglehub(dataset_slug, target_dir):
    # 1. Download dataset
    print("📥 Downloading dataset...")
    path = kagglehub.dataset_download(dataset_slug)
    print(f"📦 Dataset downloaded to: {path}")

    # 2. Cek folder dataset (misalnya: {path}/SIBI)
    source_folder = os.path.join(path, "SIBI")
    if not os.path.exists(source_folder):
        print("❌ Dataset tidak ditemukan di folder cache.")
        return

    # 3. Buat target_dir jika belum ada
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 4. Pindahkan folder
    destination = os.path.join(target_dir, "sibi_dataset")
    print(f"📂 Memindahkan dataset ke: {destination}")
    shutil.move(source_folder, destination)
    print("✅ Dataset berhasil dipindahkan.")

    # 5. Hapus cache dataset dari kagglehub
    username, dataset_name = dataset_slug.split("/")
    cache_base = os.path.expanduser("~/.cache/kagglehub/datasets")
    dataset_cache_path = os.path.join(cache_base, username, dataset_name)

    print(f"🧹 Menghapus cache di: {dataset_cache_path}")
    try:
        shutil.rmtree(dataset_cache_path)
        print("✅ Cache berhasil dihapus.")
    except Exception as e:
        print(f"⚠️ Gagal menghapus cache: {e}")

if __name__ == "__main__":
    dataset_slug = "alvinbintang/sibi-dataset"
    target_dir = os.path.join(".", "run")

    download_and_move_kagglehub(dataset_slug, target_dir)
