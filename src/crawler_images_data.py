import os
import shutil
import random
from PIL import Image
import imagehash
from pygoogle_image import image as pygoogle_image

def scrape_images(keywords_list, limit_per_keyword, main_output_folder):
    print("===== TAHAP 1: MEMULAI PROSES SCRAPING GAMBAR =====")
    if not os.path.exists(main_output_folder):
        os.makedirs(main_output_folder)
    for keyword in keywords_list:
        print(f"\n--- Mencari dan mengunduh untuk: '{keyword}' ---")
        try:
            pygoogle_image.download(keywords=keyword, limit=limit_per_keyword)
            print(f"✅ Selesai mengunduh untuk '{keyword}'.")
            folder_sumber = "images"
            folder_tujuan = os.path.join(main_output_folder, keyword.replace(" ", "_"))
            if os.path.exists(folder_sumber):
                if os.path.exists(folder_tujuan):
                    shutil.rmtree(folder_tujuan)
                shutil.move(folder_sumber, folder_tujuan)
                print(f"   -> Gambar dipindahkan ke folder: '{folder_tujuan}'")
        except Exception as e:
            print(f"❌ Terjadi kesalahan saat memproses '{keyword}': {e}")
    print("\n===== TAHAP 1 SELESAI: Semua gambar awal telah diunduh. =====")
    return main_output_folder

def preprocess_images(root_dir, target_size=(224, 224)):
    print("\n===== TAHAP 3: MEMULAI PRE-PROCESSING OTOMATIS =====")
    
    format_gambar = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    jumlah_gambar_diproses = 0
    jumlah_duplikat_dihapus = 0
    hashes = {}
    for subdir, dirs, files in os.walk(root_dir):
        if not files: continue
        print(f"Memproses folder: {os.path.basename(subdir)}")
        for file in files:
            file_path = os.path.join(subdir, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in format_gambar: continue
            try:
                img = Image.open(file_path)
                jumlah_gambar_diproses += 1
                img_hash = imagehash.average_hash(img)
                if img_hash in hashes:
                    os.remove(file_path)
                    jumlah_duplikat_dihapus += 1
                    continue
                else:
                    hashes[img_hash] = file_path
                img_resized = img.resize(target_size)
                img_resized.save(file_path)
            except Exception as e:
                print(f"   - Gagal memproses gambar {file}: {e}. Menghapus file...")
                try: os.remove(file_path)
                except OSError: pass
    print(f"\n===== TAHAP 3 SELESAI =====")
    print(f"Total gambar yang diproses setelah kurasi: {jumlah_gambar_diproses}")
    print(f"Total gambar duplikat yang dihapus: {jumlah_duplikat_dihapus}")
    print(f"Semua gambar yang tersisa telah diubah ukurannya menjadi: {target_size}")

def split_dataset(source_dir, split_dir, train_ratio=0.8):
    """
    Membagi dataset menjadi folder train dan validation.
    """
    print("\n===== TAHAP 4: MEMULAI PEMBAGIAN DATASET (SPLIT) =====")
    
    
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir) 
    os.makedirs(split_dir)

    
    train_dir = os.path.join(split_dir, 'train')
    validation_dir = os.path.join(split_dir, 'validation')
    os.makedirs(train_dir)
    os.makedirs(validation_dir)

    
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        
        os.makedirs(os.path.join(train_dir, class_name))
        os.makedirs(os.path.join(validation_dir, class_name))
        
        
        images = os.listdir(class_dir)
        random.shuffle(images)
        
        
        split_point = int(len(images) * train_ratio)
        
        
        train_images = images[:split_point]
        validation_images = images[split_point:]

        
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))
        
        
        for img in validation_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(validation_dir, class_name, img))
            
        print(f"Kelas '{class_name}': {len(train_images)} train, {len(validation_images)} validation.")

    print("\n===== TAHAP 4 SELESAI: Dataset telah dibagi menjadi train dan validation. =====")


if __name__ == "__main__":
    NAMA_FOLDER_RAW = "dataset_tanah_raw"
    NAMA_FOLDER_SPLIT = "dataset_tanah_split"
    
    LIST_TANAH_YOGYAKARTA = [
        "Tanah Andosol profil", "Tanah Regosol vulkanik", "Tanah Aluvial persawahan",
        "Tanah Grumusol kering retak", "Tanah Latosol merah", "Tanah Mediteran Terra Rossa"
    ]
    
    JUMLAH_GAMBAR_PER_KEYWORD = 500 
    UKURAN_TARGET_GAMBAR = (224, 224)
    RASIO_TRAIN = 0.8 

    folder_hasil_scrape = scrape_images(LIST_TANAH_YOGYAKARTA, JUMLAH_GAMBAR_PER_KEYWORD, NAMA_FOLDER_RAW)

    print("\n" + "="*70 + "\n " + "TAHAP 2: TINDAKAN MANUAL DIPERLUKAN" + "\n" + "="*70)
    print(f"\nSilakan buka folder '{folder_hasil_scrape}' dan hapus semua gambar yang TIDAK RELEVAN.")
    input("\n>>> TEKAN ENTER untuk melanjutkan jika Anda sudah selesai membersihkan gambar...")

    preprocess_images(folder_hasil_scrape, UKURAN_TARGET_GAMBAR)
    
    split_dataset(folder_hasil_scrape, NAMA_FOLDER_SPLIT, RASIO_TRAIN)

    print("\nSELURUH PROSES PIPELINE DATA TELAH SELESAI.")
    print(f"Dataset akhir yang siap untuk training ada di dalam folder: '{NAMA_FOLDER_SPLIT}'")