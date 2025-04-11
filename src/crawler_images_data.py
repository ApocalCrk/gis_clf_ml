import os
from PIL import Image
from tqdm import tqdm
from icrawler.builtin import GoogleImageCrawler

# -------------------- Configuration --------------------
soil_types = [
    "andosol soil",
    "alluvial soil",
    "latosol soil",
    "regosol soil",
    "organosol soil",
    "gleysol profile",
    "rendzina soil"
]

base_raw_dir = "dataset"
base_clean_dir = "dataset_clean"
max_images = 50
min_resolution = (200, 200)
target_size = (224, 224)
# -------------------------------------------------------

os.makedirs(base_clean_dir, exist_ok=True)

def convert_and_resize_images(folder_name):
    raw_path = os.path.join(base_raw_dir, folder_name)
    clean_path = os.path.join(base_clean_dir, folder_name)
    os.makedirs(clean_path, exist_ok=True)

    image_files = [
        f for f in os.listdir(raw_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]

    for i, filename in enumerate(tqdm(image_files, desc=f"Processing {folder_name}", unit="img")):
        filepath = os.path.join(raw_path, filename)
        try:
            img = Image.open(filepath).convert("RGB")
            img = img.resize(target_size)
            new_filename = f"{i}.png"
            img.save(os.path.join(clean_path, new_filename), format="PNG")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

# -------------------- Scraping + Resizing --------------------
for soil in soil_types:
    folder_name = soil.replace(" ", "_")
    save_path = os.path.join(base_raw_dir, folder_name)

    print(f"\n📦 Scraping: {soil} → {save_path}")
    google_crawler = GoogleImageCrawler(storage={'root_dir': save_path})
    google_crawler.crawl(keyword=soil + " close up photo", max_num=max_images, min_size=min_resolution)

    print(f"\n🧹 Resizing & renaming to numbered PNGs for: {folder_name}")
    convert_and_resize_images(folder_name)

print("\n✅ Done! Clean dataset with 224x224 PNGs saved in:", base_clean_dir)
