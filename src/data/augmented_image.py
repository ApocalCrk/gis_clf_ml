# === STEP 1: Install jika belum ada ===
# pip install albumentations opencv-python tqdm

import os
import cv2
import albumentations as A
from tqdm import tqdm

# === STEP 2: Konfigurasi Folder ===
base_input_dir = "../dataset/_train"
base_output_dir = "../dataset/train_aug"

target_classes = ["05_humus"]
augment_count = 4

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
])

def augment_class_folder(class_name):
    input_dir = os.path.join(base_input_dir, class_name)
    output_dir = os.path.join(base_output_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc=f"Augmenting {class_name}"):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            for i in range(augment_count):
                augmented = transform(image=image)
                aug_img = augmented['image']
                aug_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}.jpg"
                cv2.imwrite(os.path.join(output_dir, aug_filename), aug_img)

for class_name in target_classes:
    augment_class_folder(class_name)

print("✅ Augmentasi selesai untuk semua kelas.")
