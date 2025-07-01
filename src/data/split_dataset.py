import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, target_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    class_names = os.listdir(source_dir)

    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)
        images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        total = len(images)
        train_count = int(split_ratio[0] * total)
        val_count = int(split_ratio[1] * total)

        splits = {
            'train': images[:train_count],
            'val': images[train_count:train_count + val_count],
            'test': images[train_count + val_count:]
        }

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for img_name in tqdm(splits[split], desc=f"{split}/{class_name}"):
                src_path = os.path.join(class_path, img_name)
                dst_path = os.path.join(split_dir, img_name)
                shutil.copy(src_path, dst_path)

split_dataset(
    source_dir="../dataset/train",
    target_dir="../dataset_split",
    split_ratio=(0.8, 0.1, 0.1)
)
