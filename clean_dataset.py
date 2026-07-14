"""
Dataset Cleaner - Fire Detection System
------------------------------------------------
Scans dataset/fire_images and dataset/non_fire_images, tries to open
every image, and deletes any that are broken/corrupted so training
doesn't crash.

HOW TO RUN:
    python clean_dataset.py
"""

import os
from PIL import Image

FOLDERS = ["dataset/fire_images", "dataset/non_fire_images"]

total_checked = 0
total_removed = 0

for folder in FOLDERS:
    if not os.path.isdir(folder):
        print(f"Folder not found, skipping: {folder}")
        continue

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        total_checked += 1

        try:
            with Image.open(filepath) as img:
                img.verify()  # checks the file is a valid, complete image

            # verify() closes the file, so re-open to fully load pixel data too
            # (verify() alone sometimes misses truncated files)
            with Image.open(filepath) as img:
                img.load()

        except Exception as e:
            print(f"Removing broken image: {filepath}  ({e})")
            os.remove(filepath)
            total_removed += 1

print("\n--------------------------------------")
print(f"Checked: {total_checked} images")
print(f"Removed: {total_removed} broken images")
print(f"Remaining: {total_checked - total_removed} good images")
print("--------------------------------------")
print("You can now run: python train_model.py")
