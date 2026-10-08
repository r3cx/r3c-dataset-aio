import os
import shutil
from pathlib import Path

# Selects files in tag dir from the src dir
# Copies image from tag dir into output dir
# Copies tag from src dir into output dir

# Paths
tag_dir = Path(r"F:\StableDiffusion\Datasets\2 Baking\Serin199\1_Style")
src_dir = Path(r"F:\StableDiffusion\Datasets\2 Baking\Serin199\Ext\1_Style")
output_dir = Path(r"F:\StableDiffusion\Datasets\2 Baking\Serin199\Ext\Select")

# Supported image extensions
image_exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through images in tag_dir
for image_file in tag_dir.iterdir():
    if image_file.suffix.lower() not in image_exts:
        continue

    base_name = image_file.stem
    tag_file = src_dir / f"{base_name}.txt"

    # Copy image from tag_dir
    shutil.copy(image_file, output_dir / image_file.name)

    # Copy tag file from src_dir if it exists
    if tag_file.exists():
        shutil.copy(tag_file, output_dir / tag_file.name)
    else:
        print(f"[!] Tag file not found for: {image_file.name}")

print("Done. Images and their corresponding tag files have been copied.")