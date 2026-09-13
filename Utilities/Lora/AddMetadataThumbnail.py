import base64
import hashlib
import io
import json
import os
import shutil
from typing import Dict, Optional, Tuple

from PIL import Image, UnidentifiedImageError
from safetensors import safe_open
from safetensors.torch import save_file

# ============================================================
# Configuration
# ============================================================

# Execution Mode: "FOLDER" (batch processing) or "FILE" (single LoRA)
MODE = "FILE"  # Options: "FOLDER", "FILE"

# Target path (folder path if MODE="FOLDER", file path if MODE="FILE")
TARGET_PATH = r"F:\StableDiffusion\Models\Lora\Anima\Internal\Pending\Nebukurochan-Anima-A5.safetensors"

# Activation / Trigger Tag (e.g., "@ahonise"). Leave as "" to disable.
ACTIVATION_TAG = r"@nebukurochan"

# Toggle to clear dataset information in metadata
# NOTE THAT THIS OPERATION IS sDESTRUCTIVE AND CANNOT BE UNDONE
# True - Deletes dataset tags and info from the metadata
CLEAR_DATASET_INFO = False  

IMAGE_FORMAT = "JPEG"  # "PNG" or "JPEG"
JPEG_QUALITY = 90
TARGET_SIZE = 768

CREATE_BACKUP = False
DRY_RUN = False

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


# ============================================================
# Utility Helpers
# ============================================================

def sha256(data: bytes) -> str:
    """Computes the SHA-256 hexadecimal hash of raw bytes.

    Args:
        data (bytes): Binary data to hash.

    Returns:
        str: Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def human_size(size: int) -> str:
    """Converts a byte count into a human-readable megabyte string.

    Args:
        size (int): Size in bytes.

    Returns:
        str: Formatted string representing size in MB.
    """
    return f"{size / 1024 / 1024:.2f} MB"


def find_cover_image(base_path: str) -> Optional[str]:
    """Locates a matching cover image file using supported image extensions.

    Args:
        base_path (str): File path without extension (e.g., 'path/to/model').

    Returns:
        Optional[str]: Full path to the found image, or None if no match exists.
    """
    for ext in SUPPORTED_EXTENSIONS:
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate
    return None


# ============================================================
# Image Processing Functions
# ============================================================

def resize_long_edge(image: Image.Image) -> Image.Image:
    """Resizes an image so its longest edge matches TARGET_SIZE without upscaling.

    Maintains aspect ratio and handles transparent backgrounds (alpha channels)
    by converting them onto a solid white background.

    Args:
        image (Image.Image): Input PIL Image object.

    Returns:
        Image.Image: Resized RGB PIL Image object.
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")

    width, height = image.size
    max_dim = max(width, height)

    scale = min(1.0, TARGET_SIZE / max_dim)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def encode_image(image: Image.Image) -> bytes:
    """Encodes a PIL Image object into binary byte data based on config settings.

    Args:
        image (Image.Image): PIL Image to encode.

    Returns:
        bytes: Compressed binary image data (JPEG or PNG).
    """
    buffer = io.BytesIO()
    if IMAGE_FORMAT.upper() == "JPEG":
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    else:
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue()


def process_cover_image(cover_path: str) -> Optional[bytes]:
    """Opens, resizes, and encodes a cover image from a file path.

    Args:
        cover_path (str): Path to the image file.

    Returns:
        Optional[bytes]: Encoded image byte data, or None if processing fails.
    """
    try:
        with Image.open(cover_path) as img:
            processed_img = resize_long_edge(img)
            return encode_image(processed_img)
    except UnidentifiedImageError:
        print("Error: Unsupported image format.")
    except Exception as e:
        print(f"Image Error: {e}")
    return None


# ============================================================
# Safetensors Processing Functions
# ============================================================

def read_lora_data(lora_path: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Loads weight tensors and header metadata from a .safetensors file.

    Args:
        lora_path (str): Path to the .safetensors file.

    Returns:
        Tuple[Optional[Dict], Optional[Dict]]: A tuple containing (tensors_dict, metadata_dict),
        or (None, None) if reading fails.
    """
    tensors = {}
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = dict(f.metadata()) if f.metadata() else {}
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors, metadata
    except Exception as e:
        print(f"Safetensors Read Error: {e}")
        return None, None


def is_metadata_identical(metadata: Dict, new_image_bytes: bytes, target_description: Optional[str]) -> bool:
    """Compares existing metadata against proposed thumbnail data and activation tag.

    Args:
        metadata (Dict): Safetensors header metadata dictionary.
        new_image_bytes (bytes): Encoded bytes of the proposed thumbnail.
        target_description (Optional[str]): Expected 'ssmd_description' string or None.

    Returns:
        bool: True if both thumbnail and description are already identical, False otherwise.
    """
    # Check Thumbnail Match
    if "ssmd_cover_images" not in metadata:
        return False

    image_matches = False
    try:
        images = json.loads(metadata["ssmd_cover_images"])
        if len(images):
            existing_bytes = base64.b64decode(images[0])
            image_matches = (sha256(existing_bytes) == sha256(new_image_bytes))
    except Exception:
        image_matches = False


    desc_matches = False
    # Check Description Match
    if target_description:
        existing_description = metadata.get("ssmd_description")
        if existing_description == target_description:
            desc_matches = True

    
    if not desc_matches or not image_matches:
        return False

    return True


def handle_backup(lora_path: str) -> bool:
    """Creates a `.safetensors.backup` file if one does not already exist.

    Args:
        lora_path (str): Path to the original .safetensors file.

    Returns:
        bool: True if a new backup file was created (or would be created in dry run), False otherwise.
    """
    backup_path = lora_path + ".backup"
    if not os.path.exists(backup_path):
        if not DRY_RUN:
            shutil.copy2(lora_path, backup_path)
        print("Backup created." if not DRY_RUN else "[DRY RUN] Would create backup.")
        return True

    print("Backup already exists.")
    return False


def save_updated_lora(lora_path: str, tensors: Dict, metadata: Dict) -> None:
    """Saves updated tensors and metadata atomically to disk using a temporary file.

    Args:
        lora_path (str): Target file path for the .safetensors model.
        tensors (Dict): Weight tensors dictionary.
        metadata (Dict): Updated header metadata dictionary containing the cover image and description.
    """
    original_size = os.path.getsize(lora_path)

    if DRY_RUN:
        print("Status       : [DRY RUN] Would Update")
        print("Original Size:", human_size(original_size))
        return

    tmp_path = lora_path + ".tmp"
    save_file(tensors, tmp_path, metadata=metadata)
    os.replace(tmp_path, lora_path)

    updated_size = os.path.getsize(lora_path)
    diff = updated_size - original_size

    print("Status       : UPDATED")
    print("Original Size:", human_size(original_size))
    print("Updated Size :", human_size(updated_size))
    print(f"Difference   : {diff / 1024:.1f} KB")


# ============================================================
# Core Pipeline Execution
# ============================================================

def process_single_lora(lora_path: str) -> Tuple[str, bool]:
    """Orchestrates metadata embedding (thumbnail, activation tag, and/or tag frequency reset) for a LoRA file path.

    Args:
        lora_path (str): Full path to the .safetensors file.

    Returns:
        Tuple[str, bool]: (status_str, backup_created_flag)
    """
    if not os.path.exists(lora_path):
        print(f"Error: Target file does not exist: {lora_path}")
        return "error", False

    if not lora_path.lower().endswith(".safetensors"):
        print(f"Error: Target file is not a .safetensors file: {lora_path}")
        return "error", False

    base_path = os.path.splitext(lora_path)[0]

    # --- 1. Process Cover Image (Optional) ---
    cover_path = find_cover_image(base_path)
    encoded_bytes = None

    if cover_path:
        print("Cover        :", os.path.basename(cover_path))
        encoded_bytes = process_cover_image(cover_path)
        if encoded_bytes is None:
            print("Warning: Failed to process image, proceeding with metadata update only.")
    else:
        print("Cover        : None found")

    # --- 2. Format Activation Tag ---
    target_description = f"Activation Tag: {ACTIVATION_TAG.strip()}" if ACTIVATION_TAG.strip() else None

    # --- 3. Early Skip Guard Clause ---
    # Skip before reading heavy safetensors tensors if no operations are enabled/available
    if encoded_bytes is None and target_description is None and not CLEAR_DATASET_INFO:
        print("Skipped      : No cover image, activation tag, or dataset info clearing requested.")
        return "skipped", False

    # --- 4. Read Safetensors Data ---
    tensors, metadata = read_lora_data(lora_path)
    if tensors is None or metadata is None:
        return "error", False

    # --- 5. Update Metadata Header ---
    if encoded_bytes is not None:
        metadata["ssmd_cover_images"] = json.dumps([
            base64.b64encode(encoded_bytes).decode("utf-8")
        ])

    if target_description:
        metadata["ssmd_description"] = target_description
        print(f"Description  : {target_description}")

    if CLEAR_DATASET_INFO:
        metadata["ss_dataset_info"] = ""
        metadata["ss_datasets"] = ""
        metadata["ss_tag_frequency"] = ""
        print("Dataset Info : Cleared ('')")

    # --- 6. Backup & Save ---
    backup_created = False
    if CREATE_BACKUP:
        backup_created = handle_backup(lora_path)

    save_updated_lora(lora_path, tensors, metadata)
    return "updated", backup_created


def main():
    """Main execution function that handles single file or folder batch modes."""
    stats = {"updated": 0, "skipped": 0, "errors": 0, "backups": 0}

    mode = MODE.upper()

    if mode == "FILE":
        print("=" * 60)
        print(f"Processing Single File Mode: {os.path.basename(TARGET_PATH)}")
        print("=" * 60)

        status, backup_created = process_single_lora(TARGET_PATH)

        if status == "updated":
            stats["updated"] += 1
        elif status == "skipped":
            stats["skipped"] += 1
        elif status == "error":
            stats["errors"] += 1

        if backup_created:
            stats["backups"] += 1

    elif mode == "FOLDER":
        if not os.path.isdir(TARGET_PATH):
            print(f"Error: Target directory does not exist: {TARGET_PATH}")
            return

        for filename in sorted(os.listdir(TARGET_PATH)):
            if not filename.lower().endswith(".safetensors"):
                continue

            print("=" * 60)
            print(filename)

            full_path = os.path.join(TARGET_PATH, filename)
            status, backup_created = process_single_lora(full_path)

            if status == "updated":
                stats["updated"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "error":
                stats["errors"] += 1

            if backup_created:
                stats["backups"] += 1
    else:
        print(f"Error: Invalid MODE '{MODE}'. Expected 'FOLDER' or 'FILE'.")
        return

    print("\n" + "=" * 60)
    print("Finished")
    print("=" * 60)
    print(f"Updated : {stats['updated']}")
    print(f"Skipped : {stats['skipped']}")
    print(f"Errors  : {stats['errors']}")
    print(f"Backups : {stats['backups']}")


if __name__ == "__main__":
    main()