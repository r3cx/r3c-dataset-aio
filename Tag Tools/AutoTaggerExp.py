"""
AutoTaggerExp.py — Dynamic Window WD1.4 Tagger Pipeline

Uses SmilingWolf's WD1.4 tagger models via ONNX Runtime to auto-tag images
with Danbooru-style tags. Uses dynamic window inference sized to the image's
short axis for full-body context, batched inference for speed, and composition
tag override to route full-image context tags through the global scan.

Pipeline:
    1. Preprocess images (dynamic window crops + global scan)
    2. Run ONNX inference on all crops (batched or sequential)
    3. Merge crop probabilities (max for objects, global-only for composition)
    4. Apply thresholds, filter undesired/kaomoji tags
    5. Write tags to .txt files (append or overwrite mode)

Usage:
    python AutoTaggerExp.py --data_dir "path/to/images" --general_threshold=0.5 \
        --character_threshold=0.6 --num_data_loader_workers=6 --frequency_tags \
        --mode_append
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import csv
import argparse
import os
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download
import onnxruntime

onnxruntime.get_device()  # Initialize ONNX runtime device

# -----------------------------------------------------------------------------
# Model Configuration — Edit TAGGERS to change which model(s) to use
# -----------------------------------------------------------------------------
TAGGERS = [
    # "wd-v1-4-swinv2-tagger-v2",
    # "wd-v1-4-convnextv2-tagger-v2",
    # "wd-v1-4-moat-tagger-v2",
    # "wd-swinv2-tagger-v3",
    # "wd-vit-tagger-v3",
    # "wd-convnext-tagger-v3",
    # "wd-vit-large-tagger-v3",
    "wd-eva02-large-tagger-v3",
]

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_SIZE = 448  # Fixed model input resolution

# Dynamic window configuration
WINDOW_SIZE_CAP = 0             # 0 = use image short axis, >0 = hard cap in pixels
WINDOW_STRIDE_RATIO = 0.9       # 90% stride -> 10% overlap between adjacent windows

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

# Kaomoji tags that look like noise (underscore-separated facial expressions)
DEFAULT_KAOMOJIS = '0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
KAOMOJI_SET = {t.strip() for t in DEFAULT_KAOMOJIS.split(",")}

# Files expected in each HuggingFace tagger repo
FILES = ["model.onnx", "selected_tags.csv"]
CSV_FILE = FILES[-1]
DEFAULT_WD14_TAGGER_REPO = "wd-v1-4-swinv2-tagger-v2"

FILETYPE_TXT = ".txt"
TAGGER_PATH = "../Taggers/"

# Error message prefixes
ERROR = "Error: "
FILE_OPEN_ERROR = "Failed to open file: "
CSV_HEADER_ERROR = "Unexpected .csv header format detected: "

# -----------------------------------------------------------------------------
# Composition Tags — Full-image context tags that must use global scan only
# -----------------------------------------------------------------------------
# These tags describe overall image composition, framing, subject count,
# or spatial relationships. Dynamic window crops still lack the full-image
# context needed to detect them reliably, so their probability is taken
# exclusively from the global scan crop (the last crop produced).
#
# Categories:
#   A. Composition / format / perspective
#   B. Subject count — must come from global scan
#   C. Spatial position — full-image relationship
#   D. Focus / framing tags — crops create artificial frames
#   E. Relationship / implied tags — full scene context required
#   F. Censorship — requires full image
#   G. Frame-edge artifacts
#   H. Length-variant features — crops truncate features
#   I. Symmetry / count features — crops break bilateral symmetry
#   K. Clothing / appearance — crops change garment classification
#   L. Character tags — character ID requires full-face + context

COMPOSITION_TAGS = {
    # ========================================================================
    # A. VIEW ANGLE / PERSPECTIVE / COMPOSITION / FORMAT
    # ========================================================================
    "dutch angle", "from above", "from behind", "from below", "from side",
    "multiple views", "sideways", "straight-on", "upside-down",
    "fisheye", "perspective", "vanishing point",
    "afterimage", "border", "inset border", "ornate border",
    "outside border", "rounded corners", "viewfinder",
    "chart", "reference sheet", "stats", "collage", 
    "from outside", "glitch", "isometric", "letterboxed", "pillarboxed",
    "lineup", "column lineup", "faux figurine", "negative space",
    "out of frame", "out-of-frame censoring",
    "partially underwater shot", "pov",
    "symmetry", "rotational symmetry",
    "tachi-e", "zoom layer", "projected inset", "card",
    "1koma", "2koma", "3koma", "4koma", "multiple 4koma", "5koma", 
    "comic", "left-to-right manga", "silent comic","segmented comic",
    "cover", "album cover", "cover page", "doujin cover", "dvd cover", 
    "fake cover", "magazine cover", "manga cover",
    "fake screenshot", "fake phone screenshot",

    # ========================================================================
    # B. SUBJECT COUNT — must come from global scan, not window crops
    # ========================================================================
    # Girls
    "1girl", "multiple girls", "2girls", "3girls", "4girls", "5girls", "6+girls",
    # Boys
    "1boy", "multiple boys", "2boys", "3boys", "4boys", "5boys", "6+boys",
    # Others
    "1other", "multiple others", "2others", "3others", "4others", "6+others",
    # Related
    "solo", "people", "crowd", "ambiguous gender",

    # ========================================================================
    # C. SPATIAL POSITION — full-image relationship, not body-relative pose
    #    Window crops lack full-image context and may misidentify spatial 
    #    relationships between the subject and their environment.
    # ========================================================================
    # On surfaces
    "on bed", "on floor", "on couch", "on chair", "on desk", "on table",
    "on bench", "on stool", "on grass", "on roof", "on ground",
    "on lap", "on person", "on vehicle", "on motorcycle", "on scooter",
    "on railing", "on shoulder", "on head",
    # Under
    "under table", "under tree", "under covers", "under kotatsu",
    # Against
    "against wall", "against glass", "against tree", "against railing",
    "against fourth wall",
    # Behind/in front
    "behind another",
    # In/through/containment
    "in box", "in container", "in tree", "in water", "in bucket", "in food",
    "in cup", "in palm", "in the face",
    "through wall", "through screen", "through medium",
    # Above
    "above clouds",
    # Subject matter
    "everyone", "absolutely everyone",
    "landscape", "nature", "no humans",
    "scenery", "still life",
        
    # ========================================================================
    # D. FOCUS / FRAMING TAGS — window crops create artificial focus or the
    #    lack thereof certain features of the subject.
    # ========================================================================
    "ass focus", "back focus", "breast focus", "eye focus",
    "foot focus", "hand focus", "hip focus",
    "solo focus", "male focus", "pectoral focus", "animal focus", 
    "other focus", "food focus", "vehicle focus", "text focus",
    # Body framing
    "head only", "headless",
    "portrait", "upper body", "cowboy shot", "feet out of frame",
    "full body", "wide shot", "very wide shot",
    "lower body", "head out of frame",
    # Out of frame
    "foot out of frame",
    # Other framing
    "close-up", "cropped legs", "cropped torso", "cropped arms",
    "cropped shoulders", "profile",

    # ========================================================================
    # E. RELATIONSHIP / IMPLIED — full scene context required
    #    Implied actions can be falsely detected from crops of the subjects.
    # ========================================================================
    "yuri", "yaoi",
    "implied sex", "implied yuri", "implied yaoi",
    "implied fellatio", "implied fingering", "implied masturbation",
    "implied kiss", "implied futanari", "implied extra ears",

    # ========================================================================
    # F. CENSORSHIP — requires full image; window crops of uncensored areas
    #    produce false "censored" from local patterns
    #    Consider if onlu the "censored" tag should be filtered as the other 
    #    tags can potentially be specifically identified from within crops.
    # ========================================================================
    "censored", "censored nipples",
    "mosaic censoring", "bar censor", "blank censor", "blur censor",
    "heart censor", "steam censor", "soap censor", "hair censor",
    "tail censor", "transparent censoring", "character censor",
    "convenient censoring", "pointless censoring",
    "censored text", "novelty censor", "light censor",

    # ========================================================================
    # G. FRAME-EDGE ARTIFACTS — window crop boundaries create false detections
    #    A crop edge cutting through a subject looks like they're "peeking" or
    #    "peeking out" from behind something.Eedge-relative tags belong here.
    # ========================================================================
    "peeking", "peeking out", "hiding", "taking cover", "around corner"

    # ========================================================================
    # H. LENGTH-VARIANT FEATURES — crops truncate features, changing apparent
    #    size. Such features may be misidentified leading to multiple lengths
    #    being detected from a single subject if they are long enough to be
    #    cropped into multiple windows. 
    #    - Long hair detected in global scan, Short hair detected in window
    # ========================================================================
    # Hair length
    "long hair", "very long hair", "absurdly long hair", "short hair", "medium hair",
    # Clothing
    "long sleeves", "short sleeves",
    "long skirt","long coat",
    "long dress", "short dress",
    # Body Features
    "huge breasts", "gigantic breasts", "large breasts",
    "medium breasts", "small breasts", "flat chest",
    "huge nipples", "small nipples", "large areolae",
    "huge ass", "thick thighs", "muscular",
    "large penis",  "huge penis", "small penis", "large testicles", "huge testicles",
    
    # ========================================================================
    # I. SYMMETRY / COUNT FEATURES — crops break bilateral symmetry detection
    #    A crop showing only one side of the face/body will misidentify
    #    symmetric features as asymmetric, and vice versa.
    # ========================================================================
    # --- BODY: Head/Face ---
     # Hair style — one side cropped → false "side ponytail" from "twintails" etc.
    "side ponytail", "hair bun", "asymmetrical hair",
    "one eye closed", "closed eyes", "one-eyed", "cyclops",
    "single blank eye", "single empty eye", "single eyebrow",
    "single tear", "single blush sticker",
    "single horn", "single antler", "single floppy ear", "single animal ear",
    "single bang", "single sidelock", "single braid", "single drill",
    "single hair bun", "single side bun", "single hair intake",
    "single hair streak", "single hair ring", "single hair tube",
    "single ear cover", "single earring", "single tooth",
    "single head wing", "single feather", "single flame",
    # --- BODY: Torso ---
    "single breast", "single inverted nipple", "single nipple piercing","single pasty",
    "single wing", "single mechanical wing",
    # --- BODY: Arms/Hands ---
    "single extra arm", "single mechanical arm", "single mechanical hand",
    "single bare arm", "single hand",
    # --- BODY: Legs/Feet ---
    "single mechanical leg", "single bare leg", "single bare foot",
    # --- CLOTHING: Arms/Hands ---
    "single bare shoulder", "single off shoulder",
    "asymmetrical sleeves",
    "single sleeve", "single detached sleeve",
    "single wide sleeve", "single sleeve cuff",
    "single sleeve past fingers", "single sleeve past wrist",
    "single glove", "single mechanical glove",
    "single fingerless glove", "single half glove", "single elbow glove",
    "single bridal gauntlet", "single gauntlet", "single mitten",
    "single unworn glove", "single handcuff", "single arm armor", 
    "single arm warmer", "single arm guard", "single arm cuff",
    "single wrist cuff", "single wrist guard", "single elbow pad",
    # --- CLOTHING: Torso ---
    "single pauldron", "single shoulder pad",
    "single epaulette", "single sode", "single vambrace",
    "single bracer", "single couter",
    "single strap", "single garter strap",
    "single suspender", "single stripe",
    "single vertical stripe", "single horizontal stripe",
    # --- CLOTHING: Legs/Feet ---
    "single thighhigh", "single over-kneehigh", "single kneehigh",
    "single hiphigh", "single leg pantyhose", "single fishnet legwear",
    "single leg warmer", "single legwear garter", "single leg bodysuit",
    "single pantsleg", "single detached legging", "single fishnet armwear",
    "single boot", "single armored boot", "single knee boot", "single ankle boot",
    "single thigh boot", "single shoe", "single sandal", "single slipper",
    "single sock", "single loose sock", "single sock removed",
    "single ankle cuff", "single knee pad"

    # ========================================================================
    # K. CLOTHING / APPEARANCE — crops change garment classification
    #    Tags that require full-body context to disambiguate, or whose defining
    #    features may be split across crops (garment extent, connected pieces,
    #    full-outfit classification).
    # ========================================================================
    # Garment classification ambiguity — crop can't tell dress from top+skirt,
    # or crop top from dress, without seeing where the garment ends.
    "dress", "crop top",
    # Bottom garments — crop of upper portion may confuse skirt vs dress bottom,
    # or pants vs shorts (distinction is purely hem length, which crops truncate).
    "skirt", "pants", "shorts",
    # Full-body connected garments — need to see torso+bottom connection to
    # distinguish from separate pieces.
    "swimsuit",
    "bikini", "one-piece swimsuit",
    "leotard", "bodysuit",
    "bodystocking",
    # Outfit classification — requires seeing the full ensemble to identify.
    "uniform", "school uniform",
    # Clothing state — crop may show a torn area that's a design detail,
    # or miss the tear entirely; global view confirms intent.
    "torn clothes",
    "partially unbuttoned", "partially unzipped",
    "asymmetrical clothes", "asymmetrical dress", "asymmetrical skirt",
    # Body coverage — requires seeing the full body to confirm what's absent.
    "nude", "partially undressed",
    "topless female", "topless male", "topless other",
    "bottomless",
    "no shirt", "no pants",
    # Bilateral coverage — crop may only show one side.
    "bare shoulders", "barefoot",

    # ========================================================================
    # L. CHARACTER TAGS — character identification requires full-face + context
    # ========================================================================
}

# -----------------------------------------------------------------------------
# Image Preprocessing
# -----------------------------------------------------------------------------

def preprocess_image(image, image_size):
    """Convert a PIL image to a padded square BGR tensor at the target size.

    Steps:
        1. RGB -> BGR channel swap (OpenCV convention)
        2. White-pad to a square (preserves aspect ratio)
        3. Resize to (image_size, image_size) with appropriate interpolation

    Returns:
        np.ndarray of shape (image_size, image_size, 3), dtype float32.
    """
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB -> BGR

    # Pad to square with white borders
    size = max(image.shape[0:2])
    pad_x, pad_y = size - image.shape[1], size - image.shape[0]
    pad_l, pad_t = pad_x // 2, pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    # Resize: INTER_AREA for shrinking, LANCZOS4 for expanding
    interp = cv2.INTER_AREA if size > image_size else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (image_size, image_size), interpolation=interp)
    return image.astype(np.float32)


def compute_window_size(h, w, cap):
    """Compute window size from the image's short axis, optionally capped.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        cap: Maximum window size (0 = no cap, use short axis).

    Returns:
        int — window size in pixels.
    """
    short_axis = min(h, w)
    if cap and cap > 0:
        return min(short_axis, cap)
    return short_axis


def generate_window_coords(h, w, window_size, stride_ratio):
    """Return a list of (y, x) top-left coordinates for a sliding window grid.

    A final row/column is always added to guarantee full edge coverage.
    """
    stride = max(1, int(window_size * stride_ratio))

    def _axis_positions(length):
        positions = []
        pos = 0
        while pos + window_size <= length:
            positions.append(pos)
            pos += stride
        if not positions or positions[-1] + window_size < length:
            positions.append(max(0, length - window_size))
        return positions

    rows = _axis_positions(h)
    cols = _axis_positions(w)
    return [(y, x) for y in rows for x in cols]


def extract_window_crop(image, y, x, window_size, target_size):
    """Crop a region from `image`, pad to square, and resize to `target_size`.

    Args:
        image: BGR numpy array (H, W, 3).
        y, x: Top-left corner of the crop window.
        window_size: Desired crop dimension (clamped to image bounds).
        target_size: Final dimension after resize (MODEL_SIZE).

    Returns:
        np.ndarray of shape (target_size, target_size, 3), dtype float32.
    """
    h, w = image.shape[0], image.shape[1]
    y1, y2 = y, min(y + window_size, h)
    x1, x2 = x, min(x + window_size, w)
    crop = image[y1:y2, x1:x2]

    # Pad to square with white
    ch, cw = crop.shape[0], crop.shape[1]
    pad_size = max(ch, cw)
    pad_t = (pad_size - ch) // 2
    pad_l = (pad_size - cw) // 2
    crop = np.pad(crop, ((pad_t, pad_size - ch - pad_t), (pad_l, pad_size - cw - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if pad_size > target_size else cv2.INTER_LANCZOS4
    crop = cv2.resize(crop, (target_size, target_size), interpolation=interp)
    return crop.astype(np.float32)


def preprocess_image_windows(image_pil):
    """Generate a list of preprocessed crop arrays for a single image.

    The window size is computed from the image's short axis so each crop
    captures full body/pose context. A full-image global scan is always
    appended as the last crop.

    Returns:
        List[np.ndarray] — each element is (MODEL_SIZE, MODEL_SIZE, 3) float32.
    """
    image = np.array(image_pil)[:, :, ::-1]  # RGB -> BGR
    h, w = image.shape[0], image.shape[1]
    crops = []

    window_size = compute_window_size(h, w, WINDOW_SIZE_CAP)

    if h > window_size or w > window_size:
        coords = generate_window_coords(h, w, window_size, WINDOW_STRIDE_RATIO)
        for y, x in coords:
            crops.append(extract_window_crop(image, y, x, window_size, MODEL_SIZE))
    else:
        crops.append(preprocess_image(image_pil, MODEL_SIZE))

    # Global scan is always the last crop
    crops.append(preprocess_image(image_pil, MODEL_SIZE))
    return crops

# -----------------------------------------------------------------------------
# Probability Merging
# -----------------------------------------------------------------------------

def merge_probabilities(probability_list, global_crop_idx):
    """Merge crop probabilities: max across crops, plus raw global scan probs.

    Returns (merged_max, global_prob) so the caller can override composition
    tags with the global scan value.
    """
    if len(probability_list) == 1:
        return probability_list[0], probability_list[0]
    return np.max(probability_list, axis=0), probability_list[global_crop_idx]

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_taggers(force_download=False, use_gpu=False):
    """Download (if needed) and load ONNX tagger models from HuggingFace."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    models = []
    for tagger in TAGGERS:
        tagger_dir = f"{TAGGER_PATH}{tagger}"
        if not os.path.exists(tagger_dir) or force_download:
            print(f"Downloading WD14 tagger from HF: SmilingWolf/{tagger}")
            for filename in FILES:
                hf_hub_download(f"SmilingWolf/{tagger}", filename, cache_dir=tagger_dir, force_download=True, force_filename=filename)
        else:
            print(f"Using existing WD14 tagger: SmilingWolf/{tagger}")

        print(f"Loading tagger model: {tagger}")
        session = onnxruntime.InferenceSession(f"{tagger_dir}/{FILES[0]}", None, providers=providers)
        models.append(session)

    print(f"Completed loading {len(TAGGERS)} tagger model(s) on {providers[0]}")
    return models


def load_tag_lists():
    """Parse selected_tags.csv into general and character tag lists."""
    csv_path = os.path.join(f"{TAGGER_PATH}{TAGGERS[0]}", CSV_FILE)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header, data = rows[0], rows[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"{CSV_HEADER_ERROR}{header}"

    general_tags = [row[1] for row in data if row[2] == "0"]
    character_tags = [row[1] for row in data if row[2] == "4"]
    return general_tags, character_tags

# -----------------------------------------------------------------------------
# Image Discovery
# -----------------------------------------------------------------------------

def gather_image_paths(dir_path, recursive=False):
    """Collect sorted, deduplicated image file paths from a directory."""
    print(f"Accessing {dir_path}")
    image_paths = []
    search_fn = dir_path.rglob if recursive else dir_path.glob
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(search_fn("*" + ext))

    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images.")
    return image_paths

# -----------------------------------------------------------------------------
# Data Loader — Multi-worker image preprocessing
# -----------------------------------------------------------------------------

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    """PyTorch dataset that loads images and produces preprocessed crop tensors."""
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            crops = preprocess_image_windows(image)
            tensors = [torch.tensor(c) for c in crops]
        except Exception as e:
            print(f"{FILE_OPEN_ERROR}{img_path}, {ERROR}{e}")
            return None
        return (tensors, img_path)


def discard_corrupted_in_batch(batch):
    """Collate function: filter out None entries from failed image loads."""
    return [item for item in batch if item is not None]


def setup_data_loader(dataset, num_workers):
    """Create a DataLoader with the given dataset and worker count."""
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=discard_corrupted_in_batch, drop_last=False)

# -----------------------------------------------------------------------------
# ONNX Inference
# -----------------------------------------------------------------------------

def run_inference_batched(crop_tensors, models):
    """Run all crops in a single batch per model."""
    batched_input = np.stack(crop_tensors, axis=0)

    model_results = []
    for model in models:
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: batched_input})[0]
        model_results.append(result)

    return np.mean(model_results, axis=0)


def run_inference_sequential(crop_tensors, models):
    """Process each crop individually through each model (lower memory)."""
    all_probs = []
    for crop in crop_tensors:
        crop_probs = []
        for model in models:
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            result = model.run([output_name], {input_name: [crop]})[0]
            crop_probs.append(result)
        all_probs.append(np.mean(np.array(crop_probs), axis=0)[0])
    return np.array(all_probs)

# -----------------------------------------------------------------------------
# Tag Selection & Output
# -----------------------------------------------------------------------------

def _build_composition_tag_indices(general_tags, character_tags):
    """Map COMPOSITION_TAGS (space-separated) to their CSV indices (underscore)."""
    comp_underscore = {t.replace(" ", "_") for t in COMPOSITION_TAGS}
    indices = set()
    for i, tag in enumerate(general_tags):
        if tag in comp_underscore:
            indices.add(i)
    offset = len(general_tags)
    for i, tag in enumerate(character_tags):
        if tag in comp_underscore:
            indices.add(offset + i)
    return indices


def _select_tags(tag_probs, global_probs, general_tags, character_tags, composition_indices, undesired, gen_thresh, char_thresh):
    """Filter tags by threshold, undesired list, and kaomoji filter.

    Composition tags use global_probs; all others use tag_probs (max across crops).
    Character tags always use global_probs.
    The first 4 probability entries (rating tags) are already stripped before call.

    Returns:
        (combined_tags, general_tag_text, character_tag_text, updated_frequencies)
    """
    combined = []
    gen_text_parts = []
    char_text_parts = []
    freq_updates = {}
    num_general = len(general_tags)

    for i, p in enumerate(tag_probs):
        if i in composition_indices or i >= num_general:
            p = global_probs[i]

        if i < num_general and p >= gen_thresh:
            raw_name = general_tags[i]
            if raw_name in KAOMOJI_SET:
                continue
            name = raw_name.replace("_", " ")
            if name in undesired:
                continue
            freq_updates[name] = freq_updates.get(name, 0) + 1
            gen_text_parts.append(name)
            combined.append(name)
        elif i >= num_general and p >= char_thresh:
            name = character_tags[i - num_general].replace("_", " ")
            if name in undesired:
                continue
            freq_updates[name] = freq_updates.get(name, 0) + 1
            char_text_parts.append(name)
            combined.insert(0, name)

    return combined, ", ".join(gen_text_parts), ", ".join(char_text_parts), freq_updates


def _write_tags(image_path, tag_text, mode_append):
    """Write tag string to the .txt file beside the image."""
    txt_path = os.path.splitext(image_path)[0] + FILETYPE_TXT
    if not mode_append:
        with open(txt_path, "wt", encoding="utf-8") as f:
            f.write(f"{tag_text}\n")
        return

    prefix = ", " if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 else ""
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"{prefix}{tag_text}\n")


def run_inference(image_with_crops, models, general_tags, character_tags, undesired, tag_frequencies, gen_thresh, char_thresh, mode_append, debug_print, use_batched):
    """Run the full tagging pipeline for a single image.

    Steps:
        1. Run ONNX inference on all crops (batched or sequential)
        2. Merge crop probabilities (max for objects, global for composition)
        3. Select tags above threshold
        4. Write .txt output file
    """
    image_path, crop_tensors = image_with_crops
    global_crop_idx = len(crop_tensors) - 1

    # --- Inference ---
    if use_batched:
        probability = run_inference_batched(crop_tensors, models)
    else:
        probability = run_inference_sequential(crop_tensors, models)

    # --- Merge probabilities ---
    merged, global_prob = merge_probabilities(probability, global_crop_idx)

    # Skip first 4 entries (rating tags)
    tag_probs = merged[4:]
    global_tag_probs = global_prob[4:]

    # --- Composition tag index lookup ---
    composition_indices = _build_composition_tag_indices(general_tags, character_tags)

    # --- Tag selection ---
    combined, gen_text, char_text, freq_updates = _select_tags(
        tag_probs, global_tag_probs, general_tags, character_tags,
        composition_indices, undesired, gen_thresh, char_thresh
    )

    tag_text = ", ".join(combined)

    if debug_print:
        all_tag_probs = []
        num_general = len(general_tags)
        for i, p in enumerate(merged[4:]):
            use_global = i in composition_indices or i >= num_general
            if use_global:
                p = global_prob[4 + i]
            raw = general_tags[i] if i < num_general else character_tags[i - num_general]
            thresh = gen_thresh if i < num_general else char_thresh
            if p >= thresh:
                src = "global" if use_global else "max"
                all_tag_probs.append((raw.replace("_", " "), p, thresh, src))
        all_tag_probs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n[{os.path.basename(image_path)}] Tags meeting threshold:")
        for tag, prob, thresh, src in all_tag_probs:
            print(f"  {tag:40s} {prob:.4f} (threshold: {thresh}, source: {src})")
        print()
        print(f"\n{image_path}:\n  Character tags: {char_text}\n  General tags: {gen_text}")

    # --- Write output ---
    _write_tags(image_path, tag_text, mode_append)

    for tag, count in freq_updates.items():
        tag_frequencies[tag] = tag_frequencies.get(tag, 0) + count
    return tag_frequencies

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def start_autotagger(cfg):
    """Run the full autotagging pipeline."""
    models = load_taggers(cfg.force_download, cfg.use_gpu)
    general_tags, character_tags = load_tag_lists()

    image_paths = gather_image_paths(Path(cfg.data_dir), cfg.recursive_gather)
    undesired = set(t for t in cfg.undesired_tags.split(",") if t)

    if cfg.num_data_loader_workers is not None:
        data_pairs = setup_data_loader(ImageLoadingPrepDataset(image_paths), cfg.num_data_loader_workers)
    else:
        data_pairs = [[(None, ip)] for ip in image_paths]

    tag_frequencies = {}

    for batch in tqdm(data_pairs, smoothing=0.0):
        for item in batch:
            if item is None:
                continue

            crop_tensors, image_path = item

            if crop_tensors is not None:
                crops = [c.detach().numpy() for c in crop_tensors]
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    crops = preprocess_image_windows(image)
                except Exception as e:
                    print(f"{FILE_OPEN_ERROR}{image_path}, {ERROR}{e}")
                    continue

            tag_frequencies = run_inference(
                (str(image_path), crops), models, general_tags, character_tags,
                undesired, tag_frequencies, cfg.general_threshold,
                cfg.character_threshold, cfg.mode_append, cfg.debug_print, cfg.use_batched
            )

    if cfg.frequency_tags:
        print("\nTag frequencies:")
        for tag, freq in sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {freq}")

    print("\nAutotagging Completed!")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def setup_argument_parser():
    """Build the argparse.ArgumentParser for this script."""
    parser = argparse.ArgumentParser(description="Dynamic window WD1.4 image autotagger.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images to tag")
    parser.add_argument("--num_data_loader_workers", type=int, default=None, help="Parallel workers for image loading (None = single-threaded)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default confidence threshold (overridden by category-specific flags)")
    parser.add_argument("--general_threshold", type=float, default=None, help="Confidence threshold for general tags")
    parser.add_argument("--character_threshold", type=float, default=None, help="Confidence threshold for character tags")
    parser.add_argument("--undesired_tags", type=str, default="", help="Comma-separated tags to exclude from output")
    parser.add_argument("--recursive_gather", action="store_true", help="Recursively search subdirectories for images")
    parser.add_argument("--frequency_tags", action="store_true", help="Print tag frequency report after processing")
    parser.add_argument("--force_download", action="store_true", help="Force re-download of tagger models from HuggingFace")
    parser.add_argument("--mode_append", action="store_true", help="Append tags to existing .txt instead of overwriting")
    parser.add_argument("--debug_print", action="store_true", help="Print per-image tag breakdown")
    parser.add_argument("--window_size_cap", type=int, default=WINDOW_SIZE_CAP, help=f"Maximum window size cap in pixels (0 = use image short axis, default: {WINDOW_SIZE_CAP})")
    parser.add_argument("--window_stride_ratio", type=float, default=WINDOW_STRIDE_RATIO, help=f"Window stride ratio (default: {WINDOW_STRIDE_RATIO})")
    parser.add_argument("--use_sequential", action="store_true", help="Use sequential inference instead of batched (lower memory)")
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDA GPU for inference (requires onnxruntime-gpu)")
    return parser


class DotDict(dict):
    """Dict subclass that supports dot-notation attribute access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.general_threshold is None:
        args.general_threshold = args.threshold
    if args.character_threshold is None:
        args.character_threshold = args.threshold

    WINDOW_SIZE_CAP = args.window_size_cap
    WINDOW_STRIDE_RATIO = args.window_stride_ratio

    cfg = DotDict(vars(args))
    cfg.use_batched = not args.use_sequential

    start_autotagger(cfg)
