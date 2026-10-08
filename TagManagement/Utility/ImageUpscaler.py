"""
ImageUpscaler.py — Batch upscale a folder of images with an ONNX Real-ESRGAN model.

Design:
    * The model is ALWAYS run at its native ratio on the full-resolution image (its training
      distribution), never on pre-scaled inputs.
    * --scale sets the FINAL output size relative to the original (1 = same size, 2 = double,
      4 = quadscaled), independent of the model. After the native pass the result is resampled:
        - scale == native  -> no resample at all (zero cost, zero loss)
        - scale <  native  -> downscaled (Lanczos, or INTER_AREA when the reduction is >= 3x)
        - scale >  native  -> Lanczos upscaled
      Choose with --resample_filter (auto | lanczos | area).
    * The model's native ratio is measured with one tiny 16x16 probe inference, so no filename
      guessing is needed. --model_scale overrides it.
    * Tiling is automatic by default: when the single-pass output (at the model's native
      scale) fits under --auto_tile_threshold_mp it runs one seamless full-image pass — fastest
      and no blend seams. Larger images fall back to tiled inference, which caps VRAM per tile
      and blends overlapping regions with a 10px border padding. --no_tiling forces a single
      pass for every file; --force_tiling forces tiling for every file.

Usage:
    python ImageUpscaler.py --onnx_model path/to/4xNomos8kDAT.onnx --input_dir in --output_dir out --scale 2
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch            # Crucial: Must be imported BEFORE onnxruntime
import onnxruntime
import onnxruntime as ort

# Forces ONNX to scan and borrow PyTorch's embedded CUDA/cuDNN DLLs
try:
    onnxruntime.preload_dlls() 
except AttributeError:
    pass 

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
MP_IN_PIXELS = 1024 * 1024  # 1 MP is defined as 1024x1024

def postprocess_tensor(output_tensor):
    output = output_tensor.squeeze(0)  # Remove batch dim
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))  # HWC
    return output

def measure_model_scale(session, override=None):
    """Return the model's native multiplier (as a float, integer-checked).

    Prefers an explicit override; otherwise runs one 16x16 probe inference and reads the
    output shape, so any model file works regardless of its name.
    """
    if override:
        val = float(override)
        print(f"Model scale: {val} (override)")
    else:
        probe = np.zeros((1, 3, 16, 16), dtype=np.float32)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        probe_out = session.run([output_name], {input_name: probe})[0]
        val = probe_out.shape[2] / 16
        print(f"Model scale: {val} (measured via probe inference)")
    if abs(val - round(val)) > 1e-6:
        raise ValueError(f"Model scale factor {val} is not an integer; set --model_scale for special models")
    return val

def gather_image_paths(dir_path, recursive=False):
    """Collect image paths with pixel counts (reads headers only, never full decodes)."""
    dir_path = Path(dir_path)
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths += list(dir_path.rglob(f"*{ext}") if recursive else dir_path.glob(f"*{ext}"))
    image_paths = list(set(image_paths))
    image_paths.sort()

    results = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                results.append((p, img.width * img.height))
        except Exception as e:
            print(f"Skipping unreadable file {p}: {e}")
    return results

def select_images(image_dir, recursive=False, target_mp=None):
    """Gather images and apply the megapixel filter.

    Returns (selected, skipped), each a list of (path, pixel_count).
    An image is selected when pixel_count < target_mp * MP_IN_PIXELS.
    """
    all_images = gather_image_paths(image_dir, recursive)
    if target_mp is None:
        return all_images, []
    cutoff = target_mp * MP_IN_PIXELS
    selected = [(p, n) for p, n in all_images if n < cutoff]
    skipped = [(p, n) for p, n in all_images if n >= cutoff]
    return selected, skipped

def run_tile(session, tile_f32, input_name, output_name):
    """Runs the model on a float32 HWC tile in [0,1]. Returns uint8 HWC at native scale."""
    tensor = np.expand_dims(np.transpose(tile_f32, (2, 0, 1)), axis=0)  # NCHW
    result = session.run([output_name], {input_name: tensor})[0]
    return postprocess_tensor(result)

def choose_resample(filter_choice, native, scale):
    """Pick the final resampler: None (native, no-op), 'lanczos', or 'area'."""
    if filter_choice != "auto":
        return filter_choice
    if abs(scale - native) < 1e-6:
        return None
    if scale > native:
        return "lanczos"
    # Downsampling: Lanczos keeps anime lineart crisp; INTER_AREA is the alias-safer
    # choice once the reduction gets large
    return "area" if (native / scale) >= 3.0 else "lanczos"

def resample_pil(img, target_w, target_h, mode):
    arr = np.array(img)
    if mode == "area":
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(arr)

def decide_one_pass(width, height, native, no_tiling, force_tiling, auto_tile,
                    auto_tile_threshold_mp):
    """Pick the inference strategy for one image.

    Precedence: --no_tiling always wins (single pass), --force_tiling always tiling, and with
    auto tiling on a single seamless pass is used only when the native-scale output stays under
    the threshold (a proxy for single-pass VRAM). Falls back to tiling otherwise.
    """
    if no_tiling:
        return True
    if force_tiling:
        return False
    if not auto_tile or auto_tile_threshold_mp <= 0:
        return False
    out_mp = (width * native) * (height * native) / MP_IN_PIXELS
    return out_mp < auto_tile_threshold_mp

def upscale_one_pass(session, img_np, input_name, output_name):
    """Single full-image model call. No seams, but VRAM grows with image size."""
    return Image.fromarray(run_tile(session, img_np, input_name, output_name))

def upscale_tiled(session, img_np, native, tile_size=512, overlap=32, pad=10):
    """Slices the workload into a padded tile grid with 1/n weighted overlap blending.

    pad border pixels outside each tile (clamped at image edges) let the model see real
    context at tile edges; overlapping regions are averaged to dissolve grid borders.
    """
    native = int(native)  # Integer-checked by measure_model_scale
    if tile_size <= overlap:
        raise ValueError(f"tile_size ({tile_size}) must be greater than overlap ({overlap})")

    h, w = img_np.shape[:2]
    # Model scales by native natively. Calculate output dimensions based on model capacity.
    canvas = np.zeros((h * native, w * native, 3), dtype=np.uint8)
    weight = np.zeros((h * native, w * native), dtype=np.uint8)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    step = tile_size - overlap

    # Grid loop coordinates
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Form bounded window boxes
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            # Clamped border padding so tile-edge pixels have proper context
            py0 = max(0, y_start - pad)
            px0 = max(0, x_start - pad)
            py1 = min(h, y_end + pad)
            px1 = min(w, x_end + pad)

            # Pass padded patch down to CUDA, trim the padding back off natively
            up = run_tile(session, img_np[py0:py1, px0:px1, :], input_name, output_name)
            up = up[(y_start - py0) * native:(y_end - py0) * native,
                    (x_start - px0) * native:(x_end - px0) * native, :]

            # Map coordinates out to the new base scale canvas size
            oy0, oy1 = y_start * native, y_end * native
            ox0, ox1 = x_start * native, x_end * native

            # Weighted 1/n accumulation across tile overlaps dissolves grid borders safely
            # (canvas * weight always holds the accumulated tile sum per pixel)
            w_old = weight[oy0:oy1, ox0:ox1].astype(np.int16)
            w_total = w_old + 1
            canvas[oy0:oy1, ox0:ox1, :] = (
                (canvas[oy0:oy1, ox0:ox1, :].astype(np.int16) * w_old[..., None]
                 + up.astype(np.int16)) // w_total[..., None]
            ).astype(np.uint8)
            weight[oy0:oy1, ox0:ox1] = w_total

    return Image.fromarray(canvas)

def upscale_image(session, img, native, scale, resample_filter, one_pass=False,
                  tile_size=512, overlap=32, tile_pad=10):
    """Native-pass upscale of one RGB PIL image, then final resample to the target size."""
    img_np = np.array(img).astype(np.float32) / 255.0

    if one_pass:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        out_img = upscale_one_pass(session, img_np, input_name, output_name)
    else:
        out_img = upscale_tiled(session, img_np, native, tile_size, overlap, tile_pad)

    mode = choose_resample(resample_filter, native, scale)
    if mode is not None:
        target_w = max(1, int(round(img.width * scale)))
        target_h = max(1, int(round(img.height * scale)))
        out_img = resample_pil(out_img, target_w, target_h, mode)
    return out_img

def upscale_images(session, image_dir, output_dir, native, scale, resample_filter,
                   selected, no_tiling=False, force_tiling=False, auto_tile=True,
                   auto_tile_threshold_mp=3.0, tile_size=512, overlap=32, tile_pad=10):
    """Upscales each pre-selected image, mirroring its relative folder into the output dir.

    The inference strategy (single pass vs tiled) is decided per image via decide_one_pass, so a
    batch of mixed-size images can use a seamless single pass for small files and tiling for
    large ones in the same run.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = Path(image_dir)

    processed = 0
    failed = 0
    for img_path, pixels in tqdm(selected, desc="Upscaling"):
        try:
            with Image.open(img_path) as img:
                one_pass = decide_one_pass(
                    img.width, img.height, native, no_tiling, force_tiling,
                    auto_tile, auto_tile_threshold_mp)
                result_img = upscale_image(session, img.convert("RGB"), native, scale,
                                           resample_filter, one_pass, tile_size, overlap, tile_pad)
            # Mirror the relative folder structure so same-named files in different
            # subfolders do not overwrite each other in recursive mode
            output_path = Path(output_dir).joinpath(*img_path.relative_to(image_dir).parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path)
            processed += 1
        except Exception as e:
            failed += 1
            print(f"Failed to process {img_path}: {e}")

    print(f"Done. {processed} upscaled, {failed} failed.")

def main():
    parser = argparse.ArgumentParser(description="Upscale images in a folder using an ONNX model.")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save upscaled images.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="FINAL output size relative to the original image "
                             "(1 = same size, 2 = double, 4 = quadscaled). Independent of the "
                             "model's native multiplier (default: 1.0).")
    parser.add_argument("--model_scale", type=float, default=None,
                        help="Native multiplier of the model (e.g. 4 for a 4x model). "
                             "Measured via a probe inference when omitted.")
    parser.add_argument("--target_mp", type=float, default=None,
                        help="Only upscale images smaller than this many megapixels "
                             "(1 = 1024x1024). Omit to upscale all images.")
    parser.add_argument("--resample_filter", type=str, default="auto", choices=["auto", "lanczos", "area"],
                        help="Resampler for the final scale step (default: auto = Lanczos, "
                             "INTER_AREA when the reduction is >= 3x, no-op at native scale).")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile edge length in input pixels (default: 512).")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Overlap between adjacent tiles in pixels (default: 32).")
    parser.add_argument("--tile_pad", type=int, default=10,
                        help="Border padding outside each tile before the model call (default: 10).")
    parser.add_argument("--no_tiling", action="store_true",
                        help="Force a single full-image pass for every file (overrides auto "
                             "tiling). Best quality but VRAM grows with image size.")
    parser.add_argument("--force_tiling", action="store_true",
                        help="Force tiled inference for every file, even small ones that would "
                             "normally use a single seamless pass.")
    parser.add_argument("--auto_tile_threshold_mp", type=float, default=16.0,
                        help="Auto-tiling cutoff, in output megapixels measured at the model's "
                             "native scale. A single seamless pass is used when the output is "
                             "below this value, tiled inference at/above it. Set 0 to disable "
                             "auto (tiling unless --no_tiling). Default 16 fits most 8-24 GB "
                             "GPUs; raise it for more VRAM (or lower it to tile more eagerly).")
    parser.add_argument("--recursive", action="store_true",
                        help="Process images in subfolders recursively (mirrors the folder "
                             "structure into the output dir).")
    parser.add_argument("--dry_run", action="store_true",
                        help="List the images that would be processed (and skipped) without "
                             "loading the model.")
    args = parser.parse_args()

    if args.no_tiling and args.force_tiling:
        parser.error("--no_tiling and --force_tiling are mutually exclusive")
    if not args.no_tiling and args.tile_size <= args.overlap:
        parser.error("--tile_size must be greater than --overlap")
    if args.tile_pad < 0:
        parser.error("--tile_pad must be >= 0")

    # Selection runs before the model loads, which also keeps --dry_run cheap
    selected, skipped = select_images(args.input_dir, args.recursive, args.target_mp)
    if args.target_mp is not None:
        print(f"Size filter ({args.target_mp} MP): {len(selected)} selected, {len(skipped)} skipped.")

    if args.dry_run:
        print(f"Dry run: {len(selected)} image(s) would be upscaled.")
        for p, n in selected:
            print(f"  [upscale] {p}  ({n / MP_IN_PIXELS:.2f} MP)")
        for p, n in skipped:
            print(f"  [skipped] {p}  ({n / MP_IN_PIXELS:.2f} MP)")
        return

    if not selected:
        print("No images selected. Nothing to do.")
        return

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  
    opts.inter_op_num_threads = 4  

    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
    }

    providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']

    print(f"Loading ONNX model from {args.onnx_model} onto GPU...")
    session = ort.InferenceSession(args.onnx_model, sess_options=opts, providers=providers)
    print("Using active hardware provider:", session.get_providers())

    native = measure_model_scale(session, args.model_scale)

    if args.no_tiling:
        print("Tiling: off (single full-image pass for every file).")
    elif args.force_tiling:
        print(f"Tiling: forced ({args.tile_size}px tiles, {args.overlap}px overlap, "
              f"{args.tile_pad}px pad).")
    elif args.auto_tile_threshold_mp > 0:
        print(f"Auto tiling: single pass when output < {args.auto_tile_threshold_mp} MP, else "
              f"{args.tile_size}px tiles ({args.overlap}px overlap, {args.tile_pad}px pad).")
    else:
        print(f"Tiling: on ({args.tile_size}px tiles, {args.overlap}px overlap, "
              f"{args.tile_pad}px pad).")

    upscale_images(session, args.input_dir, args.output_dir, native, args.scale,
                   args.resample_filter, selected, no_tiling=args.no_tiling,
                   force_tiling=args.force_tiling,
                   auto_tile=args.auto_tile_threshold_mp > 0,
                   auto_tile_threshold_mp=args.auto_tile_threshold_mp,
                   tile_size=args.tile_size, overlap=args.overlap, tile_pad=args.tile_pad)

if __name__ == "__main__":
    main()
