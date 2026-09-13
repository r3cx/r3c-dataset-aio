import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime
import onnxruntime as ort

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def load_image(image_path, scale=1.0):
    img = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.BICUBIC)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)  # NCHW
    return img

def save_image(output_tensor, save_path):
    output = output_tensor.squeeze(0)  # Remove batch dim
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))  # HWC
    Image.fromarray(output).save(save_path)

def gather_image_paths(dir_path, recursive=False):
    dir_path = Path(dir_path)
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths += list(dir_path.rglob(f"*{ext}") if recursive else dir_path.glob(f"*{ext}"))
    image_paths = list(set(image_paths))
    image_paths.sort()
    return image_paths

def upscale_images(onnx_model_path, image_dir, output_dir, scale=1.0, recursive=False):
    print(f"Loading ONNX model from {onnx_model_path}")
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    os.makedirs(output_dir, exist_ok=True)
    image_paths = gather_image_paths(image_dir, recursive)

    for img_path in tqdm(image_paths, desc="Upscaling"):
        try:
            img_input = load_image(img_path, scale)
            result = session.run([output_name], {input_name: img_input})[0]
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            save_image(result, output_path)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upscale images in a folder using an ONNX model.")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save upscaled images.")
    parser.add_argument("--scale", type=float, default=1.5, help="Scaling factor for input images (default: 1.5).")
    parser.add_argument("--recursive", action="store_true", help="Process images in subfolders recursively.")
    args = parser.parse_args()

    # Load ONNX model with GPU (CUDA) support if available
    session = ort.InferenceSession(args.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Using provider:", session.get_providers())
    upscale_images(args.onnx_model, args.input_dir, args.output_dir, args.scale, args.recursive)

if __name__ == "__main__":
    main()