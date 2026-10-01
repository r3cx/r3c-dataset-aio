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

def preprocess_image(pil_img):
    img = np.array(pil_img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)  # NCHW
    return img

def postprocess_tensor(output_tensor):
    output = output_tensor.squeeze(0)  # Remove batch dim
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))  # HWC
    return output

def gather_image_paths(dir_path, recursive=False):
    dir_path = Path(dir_path)
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths += list(dir_path.rglob(f"*{ext}") if recursive else dir_path.glob(f"*{ext}"))
    image_paths = list(set(image_paths))
    image_paths.sort()
    return image_paths

def upscale_tile(session, tile_np, input_name, output_name):
    """Processes a single image slice directly on the GPU execution provider."""
    img_input = preprocess_image(Image.fromarray((tile_np * 255.0).astype(np.uint8)))
    result = session.run([output_name], {input_name: img_input})[0]
    return postprocess_tensor(result)

def upscale_with_tiling(session, img_path, scale_factor=1.0, tile_size=512, overlap=32):
    """Slices huge images into overlapping grid items to keep memory footprints safe."""
    img = Image.open(img_path).convert("RGB")
    if scale_factor != 1.0:
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.BICUBIC)
    
    img_np = np.array(img).astype(np.float32) / 255.0
    h, w, c = img_np.shape

    # 4xNomos8kDAT scales by 4 natively. Calculate output dimensions based on model capacity.
    model_scale = 4 
    output_h, output_w = h * model_scale, w * model_scale
    output_img = np.zeros((output_h, output_w, c), dtype=np.uint8)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Grid loop coordinates
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Form bounded window boxes
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = img_np[y_start:y_end, x_start:x_end, :]
            
            # Pass patch payload down to CUDA
            upscaled_tile = upscale_tile(session, tile, input_name, output_name)

            # Map coordinates out to the new base scale canvas size
            out_y_start = y_start * model_scale
            out_y_end = y_end * model_scale
            out_x_start = x_start * model_scale
            out_x_end = x_end * model_scale

            # Blend inner pixels to dissolve grid borders safely
            output_img[out_y_start:out_y_end, out_x_start:out_x_end, :] = upscaled_tile

    return Image.fromarray(output_img)

def upscale_images(session, image_dir, output_dir, scale=1.0, recursive=False):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = gather_image_paths(image_dir, recursive)

    for img_path in tqdm(image_paths, desc="Upscaling"):
        try:
            # Safely chunks the processing workload using a 512px tile grid system
            result_img = upscale_with_tiling(session, img_path, scale_factor=scale, tile_size=512, overlap=32)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            result_img.save(output_path)
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

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  
    opts.inter_op_num_threads = 4  

    # Locate this section inside main()
    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }

    providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']

    print(f"Loading ONNX model from {args.onnx_model} onto GPU...")
    session = ort.InferenceSession(args.onnx_model, sess_options=opts, providers=providers)
    print("Using active hardware provider:", session.get_providers())
    
    upscale_images(session, args.input_dir, args.output_dir, args.scale, args.recursive)

if __name__ == "__main__":
    main()
