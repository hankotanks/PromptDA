import argparse
import os
import zlib
import numpy as np
import lz4.block
import cv2
import glob
import tqdm

import promptda.utils.logger as logger

def extract_color_from_video(path_rgb, max_size=1008, multiple_of=14):
    cap = cv2.VideoCapture(path_rgb)
    if not cap.isOpened(): 
        raise RuntimeError("Could not open video")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def extract_color_from_video_inner(cap, max_size, multiple_of):
        while True:
            ret, frame = cap.read()
            if not ret: 
                return None

            image = np.asarray(frame).astype(np.float32)
            image = image / 255.

            max_size = max_size // multiple_of * multiple_of
            if max(image.shape) > max_size:
                h, w = image.shape[:2]
                scale = max_size / max(h, w)
                tar_h = ensure_multiple_of(h * scale)
                tar_w = ensure_multiple_of(w * scale)
                image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)

            yield to_tensor_func(image)

    return extract_color_from_video_inner(cap, max_size, multiple_of), frame_count

def extract_depth_stray(path_depth):
    for path_depth_image in sorted(glob.glob(os.path.join(path_depth, "*.png"))):
        yield load_depth(path_depth_image)

    return None

def extract_depth_scannetpp(path_depth):
    height, width = 192, 256
    sample_rate = 1

    try:
        with open(path_depth, "rb") as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in range(0, depth.shape[0], sample_rate):
            yield to_tensor_func(depth)

    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(path_depth, "rb") as infile:
            while True:
                size = infile.read(4)   # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder="little")
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width).astype(np.float32)
                    depth /= 1000.
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width).astype(np.float32)

                yield to_tensor_func(depth)

                frame_id += 1

    return None

def load_stray(args):
    path_color = os.path.join(args.path_scene, "rgb.mp4")
    if not os.path.exists(path_color): 
        raise RuntimeError("RGB video does not exist")

    path_depth = os.path.join(args.path_scene, "depth")
    if not os.path.exists(path_depth): 
        raise RuntimeError("Depth images do not exist")

    path_out = os.path.join(args.path_scene, "depth_upscaled")
    os.makedirs(path_out, exist_ok=True)    
    
    color, frame_count = extract_color_from_video(path_color)
    depth = extract_depth_stray(path_depth)  

    return path_out, frame_count, color, depth

def load_scannetpp(args):
    path_iphone = os.path.join(args.path_scene, "iphone")
    if not os.path.exists(path_iphone): 
        raise RuntimeError("Scene does not exist")

    path_color = os.path.join(path_iphone, "rgb.mkv")
    if not os.path.exists(path_color): 
        raise RuntimeError("RGB video does not exist")

    path_depth = os.path.join(path_iphone, "depth.bin")
    if not os.path.exists(path_depth): 
        raise RuntimeError("Depth binary does not exist")

    path_out = os.path.join(path_iphone, "depth_upscaled")
    os.makedirs(path_out, exist_ok=True)

    color, frame_count = extract_color_from_video(path_color)
    depth = extract_depth_scannetpp(path_depth) 

    return path_out, frame_count, color, depth

LOADER_OPTIONS = ["stray", "scannetpp"]

path_out, frame_count, color, depth = None, None, None, None
if __name__ == "__main__":
    logger.Log.log_on = False

    parser = argparse.ArgumentParser()
    parser.add_argument("loader", help=f"loader to use depending on scene source [must be one of: {LOADER_OPTIONS}]")
    parser.add_argument("path_scene", help="path to scene folder")

    clargs = parser.parse_args()

    if clargs.loader not in LOADER_OPTIONS: 
        raise RuntimeError(f"Invalid loader supplied [must be one of: {LOADER_OPTIONS}]")

    if not os.path.exists(clargs.path_scene): 
        raise RuntimeError("Scene does not exist")
    
    if clargs.loader == "stray":
        path_out, frame_count, color, depth = load_stray(clargs)
    elif clargs.loader == "scannetpp":
        path_out, frame_count, color, depth = load_scannetpp(clargs)

    if path_out is None or frame_count is None or color is None or depth is None:
        raise RuntimeError("Unreachable!")
    
    with os.scandir(path_out) as entries:
        frame_count_out = sum(1 for entry in entries if entry.is_file())
        if abs(frame_count_out - frame_count) < 2:
            print("Upscaled depth imagery found. Skipping depth inference")
            exit(0)
    
    print(f"Detected {frame_count} frames from {clargs.loader} scene: {clargs.path_scene}")
    
from promptda.utils.io_wrapper import load_depth, save_depth, to_tensor_func, ensure_multiple_of
from promptda.promptda import PromptDA

if __name__ == "__main__":   
    print("Loading PromptDA model")

    DEVICE = "cuda"
    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(DEVICE).eval()  

    print("Finished loading PromptDA model")

    for frame_index in tqdm.tqdm(range(0, frame_count), desc="Inferring"):
        frame_color = next(color, None)
        if frame_color is None:
            print(f"Finished after {frame_index + 1} frames")
            break

        frame_depth = next(depth, None)
        if frame_color is not None and frame_depth is None:
            raise RuntimeError(f"Missing depth data associated with frame {frame_index}")
        
        tensor_color, tensor_depth = frame_color.to(DEVICE), frame_depth.to(DEVICE)
        upscaled = model.predict(tensor_color, tensor_depth)

        save_depth(upscaled.detach().cpu(), output_path=os.path.join(path_out, f"{frame_index:06d}.png"), save_vis=False)