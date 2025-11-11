import os
import sys
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# ---- paths ----
CSV_PATH = Path("data/output_frames_resx5/test_clip_resx5_label_mismatch.csv")  # your filtered CSV
TEST_ROOT = Path("/data/Yaru/processed-new-dataset-ue5/test_scenes")  # frames live here: scene/dist/after_tonemapping/*.exr
OUT_ROOT  = "videos_out_max_drop_jod"  # output base

# ---- params ----
FPS = 120
FINAL_SIZE = (1920, 1080)                # (width, height)
RESIZE_FILTER = cv2.INTER_LANCZOS4       # "Lanczos"
UPSCALE_FILTER = cv2.INTER_NEAREST       # final 1920x1080 using nearest neighbor
PAD_MISSING_OK = False                   # if True, skip missing frames; if False, raise
ALLOWED_EXT = ".exr"                     # source frames are EXR

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_exr_to_u8_bgr(path: Path) -> np.ndarray:
    """Read EXR (float HDR), assume already tone-mapped; clamp to [0,1] then to uint8 BGR."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    # If EXR is float, clamp and convert; if already 8-bit, just ensure u8
    if im.dtype in (np.float32, np.float64):
        im = np.clip(im, 0.0, 1.0)
        im = (im * 255.0 + 0.5).astype(np.uint8)
    elif im.dtype != np.uint8:
        # Fallback: normalize to 0..255
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ensure 3 channels (drop alpha if present)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:
        im = im[:, :, :3]
    # OpenCV already BGR; if your EXRs are RGB, this is fine for VideoWriter
    return im

def width_from_height(h: int, aspect=(16, 9)) -> int:
    return int(round(h * aspect[0] / aspect[1]))

def process_clip(scene_name: str,
                 dist: str,
                 start: int,
                 T: int,
                 stride: int,
                 label_height: int,
                 tag: str,
                 out_dir: Path):
    """
    tag: 'max_jod' or 'drop_jod'
    label_height: height to Lanczos-resample to (e.g., 480, 720, 864, 1080)
    """
    src_dir = TEST_ROOT / scene_name / dist / "after_tonemapping"
    if not src_dir.exists():
        print(f"[WARN] Missing frames dir: {src_dir}")
        return
    ensure_dir(out_dir)
    # Build output filename (include label)
    out_name = f"{scene_name}_{dist}_{label_height}p_120fps.mp4"
    out_path = out_dir / out_name

    # Prepare writer (final size always 1920x1080)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, FINAL_SIZE, True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {out_path}")

    # Frame indices
    indices = range(int(start), int(start) + int(T), int(stride))
    target_size = (width_from_height(int(label_height)), int(label_height))  # (w, h)

    wrote = 0
    for idx in indices:
        frame_name = f"frame_{idx:05d}{ALLOWED_EXT}"
        frame_path = src_dir / frame_name
        im = read_exr_to_u8_bgr(frame_path)
        if im is None:
            msg = f"[{'SKIP' if PAD_MISSING_OK else 'ERR'}] Cannot read {frame_path}"
            print(msg)
            if PAD_MISSING_OK:
                continue
            else:
                writer.release()
                if out_path.exists() and wrote == 0:
                    out_path.unlink(missing_ok=True)
                raise FileNotFoundError(frame_path)

        # 1) Lanczos resample to the label resolution (height=label_height, width by 16:9)
        im_res = cv2.resize(im, target_size, interpolation=RESIZE_FILTER)

        # 2) Nearest-neighbor upscale to 1920x1080
        im_final = cv2.resize(im_res, FINAL_SIZE, interpolation=UPSCALE_FILTER)

        writer.write(im_final)
        wrote += 1

    writer.release()
    print(f"[OK] {tag} video saved: {out_path} ({wrote} frames @ {FPS}fps)")

def main():
    df = pd.read_csv(CSV_PATH)

    # Expect columns:
    # scene_name,start,T,stride,position,clip_path_pattern,clip_jod,drop_jod_label,max_jod_label
    # Extract 'dist' (the subfolder after scene) from clip_path_pattern like:
    # BurnedTrees-2/normal_with_TSR_from720x1280/frame_{:05d}.png
    def extract_dist(s: str) -> str:
        parts = s.split("/")
        # parts[0] = scene, parts[1] = dist, parts[2] = frame pattern...
        dist = parts[1].split('_from')[0]
        # return parts[1] if len(parts) > 1 else ""
        return dist

    for _, row in df.iterrows():
        scene = str(row["scene_name"])
        start = int(row["start"])
        T = int(row["T"])
        stride = int(row["stride"])
        clip_pattern = str(row["clip_path_pattern"])
        dist = extract_dist(clip_pattern)
        drop_h = int(row["drop_jod_label"])
        max_h  = int(row["max_jod_label"])

        # Output dirs with tag in the folder name as requested
        out_dir_drop = Path("video_drop_jod")
        out_dir_max  = Path("video_max_jod")

        # Make both videos
        try:
            process_clip(scene, dist, start, T, stride, max_h,  "max_jod",  out_dir_max)
        except Exception as e:
            print(f"[ERROR] max_jod {scene}/{dist}: {e}")
        try:
            process_clip(scene, dist, start, T, stride, drop_h, "drop_jod", out_dir_drop)
        except Exception as e:
            print(f"[ERROR] drop_jod {scene}/{dist}: {e}")


# python -m eval.make_max_drop_jod_videos
if __name__ == "__main__":
    main()
