import os
import sys
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import re
import time

# ---- paths ----
CSV_PATH = Path("data/output_frames_resx5/test_clip_resx5_label_mismatch.csv")
TEST_ROOT = Path("/data/Yaru/processed-new-dataset-ue5/test_scenes")  # scene/dist/after_tonemapping/*.exr
OUT_ROOT  = Path("videos_out_max_drop_jod")  # output base

# ---- params ----
FPS = 120
FINAL_SIZE = (1920, 1080)                # (width, height)
RESIZE_FILTER = cv2.INTER_LANCZOS4       # "Lanczos"
UPSCALE_FILTER = cv2.INTER_NEAREST       # final 1920x1080 using nearest neighbor
PAD_MISSING_OK = False                   # if True, skip missing frames; if False, raise
EXR_SUFFIX = "after_tonemapping"        # as you wrote it
ALLOWED_EXT = ".exr"


# --- enable EXR support in OpenCV ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_exr_to_u8_bgr(path: Path) -> np.ndarray:
    """Read EXR (float HDR), assume already tone-mapped; clamp to [0,1] then to uint8 BGR."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype in (np.float32, np.float64):
        im = np.clip(im, 0.0, 1.0)
        im = (im * 255.0 + 0.5).astype(np.uint8)
    elif im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:
        im = im[:, :, :3]
    return im

def width_from_height(h: int, aspect=(16, 9)) -> int:
    return int(round(h * aspect[0] / aspect[1]))

# ---- filename matching ----
# Match: "<idx>_<anyint>_after_tonemappling.exr" where <idx> can be zero-padded or not
_EXR_RE_TEMPLATE = r"^{}_\d+_{}\.exr$"

# cache directory listings to avoid repeated os.scandir calls
_dir_cache = {}

def _listdir_cached(path: Path):
    key = str(path)
    if key not in _dir_cache:
        _dir_cache[key] = [e.name for e in os.scandir(path) if e.is_file()]
    return _dir_cache[key]

def find_exr_for_index(frames_dir: Path, idx: int) -> Path | None:
    """
    Find the EXR that corresponds to a given frame index `idx`.
    Accept both no-padding and zero-padded (width 5) idx.
    """
    if not frames_dir.exists():
        return None
    files = _listdir_cached(frames_dir)

    # Try non-padded first
    pat1 = re.compile(_EXR_RE_TEMPLATE.format(idx, EXR_SUFFIX))
    # Try zero-padded width=5 (e.g., 00031)
    pat2 = re.compile(_EXR_RE_TEMPLATE.format(f"{idx:05d}", EXR_SUFFIX))

    # Prefer exact non-padded match if present
    for name in files:
        if pat1.match(name):
            return frames_dir / name
    for name in files:
        if pat2.match(name):
            return frames_dir / name
    return None

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
    out_name = f"{scene_name}_{dist}.mp4"
    out_path = out_dir / out_name
    # print(f'out_path {out_path}')
    # sys.exit()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, FINAL_SIZE, True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {out_path}")

    indices = range(int(start), int(start) + int(T), int(stride))
    target_size = (width_from_height(int(label_height)), int(label_height))  # (w, h)

    wrote = 0
    for idx in indices:
        frame_path = find_exr_for_index(src_dir, idx)
        if frame_path is None:
            msg = f"[{'SKIP' if PAD_MISSING_OK else 'ERR'}] Missing EXR for idx={idx} in {src_dir}"
            print(msg)
            sys.exit()
            if PAD_MISSING_OK:
                continue
            else:
                writer.release()
                if out_path.exists() and wrote == 0:
                    try: out_path.unlink()
                    except FileNotFoundError: pass
                raise FileNotFoundError(msg)

        im = read_exr_to_u8_bgr(frame_path)
        if im is None:
            msg = f"[{'SKIP' if PAD_MISSING_OK else 'ERR'}] Cannot read {frame_path}"
            print(msg)
            if PAD_MISSING_OK:
                continue
            else:
                writer.release()
                if out_path.exists() and wrote == 0:
                    try: out_path.unlink()
                    except FileNotFoundError: pass
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

    # clip_path_pattern example:
    # BurnedTrees-2/normal_with_TSR_from720x1280/frame_{:05d}.png
    def extract_dist(s: str) -> str:
        parts = s.split("/")
        # parts[0] = scene, parts[1] ≈ '<dist>_from...'
        return parts[1].split("_from")[0] if len(parts) > 1 else ""

    for _, row in df.iterrows():
        scene = str(row["scene_name"])
        start = int(row["start"])
        T = int(row["T"])
        stride = int(row["stride"])
        clip_pattern = str(row["clip_path_pattern"])
        dist = extract_dist(clip_pattern)
        drop_h = int(row["drop_jod_label"])
        max_h  = int(row["max_jod_label"])

        # Output dirs include OUT_ROOT / scene / dist / tag
        out_dir_drop = OUT_ROOT /  "video_drop_jod"
        out_dir_max  = OUT_ROOT /  "video_max_jod"

        try:
            process_clip(scene, dist, start, T, stride, max_h,  "max_jod",  out_dir_max)
        except Exception as e:
            print(f"[ERROR] max_jod {scene}/{dist}: {e}")
        try:
            process_clip(scene, dist, start, T, stride, drop_h, "drop_jod", out_dir_drop)
        except Exception as e:
            print(f"[ERROR] drop_jod {scene}/{dist}: {e}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        main()
    finally:
        print(f"\n✅ Total runtime: {time.perf_counter() - t0:.2f} s")
