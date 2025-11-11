import os
import re
import time
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# ---- paths ----
CSV_PATH  = Path("data/output_frames_resx5/test_clip_resx5_label_mismatch.csv")
TEST_ROOT = Path("/data/Yaru/processed-new-dataset-ue5/test_scenes")  # scene/dist/after_tonemapping/*.exr
OUT_ROOT  = Path("videos_out_max_drop_jod")

# ---- params ----
FPS = 120
FINAL_SIZE = (1920, 1080)          # (width, height)
RESIZE_FILTER  = cv2.INTER_LANCZOS4   # "Lanczos"
UPSCALE_FILTER = cv2.INTER_NEAREST    # final 1920x1080 via nearest
ALLOWED_EXT = ".exr"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_exr_to_u8_bgr(path: Path) -> np.ndarray | None:
    """Read EXR (float HDR), tone-mapped already; clamp to [0,1] then uint8 BGR."""
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

def find_frames_dir(scene: str, dist: str) -> Path | None:
    """
    Try both 'after_tonemapping' and 'aftertonemapping' to be robust.
    """
    cand1 = TEST_ROOT / scene / dist / "after_tonemapping"
    cand2 = TEST_ROOT / scene / dist / "aftertonemapping"
    if cand1.exists(): return cand1
    if cand2.exists(): return cand2
    return None

_idx_anyint_suffix_re = re.compile(r"^(\d+)_\d+_.*\.exr$", re.IGNORECASE)

def list_all_exrs_sorted(frames_dir: Path) -> list[Path]:
    """
    List ALL .exr files and sort by the leading index before the first underscore.
    Falls back to alphanumeric sort if index not found.
    """
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower()==ALLOWED_EXT]
    def key_fn(p: Path):
        m = _idx_anyint_suffix_re.match(p.name)
        if m:
            try:
                return (0, int(m.group(1)))  # primary: numeric idx
            except Exception:
                pass
        return (1, p.name.lower())          # fallback: name sort
    return sorted(files, key=key_fn)

def process_dir_all_frames(scene: str, dist: str, label_height: int, tag: str, out_dir: Path):
    """
    Build a video from ALL frames found under scene/dist/after_tonemapping(ing).
    Resample -> label_height (Lanczos), then upscale -> 1920x1080 (nearest).
    """
    src_dir = find_frames_dir(scene, dist)
    if not src_dir:
        print(f"[WARN] Missing frames dir for {scene}/{dist}")
        return
    frames = list_all_exrs_sorted(src_dir)
    if not frames:
        print(f"[WARN] No EXR frames in {src_dir}")
        return

    ensure_dir(out_dir)
    out_name = f"{scene}_{dist}_{tag}_{label_height}p_120fps.mp4"
    out_path = out_dir / out_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, FINAL_SIZE, True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {out_path}")

    target_size = (width_from_height(int(label_height)), int(label_height))  # (w, h)

    wrote = 0
    for fp in frames:
        im = read_exr_to_u8_bgr(fp)
        if im is None:
            print(f"[ERR] Cannot read {fp}")
            writer.release()
            if out_path.exists() and wrote == 0:
                try: out_path.unlink()
                except FileNotFoundError: pass
            raise FileNotFoundError(fp)

        # 1) Lanczos to label resolution (keep 16:9)
        im_res = cv2.resize(im, target_size, interpolation=RESIZE_FILTER)
        # 2) Nearest-neighbor upscale to 1920x1080
        im_final = cv2.resize(im_res, FINAL_SIZE, interpolation=UPSCALE_FILTER)
        writer.write(im_final)
        wrote += 1

    writer.release()
    print(f"[OK] {tag} video saved: {out_path} ({wrote} frames @ {FPS}fps)")

def main():
    df = pd.read_csv(CSV_PATH)

    # clip_path_pattern examples:
    #   BurnedTrees-2/normal_with_TSR_from720x1280/frame_{:05d}.png
    # We only need scene and the 'dist' component (before '_from...').
    def extract_dist(s: str) -> str:
        parts = s.split("/")
        return parts[1].split("_from")[0] if len(parts) > 1 else ""

    for _, row in df.iterrows():
        scene   = str(row["scene_name"])
        dist    = extract_dist(str(row["clip_path_pattern"]))
        drop_h  = int(row["drop_jod_label"])
        max_h   = int(row["max_jod_label"])

        out_dir_max  = OUT_ROOT / scene / dist / "video_max_jod"
        out_dir_drop = OUT_ROOT / scene / dist / "video_drop_jod"

        try:
            process_dir_all_frames(scene, dist, max_h,  "max_jod",  out_dir_max)
        except Exception as e:
            print(f"[ERROR] max_jod {scene}/{dist}: {e}")
        try:
            process_dir_all_frames(scene, dist, drop_h, "drop_jod", out_dir_drop)
        except Exception as e:
            print(f"[ERROR] drop_jod {scene}/{dist}: {e}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        main()
    finally:
        print(f"\n✅ Total runtime: {time.perf_counter() - t0:.2f} s")
