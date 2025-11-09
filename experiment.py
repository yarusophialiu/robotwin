import os
import re
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# =========================
# Config
# =========================
class Config:
    PTH_DIR = "ablation_exp_temporal_test_lr1e-05_20251108_074902"
    CKPT_NAME = "best.pth"
    TEST_DIR = "PATH/TO/Config.TEST_DIR"      # <-- set this
    OUTPUT_DIR = "predictions_out"
    START_RES = (1080, 1920)
    CLIP_LEN = 31                              # frames per hop (0..30, 31..61, ...)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_TO_TENSOR = T.Compose([T.ToTensor()])  # adapt normalization to match training

# =========================
# Checkpoint loading
# =========================
def load_checkpoint(ckpt_path: str):
    """
    Tries a few common layouts:
    - {'spatial': state_dict, 'head': state_dict, 'idx_to_label': list/dict}
    - {'model': {'spatial':..., 'head':...}, 'idx_to_label': ...}
    - {'state_dict': ...} where keys are 'spatial.*' and 'head.*'
    Returns: (spatial_module, head_module, idx_to_label)
    You MUST replace `build_spatial()` and `build_head(num_classes)` with your constructors.
    """
    blob = torch.load(ckpt_path, map_location="cpu")

    # You must implement these two to match your training code:
    def build_spatial() -> nn.Module:
        # e.g., return YourBackbone(...)
        raise NotImplementedError("Provide your spatial backbone definition here.")

    def build_head(num_classes: int) -> nn.Module:
        # e.g., return YourTemporalHead(num_classes=num_classes, ...)
        raise NotImplementedError("Provide your head definition here.")

    # Try to recover idx_to_label
    idx_to_label = None
    if isinstance(blob, dict):
        for k in ["idx_to_label", "IDX_TO_LABEL", "label_map", "classes"]:
            if k in blob:
                idx_to_label = blob[k]

    # Case A: explicit 'spatial' and 'head'
    if "spatial" in blob and "head" in blob:
        # Need num_classes for head init; infer from idx_to_label if present
        if idx_to_label is None:
            raise RuntimeError("Checkpoint missing idx_to_label. Please include mapping or hardcode it.")
        num_classes = len(idx_to_label) if hasattr(idx_to_label, "__len__") else int(max(idx_to_label)+1)
        spatial = build_spatial()
        head = build_head(num_classes)
        spatial.load_state_dict(blob["spatial"])
        head.load_state_dict(blob["head"])
        return spatial, head, idx_to_label

    # Case B: 'model' contains submodules
    if "model" in blob and isinstance(blob["model"], dict) and \
       "spatial" in blob["model"] and "head" in blob["model"]:
        if idx_to_label is None:
            raise RuntimeError("Checkpoint missing idx_to_label. Please include mapping or hardcode it.")
        num_classes = len(idx_to_label) if hasattr(idx_to_label, "__len__") else int(max(idx_to_label)+1)
        spatial = build_spatial()
        head = build_head(num_classes)
        spatial.load_state_dict(blob["model"]["spatial"])
        head.load_state_dict(blob["model"]["head"])
        return spatial, head, idx_to_label

    # Case C: single flat state_dict with 'spatial.' and 'head.' prefixes
    state_dict = blob.get("state_dict", blob if isinstance(blob, dict) else None)
    if isinstance(state_dict, dict) and any(k.startswith("spatial.") for k in state_dict.keys()):
        if idx_to_label is None:
            raise RuntimeError("Checkpoint missing idx_to_label. Please include mapping or hardcode it.")
        num_classes = len(idx_to_label) if hasattr(idx_to_label, "__len__") else int(max(idx_to_label)+1)
        spatial = build_spatial()
        head = build_head(num_classes)

        # split and load
        spatial_sd = {k.replace("spatial.", "", 1): v for k, v in state_dict.items() if k.startswith("spatial.")}
        head_sd    = {k.replace("head.", "", 1): v for k, v in state_dict.items() if k.startswith("head.")}
        spatial.load_state_dict(spatial_sd)
        head.load_state_dict(head_sd)
        return spatial, head, idx_to_label

    raise RuntimeError("Unrecognized checkpoint format. Please adapt the loader to your saved structure.")

# =========================
# Data helpers
# =========================
RES_RX = re.compile(r"_from(\d+)x(\d+)$")

def scan_test_tree(root: str) -> Dict[str, Dict[str, Dict[Tuple[int,int], str]]]:
    """
    Build mapping: scenes[scene][dist][(H,W)] -> folder path
    Accepts leaf folder names like: 'normal_from360x640'
    """
    scene_map: Dict[str, Dict[str, Dict[Tuple[int,int], str]]] = {}
    root_p = Path(root)
    for scene_dir in sorted([p for p in root_p.iterdir() if p.is_dir()]):
        scene = scene_dir.name
        scene_map.setdefault(scene, {})
        for leaf in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            m = RES_RX.search(leaf.name)
            if not m:
                continue
            H, W = int(m.group(1)), int(m.group(2))
            dist = leaf.name[:leaf.name.rfind("_from")]
            scene_map[scene].setdefault(dist, {})
            scene_map[scene][dist][(H, W)] = str(leaf)
    return scene_map

def list_frames(folder: str) -> List[str]:
    paths = sorted(Path(folder).glob("frame_*.png"))
    return [str(p) for p in paths]

def load_clip_tensor(frame_paths: List[str]) -> torch.Tensor:
    """
    Loads frames -> (T, C, H, W) in [0,1].
    Apply the SAME normalization you used in training if applicable.
    """
    imgs = []
    for fp in frame_paths:
        im = Image.open(fp).convert("RGB")
        imgs.append(Config.IMG_TO_TENSOR(im))
    return torch.stack(imgs, dim=0)  # (T,C,H,W)

# =========================
# Motion hook (implement to match training)
# =========================
def build_motion(rgb_clip: torch.Tensor) -> torch.Tensor:
    """
    Build a motion tensor of shape (1, C_m, T, H, W) to feed the head.
    Replace with your training-time motion pipeline (e.g., optical flow stacks).
    Placeholder: simple frame-diff stack with Cm=1.
    """
    # rgb_clip: (T, C, H, W) in [0,1]
    # Simple diff on luminance as placeholder
    with torch.no_grad():
        T_len, _, H, W = rgb_clip.shape
        if T_len < 2:
            diff = torch.zeros(1, 1, T_len, H, W, dtype=rgb_clip.dtype)
            return diff
        # convert to grayscale
        gray = 0.2989 * rgb_clip[:,0] + 0.5870 * rgb_clip[:,1] + 0.1140 * rgb_clip[:,2]  # (T,H,W)
        d = torch.zeros(T_len, H, W, dtype=rgb_clip.dtype)
        d[1:] = (gray[1:] - gray[:-1]).abs()
        d = d.unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)
        return d

# =========================
# Prediction glue (mirrors Trainer._forward_batch)
# =========================
@torch.no_grad()
def predict_res_for_clip(spatial: nn.Module,
                         head: nn.Module,
                         rgb_clip: torch.Tensor,
                         idx_to_label) -> Tuple[int, int]:
    """
    rgb_clip: (T,C,H,W), values in [0,1]
    idx_to_label: list/dict mapping class index -> resolution value (e.g., 360,480,720,1080)
                  If it's a dict of {idx: 720}, convert to list in order.
    Returns (H, W)
    """
    device = Config.DEVICE
    spatial.eval()
    head.eval()

    # Make inputs like Trainer: frames = rgb.view(B*T,C,H,W) with B=1
    T_len, C, H, W = rgb_clip.shape
    frames = rgb_clip.to(device).unsqueeze(0)          # (1,T,C,H,W)
    flat   = frames.view(T_len, C, H, W)               # (T,C,H,W)

    # Spatial per-frame -> (1, T, D)
    s_feats = spatial(flat).view(1, T_len, -1)

    # Motion -> (1, C_m, T, H, W)
    motion = build_motion(rgb_clip).to(device)

    # Head forward
    logits, ord_logits, _ = head(motion, s_feats, return_att=True)  # logits: (1, num_classes)
    pred_idx = int(torch.argmax(logits, dim=1).item())

    # idx_to_label could be list [360,480,720,1080] or dict {0:360,...}
    if isinstance(idx_to_label, dict):
        # ensure contiguous from 0..K-1
        label_val = idx_to_label[pred_idx]
    else:
        label_val = idx_to_label[pred_idx]

    # Map the scalar label to (H,W). If your label directly equals H (e.g., 720),
    # infer W by aspect ratio used in your test tree, e.g., 16:9 -> W = round(H/9*16) or read from folder names.
    # Safer: read available resolutions on disk and pick the one with this H.
    return int(label_val), infer_width_from_disk_if_possible(int(label_val))

def infer_width_from_disk_if_possible(H_pred: int) -> int:
    """
    If your widths are fixed (e.g., 16:9), replace with a direct mapping.
    Otherwise, this function is a placeholder. Default to 16:9 rounding.
    """
    # 16:9 default:
    return int(round(H_pred * 16 / 9))

# =========================
# Orchestration (chain hop)
# =========================
def run_chain_for_distortion(spatial, head, res2folder: Dict[Tuple[int,int], str],
                             out_fh, scene: str, distortion: str, idx_to_label):
    step = 0
    cur_res = Config.START_RES

    if cur_res not in res2folder:
        print(f"  [WARN] Missing start folder {distortion}_from{cur_res[0]}x{cur_res[1]}")
        return

    while True:
        folder = res2folder.get(cur_res)
        if folder is None:
            print(f"  [STOP] No folder for predicted res {cur_res} under {distortion}")
            break

        frames = list_frames(folder)
        start = step * Config.CLIP_LEN
        end = start + Config.CLIP_LEN
        if end > len(frames):
            print(f"  [STOP] Not enough frames in {Path(folder).name}: need {Config.CLIP_LEN} from {start}, have {len(frames)}")
            break

        clip_paths = frames[start:end]
        rgb_clip = load_clip_tensor(clip_paths)  # (T,C,H,W)

        pred_h, pred_w = predict_res_for_clip(spatial, head, rgb_clip, idx_to_label)

        rec = {
            "scene": scene,
            "distortion": distortion,
            "step": step,
            "source_folder": Path(folder).name,
            "frame_start": start,
            "frame_end_inclusive": end - 1,
            "pred_h": int(pred_h),
            "pred_w": int(pred_w),
        }
        out_fh.write(json.dumps(rec) + "\n")

        next_res = (int(pred_h), int(pred_w))
        if next_res == cur_res:
            # progress only if we still have more frames to consume at this res
            if end >= len(frames):
                print("  [STOP] Predicted same resolution and no new frames to consume.")
                break
        cur_res = next_res
        step += 1

def main():
    os.makedirs(Config.OUT_DIR if hasattr(Config, "OUT_DIR") else Config.OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(Config.PTH_DIR, Config.CKPT_NAME)
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # Build modules and load weights
    spatial, head, idx_to_label = load_checkpoint(ckpt_path)
    spatial = spatial.to(Config.DEVICE).eval()
    head = head.to(Config.DEVICE).eval()

    # Scan test tree
    scenes = scan_test_tree(Config.TEST_DIR)

    for scene, dmap in scenes.items():
        out_path = os.path.join(Config.OUTPUT_DIR, f"{scene}_preds.ndjson")
        print(f"[SCENE] {scene} -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as out_fh:
            for distortion, res_map in dmap.items():
                print(f"  [DIST] {distortion} (res: {sorted(res_map.keys())})")
                run_chain_for_distortion(spatial, head, res_map, out_fh, scene, distortion, idx_to_label)

    print("Done.")

if __name__ == "__main__":
    main()
