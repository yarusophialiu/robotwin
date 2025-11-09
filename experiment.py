import os
import re
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
from PIL import Image
import torchvision.transforms as T

# -----------------------------
# Config
# -----------------------------
class Config:
    PTH_DIR = "ablation_exp_temporal_test_lr1e-05_20251108_074902"
    CKPT_NAME = "best.pth"  # change if needed
    TEST_DIR = "/path/to/your/test/dir"  # <- set this
    OUTPUT_DIR = "predictions_out"       # where to save per-scene predictions
    START_RES = (1080, 1920)             # starting at *_from1080x1920
    CLIP_LEN = 31                        # frames per step (0..30, 31..61, ...)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Utilities
# -----------------------------
RES_RX = re.compile(r"_from(\d+)x(\d+)$")

def find_scene_distortion_folders(root: str) -> Dict[str, Dict[str, Dict[Tuple[int,int], str]]]:
    """
    Walk Config.TEST_DIR and collect:
      scenes[scene][distortion][(H,W)] -> path_to_folder
    where leaf folders are like: <TEST_DIR>/<Scene>/<Distortion>_fromHxW
    """
    scene_map: Dict[str, Dict[str, Dict[Tuple[int,int], str]]] = {}
    for scene_dir in sorted(Path(root).iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        scene_map.setdefault(scene, {})
        for dist_dir in sorted(scene_dir.iterdir()):
            if not dist_dir.is_dir():
                continue
            m = RES_RX.search(dist_dir.name)
            if not m:
                # Handle pattern like BurnedTrees-2/normal_from360x640 etc.
                # If the distortion folder has more nesting, add handling here.
                continue
            H, W = int(m.group(1)), int(m.group(2))

            # derive pure distortion name (strip the trailing _fromHxW)
            # e.g., "normal_from360x640" -> "normal"
            base = dist_dir.name
            distortion = base[:base.rfind("_from")]
            scene_map[scene].setdefault(distortion, {})
            scene_map[scene][distortion][(H, W)] = str(dist_dir)
    return scene_map

def list_frames(folder: str) -> List[str]:
    frames = sorted([p for p in Path(folder).glob("frame_*.png")])
    return [str(p) for p in frames]

def load_clip(frames: List[str]) -> torch.Tensor:
    """
    Load a list of frame paths into a (T, C, H, W) tensor.
    TODO: Adjust transforms to what your model expects.
    """
    # Example: convert to tensor in [0,1], no resizing here
    tfm = T.Compose([T.ToTensor()])
    imgs = []
    for f in frames:
        img = Image.open(f).convert("RGB")
        imgs.append(tfm(img))
    clip = torch.stack(imgs, dim=0)  # (T, C, H, W)
    return clip

# -----------------------------
# Model bits (fill these in)
# -----------------------------
def load_model(ckpt_path: str) -> torch.nn.Module:
    """
    TODO: Replace with your model and state_dict loading.
    """
    # Example skeleton:
    model = torch.nn.Identity()  # <- replace with your model class
    state = torch.load(ckpt_path, map_location="cpu")
    # If you saved directly model.state_dict():
    # model.load_state_dict(state)
    # If you saved a dict: {"model": state_dict, ...}
    # model.load_state_dict(state["model"])
    model.eval()
    return model

def preprocess_for_model(clip: torch.Tensor) -> torch.Tensor:
    """
    TODO: Resize/normalize/rearrange to model's expected input.
    E.g., if your model expects (B, C, T, H, W) and fixed spatial size:
    """
    # Example: move to (B, C, T, H, W)
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    return clip

def decode_prediction_to_resolution(model_out: torch.Tensor) -> Tuple[int, int]:
    """
    TODO: Map model output to (H, W). For classification, argmax -> index -> (H,W).
    For regression, round/clip to nearest allowed (H,W).
    """
    # Placeholder: return 720x1280 for demo
    return (720, 1280)

# If classification, define your allowed resolution set and a mapping:
# ALLOWED_RES = [(360,640), (480,854), (720,1280), (1080,1920)]
# def decode_prediction_to_resolution(model_out):
#     idx = model_out.argmax(dim=1).item()
#     return ALLOWED_RES[idx]

# -----------------------------
# Orchestrator
# -----------------------------
def run_chain_for_distortion(model, folders_by_res: Dict[Tuple[int,int], str],
                             out_fh, scene: str, distortion: str):
    """
    For a given distortion of a scene:
      start at START_RES; take frames [0..30]; predict (H1,W1); write;
      then switch to folder *_fromH1xW1; take frames [31..61]; predict (H2,W2); write; etc.
    """
    step = 0
    current_res = Config.START_RES

    if current_res not in folders_by_res:
        print(f"  [WARN] Missing start folder for {distortion}: *_from{current_res[0]}x{current_res[1]}")
        return

    # While we can find the current resolution folder and enough frames to take a next clip window
    while True:
        current_folder = folders_by_res.get(current_res)
        if current_folder is None:
            print(f"  [STOP] No folder for predicted resolution {current_res} for {distortion}.")
            break

        frames = list_frames(current_folder)
        start_idx = step * Config.CLIP_LEN
        end_idx = start_idx + Config.CLIP_LEN  # exclusive
        if end_idx > len(frames):
            print(f"  [STOP] Not enough frames in {Path(current_folder).name}: "
                  f"need {Config.CLIP_LEN} frames from {start_idx}, have {len(frames)}.")
            break

        clip_paths = frames[start_idx:end_idx]
        clip = load_clip(clip_paths).to(Config.DEVICE)
        inp = preprocess_for_model(clip)

        with torch.no_grad():
            out = model(inp)

        pred_h, pred_w = decode_prediction_to_resolution(out)

        # Write a record
        record = {
            "scene": scene,
            "distortion": distortion,
            "step": step,
            "source_folder": Path(current_folder).name,
            "frame_start": start_idx,
            "frame_end_inclusive": end_idx - 1,
            "pred_h": pred_h,
            "pred_w": pred_w
        }
        out_fh.write(json.dumps(record) + "\n")

        # Prepare next hop
        next_res = (pred_h, pred_w)
        if next_res == current_res and (start_idx + Config.CLIP_LEN) >= len(frames):
            # No progress and no more frames to consume here — stop to avoid infinite loop
            print(f"  [STOP] Predicted same resolution and no additional frames; stopping for safety.")
            break

        current_res = next_res
        step += 1

def main():
    os.makedirs(Config.OFFSET_OUTPUT_DIR if hasattr(Config, "OFFSET_OUTPUT_DIR") else Config.OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(Config.PTH_DIR, Config.CKPT_NAME)
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = load_model(ckpt_path).to(Config.DEVICE)
    scenes = find_scene_distortion_folders(Config.TEST_DIR)

    # One newline-delimited JSON file per scene (aggregates all distortions)
    for scene, distortions in scenes.items():
        out_path = os.path.join(Config.OUTPUT_DIR, f"{scene}_preds.ndjson")
        print(f"[SCENE] {scene} -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as out_fh:
            for distortion, res_map in distortions.items():
                print(f"  [DIST] {distortion} (resolutions: {sorted(res_map.keys())})")
                run_chain_for_distortion(model, res_map, out_fh, scene, distortion)

    print("Done.")

if __name__ == "__main__":
    main()
