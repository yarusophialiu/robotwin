# ==== LABEL MAP (match your dataset) ====
LABELS = [360, 480, 720, 864, 1080]
IDX_TO_LABEL = {i: v for i, v in enumerate(LABELS)}
RES_TABLE = {
    360:  (360, 640),
    480:  (480, 854),
    720:  (720, 1280),
    864:  (864, 1536),
    1080: (1080, 1920),
}

# =========================
# Config (add MV cache + normalization like dataset)
# =========================
class Config:
    PTH_DIR = "ablation_exp_temporal_test_lr1e-05_20251108_074902"
    CKPT_NAME = "best.pth"
    TEST_DIR = "PATH/TO/Config.TEST_DIR"          # <-- set
    MV_CACHE_DIR = "PATH/TO/Config.MV_CACHE_DIR"  # <-- set (e.g., ./new-dataset-ue5)
    OUTPUT_DIR = "predictions_out"
    START_RES = (1080, 1920)
    CLIP_LEN = 31
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_TO_TENSOR = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

# =========================
# Helpers mirrored from dataset_video_resolution.py
# =========================
import OpenEXR, Imath
import numpy as np

_META_CACHE = {}

def _infer_frame_index(path: str) -> int:
    import re, os
    m = re.search(r'(\d+)(?=[^\d]*\.[a-zA-Z]+$)', os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(m.group(1))

def _get_mv_meta(mv_dir: Path):
    if mv_dir in _META_CACHE:
        return _META_CACHE[mv_dir]
    files = list(mv_dir.glob("*_*_mv_lr.exr"))
    if not files:
        raise FileNotFoundError(f"No mv_lr EXR files found in {mv_dir}")
    frame_indices, views = [], []
    for p in files:
        parts = p.stem.split("_")  # e.g., "1_0_mv_lr"
        if len(parts) < 3: 
            continue
        try:
            f_idx = int(parts[0]); v = int(parts[1])
        except ValueError:
            continue
        frame_indices.append(f_idx); views.append(v)
    if not frame_indices:
        raise ValueError(f"Could not parse any mv_lr filenames in {mv_dir}")
    view = max(set(views), key=views.count)
    offset = min(frame_indices)
    _META_CACHE[mv_dir] = (view, offset)
    return view, offset

def read_mv_exr(path: str, crop: int = 518) -> np.ndarray:
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    R = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape(H, W)
    G = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape(H, W)
    exr.close()
    y0 = (H - crop) // 2; x0 = (W - crop) // 2
    R = R[y0:y0+crop, x0:x0+crop]
    G = G[y0:y0+crop, x0:x0+crop]
    mv = np.stack([R, G], axis=0).astype(np.float32)
    mv = np.nan_to_num(mv, nan=0.0, posinf=0.0, neginf=0.0)
    return mv  # [2, crop, crop]

def _cached_mv_path(cache_dir: str, rel_frame_path: str) -> str:
    # rel_frame_path: "Scene/dist_fromHxW/frame_00000.png"
    frame_path = Path(rel_frame_path)
    scene = frame_path.parts[0]
    dist_type_full = frame_path.parts[1]  # e.g., normal_from720x1280
    import re
    dist_type = re.sub(r"_from\d+x\d+$", "", dist_type_full)
    idx = _infer_frame_index(frame_path.name)
    mv_dir = Path(cache_dir) / scene / dist_type / "mv_lr"
    view, offset = _get_mv_meta(mv_dir)
    mv_filename = f"{idx + offset}_{view}_mv_lr.exr"  # mv starts at 1 if offset=1
    mv_path = mv_dir / mv_filename
    if not mv_path.exists():
        raise FileNotFoundError(f"mv_lr file not found: {mv_path}")
    return str(mv_path)

# =========================
# Build motion from cached EXR (REPLACES previous build_motion)
# =========================
def _abs_to_rel_under_test_dir(abs_path: str) -> str:
    """Turn absolute frame path into relative path under TEST_DIR."""
    abs_path = Path(abs_path).resolve()
    root = Path(Config.TEST_DIR).resolve()
    return str(abs_path.relative_to(root))

@torch.no_grad()
def build_motion(rgb_clip_paths_abs: List[str]) -> torch.Tensor:
    """
    Using the same logic as your dataset:
    - For each absolute frame path, convert to relative under TEST_DIR
    - Resolve mv_lr EXR in MV_CACHE_DIR
    - read_mv_exr -> [2,h,w], stack to [T,2,h,w], then to [1,2,T,h,w]
    """
    mv_frames = []
    for abs_fp in rgb_clip_paths_abs:
        rel_fp = _abs_to_rel_under_test_dir(abs_fp)
        exr_path = _cached_mv_path(Config.MV_CACHE_DIR, rel_fp)
        mv = read_mv_exr(exr_path)     # [2,h,w]
        mv_frames.append(torch.from_numpy(mv))  # float32
    motion = torch.stack(mv_frames, dim=0)     # [T,2,h,w]
    motion = motion.permute(1, 0, 2, 3).unsqueeze(0)  # [1,2,T,h,w]
    return motion

# =========================
# Prediction (wire in new motion)
# =========================
@torch.no_grad()
def predict_res_for_clip(spatial: nn.Module,
                         head: nn.Module,
                         rgb_clip: torch.Tensor,
                         clip_paths: List[str]) -> Tuple[int, int]:
    """
    rgb_clip:  (T,C,H,W) normalized like dataset
    clip_paths: absolute paths for the same T frames
    """
    device = Config.DEVICE
    spatial.eval(); head.eval()

    T_len, C, H, W = rgb_clip.shape
    frames = rgb_clip.to(device).unsqueeze(0)          # (1,T,C,H,W)
    flat   = frames.view(T_len, C, H, W)               # (T,C,H,W)

    # spatial per-frame -> (1,T,D)
    s_feats = spatial(flat).view(1, T_len, -1)

    # motion from EXR cache -> (1,2,T,h,w)
    motion = build_motion(clip_paths).to(device)

    logits, ord_logits, _ = head(motion, s_feats, return_att=True)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    H_val = IDX_TO_LABEL[pred_idx]
    H_out, W_out = RES_TABLE[H_val]
    return H_out, W_out

# =========================
# In the chain loop, pass clip_paths to predict_res_for_clip
# =========================
def run_chain_for_distortion(spatial, head, res2folder, out_fh, scene, distortion):
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
        frames = sorted(Path(folder).glob("frame_*.png"))
        frames = [str(p) for p in frames]
        start = step * Config.CLIP_LEN
        end = start + Config.CLIP_LEN
        if end > len(frames):
            print(f"  [STOP] Not enough frames in {Path(folder).name}: need {Config.CLIP_LEN} from {start}, have {len(frames)}")
            break

        clip_paths = frames[start:end]
        # load RGB like dataset (ToTensor + Normalize)
        imgs = []
        for fp in clip_paths:
            im = Image.open(fp).convert("RGB")
            imgs.append(Config.IMG_TO_TENSOR(im))
        rgb_clip = torch.stack(imgs, dim=0)  # (T,C,H,W)

        pred_h, pred_w = predict_res_for_clip(spatial, head, rgb_clip, clip_paths)

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
        if next_res == cur_res and end >= len(frames):
            print("  [STOP] Predicted same resolution and no new frames to consume.")
            break
        cur_res = next_res
        step += 1
