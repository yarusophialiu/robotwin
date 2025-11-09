# dataloader/dataset_video_resolution.py
import os, re, torch, OpenEXR, Imath
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


# ---- Label mapping (edit if your classes change) ----
LABELS = [360, 480, 720, 864, 1080]
LABEL_TO_IDX = {v: i for i, v in enumerate(LABELS)}
IDX_TO_LABEL = {i: v for i, v in enumerate(LABELS)}

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
_META_CACHE = {} 

def build_frame_paths(root_dir: str, pattern: str, start: int, T: int, stride: int):
    """
    pattern: 'RaceGame-1/normal_from1080x1920/frame_{:05d}.png'
    returns absolute paths for frames start ... start+(T-1)*stride
    """
    idxs = [start + i * stride for i in range(T)]
    rels = [pattern.format(j) for j in idxs]
    return [os.path.join(root_dir, r) for r in rels]


def get_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def _infer_frame_index(path: str) -> int:
    """
    Finds the last number before the extension.
    e.g. frame_00042.png -> 00042 -> 42
    """
    m = re.search(r'(\d+)(?=[^\d]*\.[a-zA-Z]+$)', os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(m.group(1))


def _get_mv_meta(mv_dir: Path):
    """
    Detect (view, offset) for an mv_lr folder.
    offset is typically 0 if files start at 0_*, or 1 if files start at 1_*.
    """
    if mv_dir in _META_CACHE:
        return _META_CACHE[mv_dir]

    files = list(mv_dir.glob("*_*_mv_lr.exr"))
    if not files:
        raise FileNotFoundError(f"No mv_lr EXR files found in {mv_dir}")

    # Parse indices from filenames like "<frameIdx>_<view>_mv_lr.exr"
    frame_indices = []
    views = []
    for p in files:
        stem = p.stem  # e.g., "1_0_mv_lr"
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            f_idx = int(parts[0])
            v = int(parts[1])
        except ValueError:
            continue
        frame_indices.append(f_idx)
        views.append(v)

    if not frame_indices:
        raise ValueError(f"Could not parse any mv_lr filenames in {mv_dir}")

    # View is constant per folder: take the most common (or first)
    view = max(set(views), key=views.count)

    # Determine offset (0-based or 1-based). Use the minimum seen.
    min_idx = min(frame_indices)
    offset = min_idx  # works for 0- or 1-based, and any other start

    _META_CACHE[mv_dir] = (view, offset)
    return view, offset


def read_mv_exr(path: str, crop: int = 518) -> np.ndarray:
    """
    Read EXR motion vectors (R,G channels), center crop, return [2, H, W] float32.
    path e.g. ./new-dataset-ue5/AnimX_Cats_R_V5-1/normal/mv_lr/1_0_mv_lr.exr
    EXR file has 4 chanels A,B,G,R, but A,B are 0, only G,R are non-zeros
    """
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read only R, G channels
    R = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape(H, W)
    G = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape(H, W)
    exr.close()

    # Center crop to crop x crop (518x518)
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    R = R[y0:y0+crop, x0:x0+crop]
    G = G[y0:y0+crop, x0:x0+crop]

    # Stack to [2, H, W], convert to float32
    mv = np.stack([R, G], axis=0).astype(np.float32)
    mv = np.nan_to_num(mv, nan=0.0, posinf=0.0, neginf=0.0)
    return mv


def _cached_mv_path(cache_dir: str, frame_path: str, view: int = 0) -> str:
    """
    Given:
        cache_dir: base dataset folder (e.g. "./new-dataset-ue5")
        frame_path: relative path like "AnimX_Cats_R_V5-1/normal/frame_00000.png"
        idx: frame index (if you already have it, otherwise can infer)
        view: usually 0, for multi-view setups
    Returns:
        Path to the corresponding mv_lr EXR file
    """
    # Resolve base parts
    frame_path = Path(frame_path)
    scene = frame_path.parts[0]          # e.g. AnimX_Cats_R_V5-1
    dist_type = frame_path.parts[1]      # e.g. normal_from720x1280
    dist_type = re.sub(r"_from\d+x\d+$", "", dist_type) # e.g. normal


    idx = _infer_frame_index(frame_path.name)
    mv_dir = Path(cache_dir) / scene / dist_type / "mv_lr"

    # EXR file name pattern: {frame_index+1}_{view}_mv_lr.exr or {frame_index}_{view}_mv_lr.exr
    # adjust the + offset below depending on your naming convention, mv exr start from 1, frame starts from 0
    view, offset = _get_mv_meta(mv_dir)
    mv_filename = f"{idx + offset}_{view}_mv_lr.exr"
    mv_path = mv_dir / mv_filename

    if not mv_path.exists():
        raise FileNotFoundError(
            f"mv_lr file not found: {mv_path} "
            f"(scene={scene}, dist={dist_type}, idx={idx}, view={view}, offset={offset})"
        )

    return str(mv_path)



def parse_scene_dist(pattern: str) -> str:
    """
    'BurnedTrees-2/setting_4_from720x1280/frame_{:05d}.png'
      -> 'BurnedTrees-2/setting_4'
    'CaveEnvironment-2/normal_with_TSR_from360x640/frame_{:05d}.png'
      -> 'CaveEnvironment-2/normal_with_TSR'
    """
    base = pattern.split('/frame_')[0]      # 'BurnedTrees-2/setting_4_from720x1280'
    return base.split('_from')[0]           # 'BurnedTrees-2/setting_4'


def parse_res_from_pattern(pattern: str) -> int:
    m = re.search(r"from(\d+)x", pattern)
    if m is None:
        raise ValueError(f"Cannot parse resolution from pattern: {pattern}")
    return int(m.group(1))


class VideoResolutionDataset(Dataset):
    """
    CSV columns: scene_name, start, T, stride, position, clip_path_pattern, clip_jod, label
    Returns:
      rgb:    [T, C, H, W] float32 (normalized)
      motion: [C_m, T, H, W] (default frame-diff, 1 channel)
      label:  int in [0..4]
      meta:   dict
    """
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        mv_cache_dir: str,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        required = {"scene_name", "start", "T", "stride", "clip_path_pattern", "clip_jod", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        self.root_dir = root_dir
        self.mv_cache_dir = mv_cache_dir

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)        

        # (scene_name, start, clip_res) -> clip_jod
        self.jod_map = {}
        for _, row in self.df.iterrows():
            if pd.isna(row["clip_jod"]):
                continue

            pattern = str(row["clip_path_pattern"])
            scene_dist = parse_scene_dist(pattern)      # e.g. 'BurnedTrees-2/setting_4'
            start = int(row["start"])
            clip_res = parse_res_from_pattern(pattern)  # e.g. 360 / 480 / ...

            self.jod_map[(scene_dist, start, clip_res)] = float(row["clip_jod"])

    def __len__(self):
        return len(self.df)

    def _load_rgb_clip(self, row):
        paths = build_frame_paths(
            self.root_dir,
            row["clip_path_pattern"],
            int(row["start"]),
            int(row["T"]),
            int(row["stride"]),
        )
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = self.to_tensor(img)     # [C,H,W] in [0,1]
            img = self.normalize(img)
            frames.append(img)
        return torch.stack(frames, dim=0) # [T,C,H,W]

    
    def _load_motion_from_cache(self, frame_paths_abs):
        """Load cached motion tensors .exr per frame and stack to [T,C,h,w]."""
        frames = []
        for p in frame_paths_abs:
            # print(f'p {p}') AnimX_Cats_R_V5-1/normal/frame_00000.png
            exr_path = _cached_mv_path(self.mv_cache_dir, p) # exr new-dataset-ue5/AnimX_Cats_R_V5-1/normal/mv_lr/1_0_mv_lr.exr
            # mv = torch.load(pt)  # [2,H,W] or [1,H,W], float32
            mv = read_mv_exr(exr_path) # 2, 518, 518
            mv = torch.from_numpy(mv) 
            frames.append(mv)
        motion = torch.stack(frames, dim=0)  # [T,C,H,W]
        return motion


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb_clip = self._load_rgb_clip(row)                 # [T,C,H,W]
        T = rgb_clip.shape[0]

        #################### motion #######################
        pattern = str(row['clip_path_pattern'])
        start   = int(row['start'])
        T       = int(row['T'])
        stride  = int(row['stride'])
        rel_paths = [pattern.format(start + k * stride) for k in range(T)]
        motion = self._load_motion_from_cache(rel_paths)  # TODO: abs path [T,C,h,w] motion for the whole clip
        ###################################################

        label_val = int(row["label"])
        if label_val not in LABEL_TO_IDX:
            raise ValueError(f"Label {label_val} not in {LABELS}")
        label_idx = LABEL_TO_IDX[label_val]
        scene_dist = parse_scene_dist(pattern)

        meta = {
            "scene_name": row["scene_name"],
            "scene_dist": scene_dist,
            "start": int(row["start"]),
            "T": int(row["T"]),
            # "stride": int(row["stride"]),
            "clip_path_pattern": row["clip_path_pattern"],
            # "clip_jod": float(row["clip_jod"]),
            "label_value": label_val, # target resolution value
            # optional: actual clip resolution used for JOD map debug
            # "clip_res": _parse_res_from_pattern(pattern),
        }

        return {
            "rgb": rgb_clip,
            "motion": motion,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "meta": meta,
        }
