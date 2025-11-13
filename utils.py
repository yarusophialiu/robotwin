from PIL import Image
from datetime import datetime
import csv, os, time, random
import numpy as np
import imageio, pyglet
from psychopy import visual, core, prefs


import os
import shutil
import subprocess

FFPLAY = shutil.which("ffplay")
if FFPLAY is None:
    print("[ffplay] WARNING: ffplay not found in PATH.")


import os
import shutil
import subprocess

# FFPLAY = shutil.which("ffplay")
# if FFPLAY is None:
#     print("[ffplay] WARNING: ffplay not found in PATH.")
FFPLAY = shutil.which("ffplay")
if FFPLAY is None:
    print("[ffplay] WARNING: ffplay not found in PATH.")


def play_movie_ffplay(path, fullscreen=True):
    if not os.path.exists(path):
        print(f"[ffplay] File not found: {path}")
        return
    if FFPLAY is None:
        print("[ffplay] ERROR: ffplay not found. Install ffmpeg / add to PATH.")
        return

    cmd = [FFPLAY, "-autoexit", "-loglevel", "error"]

    if fullscreen:
        SCREEN_W = 1920
        SCREEN_H = 1080
        vf_filter = (
            f"fps=120,"
            f"scale={SCREEN_W}:{SCREEN_H}:force_original_aspect_ratio=increase,"
            f"crop={SCREEN_W}:{SCREEN_H}"
        )
        cmd += ["-fs", "-noborder", "-alwaysontop", "-vf", vf_filter]

    cmd.append(path)
    print("[ffplay] Running:", " ".join(cmd))
    subprocess.run(cmd)



def build_video_pair(ref_folder, test_folder):
    """
    Return list of dictionary:
    TRIALS = [{"reference": f"parent_folder/BurnedTrees-1_gt_part2.mp4", "test": f"parent_folder/BurnedTrees-1_normal_part2.mp4"},]
    """
    # Get video filenames
    ref_videos = set(os.listdir(ref_folder))
    test_videos = set(os.listdir(test_folder))

    # Find common videos
    common_videos = sorted(ref_videos & test_videos)

    # Build TRIALS list
    TRIALS = [
        {
            "reference": f"{ref_folder}/{name}",
            "test": f"{test_folder}/{name}",
        }
        for name in common_videos if name.endswith(".mp4")
    ]
    return TRIALS


def overlay_label(stim_key, ordinal, debug):
    """
    stim_key: 'test' or 'reference'
    ordinal: 1 or 2
    """
    base = f"Video {ordinal}"
    if debug:
        human = "Test" if stim_key == "test" else "Reference"
        return f"{base} — {human}"
    return base


# use duration of timing, not frame count
def play_movie_imageio(
    win,
    path,
    TEXT_COLOR="white",
    kb=None,
    overlay_text=None,
    monitor_fps=120.0,   # set to 60/90/120 as needed
):
    rdr = imageio.get_reader(path, format="ffmpeg")
    try:
        meta = rdr.get_meta_data()
        duration = float(meta.get("duration", 0.0))  # seconds
    except Exception:
        duration = 0.0

    if duration <= 0:
        duration = 3.0  # fallback if metadata is weird

    frame_dt = 1.0 / monitor_fps
    print(f"[play_movie_imageio] {os.path.basename(path)} "
          f"duration={duration:.3f}s, monitor_fps={monitor_fps}")

    img_stim = None
    overlay_stim = None

    start_t = core.getTime()
    t_next = start_t

    try:
        for frame in rdr:
            now = core.getTime()
            # stop by time, not by exhausting all frames
            if now - start_t >= duration:
                break

            if kb and kb.getKeys(['escape'], waitRelease=False):
                break

            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = np.ascontiguousarray(frame)

            pil_img = Image.fromarray(frame).convert("RGB")

            if img_stim is None:
                w, h = int(win.size[0]), int(win.size[1])
                img_stim = visual.ImageStim(
                    win,
                    image=pil_img,
                    size=(w, h),
                    units="pix",
                    interpolate=True,
                )
                if overlay_text:
                    overlay_stim = visual.TextStim(
                        win,
                        text=overlay_text,
                        pos=(-w * 0.25, h * 0.42),
                        color=TEXT_COLOR,
                        height=28,
                        alignText='left'
                    )
            else:
                img_stim.image = pil_img

            img_stim.draw()
            if overlay_stim:
                overlay_stim.draw()
            win.flip()

            t_next += frame_dt
            dt = t_next - core.getTime()
            if dt > 0:
                core.wait(dt)
    finally:
        try:
            rdr.close()
        except Exception:
            pass
        win.flip(clearBuffer=True)


def show_noise(win, duration=0.5, dynamic=True, std=0.35):
    """
    Grey noise (mean ~0) in PsychoPy's float range [-1,1],
    repeated across RGB so it looks neutral grey.
    std controls contrast (0.25 is moderate; try 0.15 for softer, raise std to be brigher).
    """
    w, h = int(win.size[0]), int(win.size[1])

    def make_noise_frame():
        # grayscale noise in [-1,1] with mean 0
        g = np.random.normal(loc=0.0, scale=std, size=(h, w)).astype(np.float32)
        g = np.clip(g, -1.0, 1.0)
        # tile to RGB so it's grey (no color tint)
        rgb = np.repeat(g[..., None], 3, axis=2)
        return rgb

    frame = make_noise_frame()

    noise_img = visual.ImageStim(
        win,
        image=frame,          # float32 in [-1,1] (grey)
        size=(w, h),
        units="pix",
        interpolate=False,
        opacity=1.0
    )

    t_end = core.getTime() + float(duration)
    if dynamic:
        while core.getTime() < t_end:
            frame[:] = make_noise_frame()
            noise_img.image = frame
            noise_img.draw()
            win.flip()
    else:
        noise_img.draw()
        while core.getTime() < t_end:
            win.flip()



def replay_pair(win, first_path, second_path, kb, isi=0.25, noise_dur=0.5,
                FULLSCREEN=False):
    print(f"[Replay] First:  {first_path}")
    play_movie_ffplay(first_path, fullscreen=FULLSCREEN)
    core.wait(isi)
    show_noise(win, duration=noise_dur, dynamic=True)
    print(f"[Replay] Second: {second_path}")
    play_movie_ffplay(second_path, fullscreen=FULLSCREEN)

