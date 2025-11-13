# pip install psychopy ffpyplayer
from psychopy import visual, core, prefs
from psychopy.hardware import keyboard
from PIL import Image
from datetime import datetime
from utils.utils import *
import csv, os, time, random
import numpy as np
import imageio, pyglet

# ------------- CONFIG -------------
# parent_folder = r"C:\Users\y50046154\Projects\res-classifier\data\sample_videos"
prefs.general['movieLib'] = ['opencv', 'ffpyplayer', 'moviepy']
RESPONSE_KEYS = ["left", "right", "escape", "backspace"]  # add replay keys

# Paths
# root = r'C:\Users\a84403325\Desktop\copy'
root = r'D:\res-classifier\test_videos\reencode'
reference_resolution = 720 # TODO: 1080 and 720
# ref_folder = f"{root}/videos_out_reference{reference_resolution}p"
ref_folder = f"{root}/videos_out_reference{reference_resolution}p_smaller_pc"
test_folder = f"{root}/videos_out_smaller_pc"  # change if needed

DEBUG = True
FULLSCREEN = True # False True
# SIZE = (1280, 720)
BG_COLOR = "black"
TEXT_COLOR = "white"
ISI = 0.25            # gap between videos (s)
RANDOMIZE_ORDER = True  # AB vs BA randomization per trial
RESPONSE_KEYS = ["left", "right", "escape"]  # ←=first video, →=second video

results_folder = f"results{reference_resolution}"
os.makedirs(results_folder, exist_ok=True)

# Define CSV file path
csv_path = f"{results_folder}/pairwise_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
OUTPUT_CSV = os.path.join(os.getcwd(), csv_path)

start_time = time.time()

# Encode choice: 1 = test, 0 = reference
# ----------------------------------

def message(win, text, wait_key=True, kb=None, height=24):
    msg = visual.TextStim(win, text=text, color=TEXT_COLOR, height=height, wrapWidth=1400)
    msg.draw()
    win.flip()
    if wait_key:
        kb.clearEvents()
        kb.waitKeys()


def main():
    # Get the number of available screens
    display = pyglet.canvas.get_display()
    screens = display.get_screens()
    num_screens = len(screens)
    print(f"Detected {num_screens} screen(s).")

    # # Use the last one (usually the external monitor)
    # # SCREEN_INDEX = num_screens - 1
    # SCREEN_INDEX = 0
    # if FULLSCREEN:
    #     # If fullscreen is True, the 'size' parameter is often ignored or set to the screen resolution by default
    #     win = visual.Window(fullscr=FULLSCREEN, color=BG_COLOR, units="pix", waitBlanking=False, screen=SCREEN_INDEX,)
    # else:
    #     win = visual.Window(fullscr=FULLSCREEN, size=SIZE, color=BG_COLOR, units="pix", waitBlanking=False, screen=SCREEN_INDEX,)


    SCREEN_INDEX = 0
    scr = screens[SCREEN_INDEX]

    # Fake fullscreen: windowed but same size as the screen
    SIZE = (scr.width, scr.height)   # e.g. 1920x1080
    # FULLSCREEN = False               # IMPORTANT: no exclusive fullscreen!

    win = visual.Window(
        fullscr=False,               # windowed, but fills the screen
        size=SIZE,
        color=BG_COLOR,
        units="pix",
        waitBlanking=False,
        screen=SCREEN_INDEX,
    )


        
    kb = keyboard.Keyboard()
    win.mouseVisible = False

    # noise_screen = visual.Rect(win=win, width=win.size[0], height=win.size[1], fillColor=BG_COLOR, autoLog=False)

    # Preload all movies (faster & avoids per-trial open)
    # We’ll keep a cache keyed by path
    TRIALS = build_video_pair(ref_folder, test_folder)
    N_TRIALS = len(TRIALS)
    print(f'Total trials {N_TRIALS}')

    message(
        win,
        f"Pairwise Video Test\n\n"
        f"Total trials {N_TRIALS}\n"
        "You will see two videos one after the other.\n"
        "Choose which one looks better.\n\n"
        "Press any key to begin.",
        kb=kb
    )

    movie_cache = {}
    for t in TRIALS:
        for label in ("reference", "test"):
            path = t[label]
            if path not in movie_cache:
                if not os.path.exists(path):
                    win.close(); core.quit()

    # CSV setup
    fieldnames = [
        "trial_index",
        "video_first_label", "video_first_path",
        "video_second_label", "video_second_path",
        "choice", 
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Run trials
        for i, t in enumerate(TRIALS, start=1):
            print(f'\n========== Trial {i} ==========')
            # Determine order (AB or BA)
            order = [("A", "reference"), ("B", "test")]
            if RANDOMIZE_ORDER and random.random() < 0.5:
                order = [("A", "test"), ("B", "reference")]

            # Fixation / ready
            # message(win, f"Trial {i}\n\nPress any key to see the first video.", True, kb=kb)

            # First video
            first_label, first_key = order[0] # [A, test] or [A, reference]
            first_path = t[first_key]
            print(f'first_key {first_key}, path1 {first_path}')

            # Second video
            second_label, second_key = order[1]
            second_path = t[second_key]
            # print(f'second_key {second_key}, path2 {second_path}\n')

            # Overlay text selection
            first_overlay = first_key if DEBUG else "Video 1"
            second_overlay = "test" if order[1][1] == "test" else "reference"
            second_overlay = second_overlay if DEBUG else "Video 2"

            first_overlay  = overlay_label(first_key, 1, DEBUG) # first_key is test or Video 1
            second_overlay = overlay_label(second_key, 2, DEBUG)

            # play_movie_imageio(win, first_path, kb=kb, overlay_text=first_overlay)
            print(f'first_key {first_key}, path1 {first_path}')
            # first_path, win=win, fullscreen=True

            play_movie_ffplay(first_path, fullscreen=FULLSCREEN)
            core.wait(ISI)
            show_noise(win, duration=0.5, dynamic=True)
            # play_movie_imageio(win, second_path, kb=kb, overlay_text=second_overlay)
            print(f'second_key {second_key}, path2 {second_path}')
            play_movie_ffplay(second_path, fullscreen=FULLSCREEN)


            # play_movie_imageio(win, first_path, kb=kb, overlay_text=first_overlay)
            # core.wait(ISI)
            # show_noise(win, duration=0.5, dynamic=True)
            # play_movie_imageio(win, second_path, kb=kb, overlay_text=second_overlay)


            # Prompt for response
            prompt = visual.TextStim(
                win,
                text=(f"Trial {i}/{N_TRIALS}\n"
                      "Which video is better?\n\n"
                    "← First    → Second     (Esc to quit)\n"
                    "Backspace = Replay"), 
                color=TEXT_COLOR, height=28
            )
            prompt.draw(); 
            win.flip()
            kb.clearEvents()
            t0 = core.getTime()
            choice = None

            while True:
                keys = kb.getKeys(waitRelease=False)
                if keys:
                    k = keys[-1].name
                    print(f"Key pressed: {k}")  # <-- DEBUG print to see what PsychoPy reads

                    if k in ("escape", "esc"):
                        win.mouseVisible = True
                        win.close()
                        print(f"Results saved to: {OUTPUT_CSV}")
                        core.quit()

                    elif k in ("backspace", "delete", "backspace (8)"):  # handle OS variations
                        print("Replaying both videos...")
                        replay_pair(win, first_path, second_path, kb, isi=ISI, noise_dur=0.5, FULLSCREEN=FULLSCREEN)
                        # Re-show the prompt after replay
                        prompt.draw()
                        win.flip()
                        kb.clearEvents()
                        continue

                    elif k in ("left", "right"):
                        if k == "left":
                            chosen_label = first_key
                        else:
                            chosen_label = second_key
                        choice = 1 if chosen_label == "test" else 0
                        break


            writer.writerow({
                "trial_index": i,
                "video_first_label": first_key,     # which stimulus (reference/test) was first
                "video_first_path": first_path,
                "video_second_label": second_key,   # which was second
                "video_second_path": second_path,
                "choice": choice,                    # 'first' or 'second'
            })
            f.flush()

            # brief inter-trial screen
            message(win, " ", True, kb)

    # Done
    message(win, "All done, thank you!\nPress any key to exit.", True, kb)
    for m in movie_cache.values():
        try:
            m.unload()
        except Exception:
            pass
    win.mouseVisible = True
    win.close()
    print(f"Results saved to: {OUTPUT_CSV}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print elapsed time in a nice format
    print(f"Program finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")

if __name__ == "__main__":
    main()
