from __future__ import annotations
import os, glob
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

REC_DIR = 'data/recordings'
OUT_DIR = 'datasets/mt2/images/train'
STEP = 15  # co ile klatek zapisać (przy 15 FPS → 1 kl/s)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Extract frames from recordings")
parser.add_argument("--global-progress", action="store_true",
                    help="Show a single progress bar for all frames")
args = parser.parse_args()

videos = sorted(glob.glob(os.path.join(REC_DIR, '*.mp4')))
print(f"Znaleziono {len(videos)} nagrań…")

if args.global_progress:
    # Pre-compute frame counts for all videos
    frame_counts = []
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    total_frames = sum(frame_counts)
    pbar = tqdm(total=total_frames, desc="Wszystkie nagrania", unit="kl")

    for vid, fcount in zip(videos, frame_counts):
        cap = cv2.VideoCapture(vid)
        i = 0; saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % STEP == 0:
                out = os.path.join(OUT_DIR, f"{Path(vid).stem}_{i:06d}.jpg")
                cv2.imwrite(out, frame)
                saved += 1
            i += 1
            pbar.update(1)
        cap.release()
        print(f"{vid}: zapisano {saved} klatek")

    pbar.close()
else:
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        pbar = tqdm(total=frame_count, desc=vid, unit="kl")
        i = 0; saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % STEP == 0:
                out = os.path.join(OUT_DIR, f"{Path(vid).stem}_{i:06d}.jpg")
                cv2.imwrite(out, frame)
                saved += 1
            i += 1
            pbar.update(1)
        cap.release()
        pbar.close()
        print(f"{vid}: zapisano {saved} klatek")

print("Gotowe.")
