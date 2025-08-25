from __future__ import annotations
import os, glob
import cv2
from pathlib import Path

REC_DIR = 'data/recordings'
OUT_DIR = 'datasets/mt2/images/train'
STEP = 15  # co ile klatek zapisać (przy 15 FPS → 1 kl/s)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

videos = sorted(glob.glob(os.path.join(REC_DIR, '*.mp4')))
print(f"Znaleziono {len(videos)} nagrań…")

for vid in videos:
    print(f"Przetwarzam: {vid}")
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
    cap.release()
    print(f" zapisano {saved} klatek")
print("Gotowe.")
