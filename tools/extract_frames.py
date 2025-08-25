from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import cv2

REC_DIR = "data/recordings"
OUT_DIR = "datasets/mt2/images/train"
STEP = 15  # co ile klatek zapisać (przy 15 FPS → 1 kl/s)


def extract_frames(rec_dir: str = REC_DIR, out_dir: str = OUT_DIR, step: int = STEP) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(rec_dir, "*.mp4")))
    if not videos:
        logging.warning("Nie znaleziono nagrań…")
        return
    logging.info(f"Znaleziono {len(videos)} nagrań…")
    for vid in videos:
        logging.info(f"Przetwarzam: {vid}")
        cap = cv2.VideoCapture(vid)
        i = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                out = os.path.join(out_dir, f"{Path(vid).stem}_{i:06d}.jpg")
                cv2.imwrite(out, frame)
                saved += 1
            i += 1
        cap.release()
        logging.info(f"Zapisano {saved} klatek")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from recorded videos")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="włącz poziom logowania INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    extract_frames()
    logging.info("Gotowe.")


if __name__ == "__main__":
    main()
