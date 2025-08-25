from __future__ import annotations
import os, glob
import cv2
from pathlib import Path
import argparse


def extract_frames(rec_dir: str, out_dir: str, step: int, fmt: str,
                   quality: int | None = None) -> None:
    """Extract frames from all MP4 videos in *rec_dir*.

    Frames are saved to *out_dir* every *step* frames using *fmt* extension
    and optional *quality* parameter forwarded to ``cv2.imwrite``.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(rec_dir, '*.mp4')))
    print(f"Znaleziono {len(videos)} nagrań…")

    for vid in videos:
        print(f"Przetwarzam: {vid}")
        cap = cv2.VideoCapture(vid)
        i = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                out = os.path.join(out_dir, f"{Path(vid).stem}_{i:06d}.{fmt}")
                params: list[int] = []
                if quality is not None:
                    fmt_lower = fmt.lower()
                    if fmt_lower in ("jpg", "jpeg"):
                        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    elif fmt_lower == "png":
                        params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
                cv2.imwrite(out, frame, params)
                saved += 1
            i += 1
        cap.release()
        print(f" zapisano {saved} klatek")
    print("Gotowe.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", default="data/recordings")
    ap.add_argument("--out-dir", default="datasets/mt2/images/train")
    ap.add_argument("--step", type=int, default=15,
                    help="co ile klatek zapisać (przy 15 FPS → 1 kl/s)")
    ap.add_argument("--format", default="jpg",
                    help="format zapisu klatek (np. jpg, png)")
    ap.add_argument("--quality", type=int,
                    help="parametr jakości przekazywany do cv2.imwrite")
    args = ap.parse_args()
    extract_frames(args.rec_dir, args.out_dir, args.step,
                   args.format, args.quality)


if __name__ == "__main__":
    main()

