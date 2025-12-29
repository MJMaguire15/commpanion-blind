# stream_from_mp4_benchmark.py
import sys
import time
import argparse
import subprocess
from pathlib import Path
from collections import deque

import cv2
import torch
import numpy as np

from PyQt6.QtWidgets import QApplication

from subtitle_overlay import (
    BlackBaseLayer,
    LowerDimLayer,
    SubtitleOverlay,
    DIM_ENABLED
)

from inference_cpu_stream import load_model, run

# ===================== DEFAULTS =====================
DEFAULT_VIDEO = "input_test_video.mp4"
DEFAULT_WINDOW_S = 2.8
DEFAULT_STEP_S = 0.7
ROI_SIZE = 88
CKPT = r".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"
OUT_DIR = Path("benchmark_inputs")
# ====================================================


# ---------- YouTube helper ----------
def download_youtube_video(url: str, out_mp4: Path, max_height=720):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if out_mp4.exists():
        print(f"[INFO] Using cached video: {out_mp4}")
        return out_mp4

    print(f"[INFO] Downloading YouTube video → {out_mp4}")

    cmd = [
        "yt-dlp",
        "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
        "--merge-output-format", "mp4",
        "-o", str(out_mp4),
        url,
    ]
    subprocess.run(cmd, check=True)
    return out_mp4


# ---------- ACCURACY-FIRST ROI ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_mouth_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
    )

    if len(faces) == 0:
        h, w = gray.shape
        s = ROI_SIZE // 2
        cx, cy = w // 2, h // 2
        crop = gray[cy - s:cy + s, cx - s:cx + s]
        return cv2.resize(crop, (ROI_SIZE, ROI_SIZE))

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_gray = gray[y:y + h, x:x + w]
    mouth_region = face_gray[int(0.55 * h):, :]
    return cv2.resize(mouth_region, (ROI_SIZE, ROI_SIZE))


# ---------- Text post-processing (shared) ----------
def deduplicate_text(prev: str, curr: str,
                     min_prefix_chars=24,
                     min_chars=20):
    """
    Prefix-based deduplication for sliding-window VSR.
    Removes repeated leading content even when wording drifts.
    """
    if not prev or not curr:
        return curr
    if len(curr) < min_chars:
        return curr

    prev = prev.strip()
    curr = curr.strip()

    # Compare only prefixes (robust to front-drop + hallucinations)
    p = prev[:min_prefix_chars]
    c = curr[:min_prefix_chars]

    # Find longest shared prefix
    i = 0
    for a, b in zip(p, c):
        if a == b:
            i += 1
        else:
            break

    # If meaningful overlap, drop it
    if i >= min_prefix_chars // 2:
        return curr[i:].lstrip()

    return curr



# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=DEFAULT_VIDEO)
    ap.add_argument("--youtube", type=str, default="")
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_S)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    video_path = Path(args.video)

    if args.youtube:
        safe_id = args.youtube.split("v=")[-1][:11]
        video_path = OUT_DIR / f"youtube_{safe_id}.mp4"
        video_path = download_youtube_video(args.youtube, video_path, args.max_height)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    window_frames = int(args.window_seconds * fps)
    step_frames = int(args.step_seconds * fps)

    print(f"[INFO] FPS={fps:.2f}, window={window_frames}f, step={step_frames}f")
    print("[INFO] Loading Auto-AVSR model (CPU)...")
    engine = load_model(CKPT)
    print("[INFO] Model ready")

    # ----------------- Start subtitle overlay -----------------
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    screens = app.screens()
    if len(screens) < 2:
        raise RuntimeError("A3 must be in EXTENDED display mode")

    target_screen = screens[-1]
    geom = target_screen.geometry()

    base = BlackBaseLayer(geom)
    dim = None
    if DIM_ENABLED:
        dim = LowerDimLayer(geom)

    subtitles = SubtitleOverlay(geom)
    # -----------------------------------------------------------

    buffer = deque(maxlen=window_frames)
    frames_since_decode = 0
    frame_idx = 0
    last_text = ""
    last_clean_text = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        roi = extract_mouth_roi(frame)
        buffer.append(roi)
        frames_since_decode += 1

        if len(buffer) == window_frames and frames_since_decode >= step_frames:
            frames_since_decode = 0

            clip = np.stack(buffer, axis=0)      # (T, H, W)
            clip = clip[:, None, :, :]           # (T, 1, H, W)
            clip = torch.from_numpy(clip).float() / 255.0

            t0 = time.time()
            text = run(engine, clip)
            latency = (time.time() - t0) * 1000
            ts = frame_idx / fps

            if text and text != last_text:
                clean_text = deduplicate_text(last_clean_text, text)

                if clean_text:
                    print(f"[{ts:7.2f}s | {latency:6.1f} ms] {clean_text}")
                    if last_clean_text:
                        if clean_text in last_clean_text:
                            continue
                    subtitles.set_text(clean_text)
                    app.processEvents()
                    last_clean_text = clean_text

                last_text = text

    subtitles.set_text("")
    app.processEvents()

    cap.release()
    print("[INFO] Stream finished")


if __name__ == "__main__":
    main()
