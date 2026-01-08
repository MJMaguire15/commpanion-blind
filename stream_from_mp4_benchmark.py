# stream_from_mp4_benchmark.py
import sys
import time
import argparse
import subprocess
from pathlib import Path
from collections import deque
import re

import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication

from recap import SemanticRecap
from subtitle_overlay import (
    BlackBaseLayer,
    LowerDimLayer,
    SubtitleOverlay,
    DIM_ENABLED
)
from inference_cpu_stream import load_model, run

# ===================== DEFAULTS =====================
DEFAULT_VIDEO = "input_test_video.mp4"
DEFAULT_WINDOW_S = 8.0
DEFAULT_STEP_S = 2.0
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


# ---------- ROI ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_mouth_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) == 0:
        h, w = gray.shape
        s = ROI_SIZE // 2
        cx, cy = w // 2, h // 2
        crop = gray[cy - s:cy + s, cx - s:cx + s]
        return cv2.resize(crop, (ROI_SIZE, ROI_SIZE))

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    mouth_region = gray[y + int(0.55 * h): y + h, x: x + w]
    return cv2.resize(mouth_region, (ROI_SIZE, ROI_SIZE))


# ---------- Dedup ----------
def deduplicate_text(prev, curr, min_prefix_chars=24, min_chars=20):
    if not prev or not curr or len(curr) < min_chars:
        return curr

    p, c = prev.strip()[:min_prefix_chars], curr.strip()[:min_prefix_chars]
    i = 0
    for a, b in zip(p, c):
        if a == b:
            i += 1
        else:
            break

    return curr[i:].lstrip() if i >= min_prefix_chars // 2 else curr


# ---------- Simple semantic summariser ----------
KEYWORDS = {
    "intro": {"today", "video", "talk", "topic"},
    "time": {"time", "study", "practice", "spend"},
    "difficulty": {"hard", "difficult", "frustrated", "sad", "nervous"},
    "mistakes": {"mistake", "mistakes", "wrong", "error"},
    "motivation": {"fun", "enjoy", "love", "friends"}
}

def extract_keywords(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    active = set()
    for label, kws in KEYWORDS.items():
        if words & kws:
            active.add(label)
    return active


def semantic_recap_from_text(text):
    kws = extract_keywords(text)

    if "mistakes" in kws:
        return "They explain that making mistakes is a normal part of learning."
    if "difficulty" in kws:
        return "They describe the emotional challenges of learning a language."
    if "time" in kws:
        return "They explain that learning a language takes a lot of time and practice."
    if "motivation" in kws:
        return "They talk about enjoying language learning and meeting new people."
    if "intro" in kws:
        return "They introduce the topic and explain what the video is about."

    return "They continue discussing language learning."


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--youtube", default="")
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_S)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--recap-only", action="store_true")
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
    engine = load_model(CKPT)

    # ---------- AR / Headless ----------
    subtitles = None
    app = None
    if not args.headless:
        app = QApplication.instance() or QApplication(sys.argv)
        screens = app.screens()
        if len(screens) < 2:
            raise RuntimeError("A3 must be in EXTENDED display mode")

        geom = screens[-1].geometry()
        BlackBaseLayer(geom)
        if DIM_ENABLED:
            LowerDimLayer(geom)
        subtitles = SubtitleOverlay(geom)
    else:
        print("[INFO] Headless mode enabled")

    buffer = deque(maxlen=window_frames)
    frames_since_decode = 0
    frame_idx = 0
    last_text = ""
    last_clean = ""

    recap_buffer = deque()
    last_recap = ""
    last_keywords = set()
    last_recap_time = 0
    RECAP_INTERVAL = 10

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        buffer.append(extract_mouth_roi(frame))
        frames_since_decode += 1

        if len(buffer) == window_frames and frames_since_decode >= step_frames:
            frames_since_decode = 0
            clip = torch.from_numpy(np.stack(buffer)[:, None]).float() / 255.0

            t0 = time.time()
            text = run(engine, clip)
            latency = (time.time() - t0) * 1000
            ts = frame_idx / fps

            if text and text != last_text:
                clean = deduplicate_text(last_clean, text)
                if clean:
                    print(f"[{ts:7.2f}s | {latency:6.1f} ms] {clean}")
                    recap_buffer.append(clean)

                    now = time.time()
                    window_text = " ".join(recap_buffer)
                    keywords = extract_keywords(window_text)

                    if (
                        now - last_recap_time >= RECAP_INTERVAL
                        and keywords != last_keywords
                    ):
                        recap = semantic_recap_from_text(window_text)

                        if recap != last_recap:
                            label = "Summary (last 10s)" if args.window_seconds <= 12 else "Summary (last 20s)"
                            recap_text = f"[ {label.upper()} ]\n{recap}"
                            print(f"[RECAP] {recap_text}")

                            if subtitles:
                                subtitles.set_text(recap_text)
                                app.processEvents()

                            last_recap = recap
                            last_keywords = keywords
                            last_recap_time = now
                            recap_buffer.clear()

                    if subtitles and not args.recap_only:
                        subtitles.set_text(clean)
                        app.processEvents()

                    last_clean = clean
                last_text = text

    if subtitles:
        subtitles.set_text("")
        app.processEvents()

    cap.release()
    print("[INFO] Stream finished")


if __name__ == "__main__":
    main()
