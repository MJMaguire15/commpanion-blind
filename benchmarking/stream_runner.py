# benchmarking/stream_runner.py

import asyncio
import threading
from time import monotonic_ns

import cv2
import numpy as np

from benchmarking.sources import FileStreamingSource, CameraSource
from benchmarking.queues import DropOldestQueue
from benchmarking.metrics import Metrics, lag_seconds_from_src
from lipreader import LipReader
from tts import talk_stream

# Sentinel for end-of-stream through frame / roi queues
EOS = (None, None)


def center_crop_resize_norm(frame_bgr: np.ndarray, roi_size: int = 88) -> np.ndarray:
    """
    Simple, robust pre-processing:
      - take full BGR frame,
      - center-crop to square,
      - convert to grayscale,
      - resize to roi_size x roi_size,
      - normalize to [-1, 1].
    This avoids Mediapipe/face-mesh failures for now.
    """
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    norm = (resized.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    return norm


def tts_stub(text: str):
    """Run TTS in a background thread so we don’t block the pipeline."""
    text = (text or "").strip()
    if not text:
        return
    threading.Thread(target=talk_stream, args=(text,), daemon=True).start()


# =====================  PIPELINE STAGES  ===================== #

async def camera_loop(src, q_frames, metrics: Metrics, max_frames=None):
    count = 0
    while True:
        item = src.read()
        if item is None:
            break  # EOF
        t_src, frame = item
        await q_frames.put((t_src, frame))
        metrics.emit("frame_captured", t_src=t_src, lag_s=lag_seconds_from_src(t_src))
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    # Signal end-of-stream
    await q_frames.put(EOS)
    metrics.emit("camera_eos")


async def preproc_loop(q_in, q_out, metrics: Metrics, roi_size: int = 88):
    while True:
        t_src, frame = await q_in.get()
        if (t_src, frame) is EOS:
            await q_out.put(EOS)
            metrics.emit("preproc_eos")
            break

        roi = center_crop_resize_norm(frame, roi_size=roi_size)
        if roi is None:
            metrics.emit("frame_null", t_src=t_src)
            continue

        await q_out.put((t_src, roi))
        metrics.emit("frame_preprocessed", t_src=t_src)


async def window_loop(q_in, q_win, metrics: Metrics,
                      win_len_s: float = 1.5, stride_s: float = 0.5, fps: int = 25):
    """
    Collects fixed-length windows from the preprocessed ROI stream.
      - win_len_s: seconds per window
      - stride_s: seconds between window starts
    """
    buf = []
    need = int(win_len_s * fps)
    step = int(stride_s * fps)

    while True:
        t_src, roi = await q_in.get()
        if (t_src, roi) is EOS:
            # No flushing of partial windows for now; then EOS downstream
            await q_win.put(None)
            metrics.emit("window_eos")
            break

        buf.append((t_src, roi))
        if len(buf) >= need:
            win = buf[:need]
            await q_win.put(win)
            metrics.emit("window_ready", t0=win[0][0], t1=win[-1][0], n=len(win))
            buf = buf[step:]


async def inference_loop(q_win, q_text, metrics: Metrics, backend: str):
    """
    Takes windows of shape [(t0, roi0), ...] and runs LipReader._infer
    on a [T, H, W] numpy array of ROIs.
    """
    # Make window_frames consistent with what this streamrunner uses
    lr = LipReader(backend=backend, window_frames=0)  # window_frames not used directly here

    while True:
        win = await q_win.get()
        if win is None:
            await q_text.put((None, None))
            metrics.emit("inference_eos")
            break

        t0 = win[0][0]
        t1 = win[-1][0]
        rois = [roi for _, roi in win]
        window_arr = np.stack(rois, axis=0)  # [T, H, W]

        t_in_ns = monotonic_ns()
        text = lr._infer(window_arr)
        t_out_ns = monotonic_ns()

        metrics.emit(
            "inference_done",
            t0=t0,
            t1=t1,
            text=text,
            infer_dur_s=(t_out_ns - t_in_ns) / 1e9,
            cap_to_infer_s=(t_out_ns - t0) / 1e9,
        )
        await q_text.put((t0, text))


async def tts_loop(q_text, metrics: Metrics):
    while True:
        t0, text = await q_text.get()
        if t0 is None and text is None:
            metrics.emit("tts_eos")
            break

        t_start_ns = monotonic_ns()
        metrics.emit("tts_started", t0=t0, text=text)

        tts_stub(text)

        t_end_ns = monotonic_ns()
        metrics.emit(
            "tts_finished",
            t0=t0,
            tts_dur_s=(t_end_ns - t_start_ns) / 1e9,
            cap_to_tts_end_s=(t_end_ns - t0) / 1e9,
        )


# =====================  MAIN ENTRYPOINT  ===================== #

async def main(mode="file",
               path="sample.mp4",
               cam_index=0,
               policy="realtime_drop",
               fps: int = 25,
               win: float = 1.5,
               stride: float = 0.5,
               backend: str = "torchscript",
               roi_size: int = 88):
    """
    mode:      'file' or 'camera'
    path:      mp4 path (for file mode)
    cam_index: camera index (for camera mode)
    policy:    'realtime_drop' or 'catchup_no_drop'
    backend:   'torchscript' or 'avhubert' (passed into LipReader)
    """
    metrics = Metrics("./benchmarking/runs/run_stream.jsonl")

    # Source
    if mode == "file":
        src = FileStreamingSource(path, realtime=False)  # as-fast-as-possible
    else:
        src = CameraSource(cam_index)

    # Backpressure policy
    if policy == "realtime_drop":
        Q = lambda n: DropOldestQueue(maxsize=n)
    elif policy == "catchup_no_drop":
        Q = lambda n: asyncio.Queue(maxsize=0)  # unbounded
    else:
        Q = lambda n: DropOldestQueue(maxsize=n)

    q_frames = Q(64)
    q_pre = Q(64)
    q_win = Q(8)
    q_text = Q(16)

    tasks = [
        asyncio.create_task(camera_loop(src, q_frames, metrics)),
        asyncio.create_task(preproc_loop(q_frames, q_pre, metrics, roi_size=roi_size)),
        asyncio.create_task(window_loop(q_pre, q_win, metrics, win_len_s=win, stride_s=stride, fps=fps)),
        asyncio.create_task(inference_loop(q_win, q_text, metrics, backend=backend)),
        asyncio.create_task(tts_loop(q_text, metrics)),
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        metrics.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["file", "camera"], default="file")
    p.add_argument("--path", default="sample.mp4")
    p.add_argument("--cam_index", type=int, default=0)
    p.add_argument("--policy", choices=["realtime_drop", "catchup_no_drop"], default="realtime_drop")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--win", type=float, default=1.5)
    p.add_argument("--stride", type=float, default=0.5)
    p.add_argument("--backend", choices=["torchscript", "avhubert"], default="torchscript")
    p.add_argument("--roi_size", type=int, default=88)
    args = p.parse_args()

    asyncio.run(main(**vars(args)))
