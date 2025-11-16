# benchmarking/stream_runner.py

import asyncio
import threading
from time import monotonic_ns

import numpy as np

from benchmarking.sources import FileStreamingSource, CameraSource
from benchmarking.queues import DropOldestQueue
from benchmarking.metrics import Metrics, lag_seconds_from_src

from lipreader import LipReader, MouthCropper
from tts import talk_stream

# Sentinel used to signal end-of-stream through the pipeline
EOS = (None, None)


def tts_stub(text: str) -> None:
    """
    Fire-and-forget TTS so that the main pipeline is not blocked by audio playback.
    """
    text = (text or "").strip()
    if not text:
        return
    threading.Thread(target=talk_stream, args=(text,), daemon=True).start()


async def camera_loop(src, q_frames, metrics, max_frames=None):
    count = 0
    while True:
        item = src.read()
        if item is None:
            break  # EOF

        t_src, frame = item

        # Optional guard: skip bad frames here too
        if frame is None:
            metrics.emit("frame_null_src", t_src=t_src)
            continue

        await q_frames.put((t_src, frame))
        metrics.emit("frame_captured", t_src=t_src, lag_s=lag_seconds_from_src(t_src))
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    await q_frames.put(EOS)
    metrics.emit("camera_eos")



async def preproc_loop(q_in, q_out, metrics):
    """
    Convert raw BGR frames to mouth ROIs using MouthCropper.
    Drops frames where no frame or no mouth ROI is found.
    """
    cropper = MouthCropper()

    while True:
        t_src, frame = await q_in.get()
        if (t_src, frame) is EOS:
            # End-of-stream: propagate and exit
            await q_out.put(EOS)
            metrics.emit("preproc_eos")
            break

        # NEW: skip null frames defensively
        if frame is None:
            metrics.emit("frame_null", t_src=t_src)
            continue

        roi = cropper.get_mouth_roi(frame)
        if roi is None:
            # No face / mouth detected – just drop this frame
            continue

        await q_out.put((t_src, roi))
        metrics.emit("frame_preprocessed", t_src=t_src)



async def window_loop(q_in, q_win, metrics, win_len_s=1.5, stride_s=0.5, fps=25):
    """
    Build sliding windows of ROIs: length win_len_s, stride stride_s, at given fps.
    Emits window_ready + window_eos.
    """
    buf = []
    need = int(win_len_s * fps)
    step = int(stride_s * fps)

    while True:
        t_src, x = await q_in.get()
        if (t_src, x) is EOS:
            # Optional: flush a final partial window if you want.
            await q_win.put(None)  # sentinel to inference_loop
            metrics.emit("window_eos")
            break

        buf.append((t_src, x))
        if len(buf) >= need:
            win = buf[:need]
            await q_win.put(win)
            metrics.emit("window_ready", t0=win[0][0], t1=win[-1][0], n=len(win))
            buf = buf[step:]


async def inference_loop(q_win, q_text, metrics, backend="avhubert"):
    """
    Consume windows, run lipreading model, push text segments into q_text.
    Uses LipReader._load_model() + _infer(window) so it shares weights with your
    existing setup. Emits inference_done + inference_eos.
    """
    lr = LipReader(backend=backend)

    # Load model once at startup
    try:
        lr._load_model()
    except Exception as e:
        print(f"[stream_runner] LipReader _load_model() failed: {e}")

    while True:
        win = await q_win.get()
        if win is None:
            # No more windows
            await q_text.put((None, None))
            metrics.emit("inference_eos")
            break

        t0 = win[0][0]
        t1 = win[-1][0]
        # Stack ROIs to [T, H, W]
        window = np.stack([x for _, x in win], axis=0)

        t_in_ns = monotonic_ns()
        text = lr._infer(window)
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


async def tts_loop(q_text, metrics):
    """
    Read decoded text windows and send them to TTS.
    Emits tts_started / tts_finished and a final tts_eos.
    """
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


async def main(
    mode: str = "file",
    path: str = "sample.mp4",
    cam_index: int = 0,
    policy: str = "realtime_drop",
    fps: int = 25,
    win: float = 1.5,
    stride: float = 0.5,
    backend: str = "avhubert",
):
    """
    Top-level orchestrator.

    mode = "file"   -> stream from an mp4 as if it were live
    mode = "camera" -> stream from ThinkReality A3 camera index
    backend = "avhubert" or "torchscript" (whatever you actually have set up)
    """
    metrics = Metrics("benchmarking/runs/run_stream.jsonl")

    # Source: offline file or live camera
    if mode == "file":
        src = FileStreamingSource(path, realtime=False)  # as fast as possible
    else:
        src = CameraSource(cam_index, target_fps=fps)

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
        asyncio.create_task(preproc_loop(q_frames, q_pre, metrics)),
        asyncio.create_task(
            window_loop(q_pre, q_win, metrics, win_len_s=win, stride_s=stride, fps=fps)
        ),
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
    p.add_argument("--backend", choices=["avhubert", "torchscript"], default="avhubert")
    args = p.parse_args()

    asyncio.run(main(**vars(args)))
