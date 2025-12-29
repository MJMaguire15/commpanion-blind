# record_10s_and_run.py
import os, sys, time, cv2, argparse, subprocess, datetime, re
from pathlib import Path

# ===================== BACKEND SWITCH =====================
USE_NPU = False   # ← set True later when QNN backend is implemented
# ==========================================================

from inference_cpu import run_from_roi_root as run_cpu
from inference_npu import load_model as load_npu, run as run_npu

# --- defaults ---
DEFAULT_FPS = 25
DEFAULT_W, DEFAULT_H = 1280, 720
OUT_ROOT = Path("sessions")
CKPT_DEFAULT = r".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"

# ---- TTS (unchanged) ----
_TTS_ENGINE = None
def speak_now(text: str, rate: int = 200):
    global _TTS_ENGINE
    text = (text or "").strip()
    if not text:
        return
    if _TTS_ENGINE is None:
        from tts import _TTS
        _TTS_ENGINE = _TTS(rate=rate)
    _TTS_ENGINE.start(text)
# -------------------------

def open_capture(cam_id, width, height, fps):
    def _mk(arg, backend):
        cap = cv2.VideoCapture(arg, backend)
        return cap if cap.isOpened() else None

    if isinstance(cam_id, str) and cam_id.isdigit():
        cam_id = int(cam_id)

    if isinstance(cam_id, str):
        name = cam_id if cam_id.lower().startswith("video=") else f"video={cam_id}"
        cap = _mk(name, cv2.CAP_MSMF) or _mk(name, cv2.CAP_DSHOW)
    else:
        cap = _mk(cam_id, cv2.CAP_MSMF) or _mk(cam_id, cv2.CAP_DSHOW)

    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap

def countdown_and_record(cap, out_mp4, seconds, width, height, fps, preview=True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    print("Get ready…")
    for t in (3, 2, 1):
        print(t, flush=True); time.sleep(1)
    print("Recording…", flush=True)

    t0 = time.time()
    next_tick = t0 + 1
    sec = 0

    while (time.time() - t0) < seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        vw.write(frame)

        if preview:
            try:
                cv2.imshow("Preview (recording)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except cv2.error:
                preview = False

        if time.time() >= next_tick:
            sec += 1
            print(sec, flush=True)
            next_tick += 1

    vw.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"Saved raw video: {out_mp4}")

def run_mouth_crop(raw_mp4, lips_mp4, fps, scale=2.2, size=88):
    crop_py = Path("mouth_crop_to_mp4.py")
    args = [
        sys.executable, str(crop_py),
        "-i", str(raw_mp4),
        "-o", str(lips_mp4),
        "--size", str(size),
        "--scale", str(scale),
        "--fps", str(fps),
        "--root-dir", str(lips_mp4.parent.parent),
        "--dataset", "custom",
        "--clip", lips_mp4.name,
        "--write-list"
    ]
    print("Creating lips-only clip…")
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise RuntimeError("mouth_crop_to_mp4 failed.")
    if proc.stdout.strip():
        print(proc.stdout.strip())

def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, 0.0, 0, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return True, frames, fps, w, h

def playback(mp4_path, title, fps):
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        print(f"Playing: {mp4_path} (ESC to close)")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(title, frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception:
        if os.name == "nt":
            os.startfile(str(mp4_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", default="1")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--ckpt", default=os.environ.get("AUTO_AVSR_CKPT", CKPT_DEFAULT))
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-playback", action="store_true")
    ap.add_argument("--scale", type=float, default=2.2)
    ap.add_argument("--size", type=int, default=88)
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--speak-rate", type=int, default=200)
    args = ap.parse_args()

    cap = open_capture(args.camera_id, args.width, args.height, args.fps)
    if not cap:
        raise SystemExit("Cannot open camera")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session = OUT_ROOT / ts
    raw_dir = session / "raw"
    roi_root = session / "roi_root"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (roi_root / "labels").mkdir(parents=True, exist_ok=True)
    (roi_root / "custom").mkdir(parents=True, exist_ok=True)

    raw_mp4 = raw_dir / f"raw_{ts}.mp4"
    lips_mp4 = roi_root / "custom" / "clip_0001.mp4"
    list_csv = roi_root / "labels" / "list.csv"

    countdown_and_record(cap, raw_mp4, args.seconds,
                         args.width, args.height, args.fps,
                         preview=not args.no_preview)
    cap.release()

    run_mouth_crop(raw_mp4, lips_mp4, args.fps, args.scale, args.size)

    ok, frames, fps, w, h = inspect_video(lips_mp4)
    print(f"ROI video: frames={frames}, fps={fps:.2f}, size={w}x{h}")

    if not args.no_playback:
        playback(raw_mp4, "Raw", args.fps)
        playback(lips_mp4, "Lips", args.fps)

    # ================== INFERENCE SWITCH ==================
    if USE_NPU:
        engine = load_npu("artifacts/lipreader.qnn")
        text = run_npu(engine, None)
    else:
        text = run_cpu(
            roi_root=str(roi_root),
            test_file=list_csv.name,
            ckpt_path=args.ckpt
        )
    # ======================================================

    print("\n🗣️ Transcript:", text if text else "(empty)")

    if text and not args.no_speak:
        speak_now(text, rate=args.speak_rate)

if __name__ == "__main__":
    main()
