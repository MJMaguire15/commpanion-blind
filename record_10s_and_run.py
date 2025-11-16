# record_10s_and_run.py
import os, sys, time, cv2, argparse, subprocess, datetime, re
from pathlib import Path

# --- your defaults ---
DEFAULT_FPS = 25
DEFAULT_W, DEFAULT_H = 1280, 720
OUT_ROOT = Path("sessions")
CKPT_DEFAULT = r".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"

# ---- use your existing TTS (_TTS from tts.py) ----
_TTS_ENGINE = None
def speak_now(text: str, rate: int = 200):
    """Speak using your existing pyttsx3-based TTS (Windows default output)."""
    global _TTS_ENGINE
    text = (text or "").strip()
    if not text:
        return
    if _TTS_ENGINE is None:
        from tts import _TTS  # your original class
        _TTS_ENGINE = _TTS(rate=rate)
    _TTS_ENGINE.start(text)
# --------------------------------------------------

def open_capture(cam_id, width, height, fps):
    """Open by MSMF then DSHOW, supports int index or 'video=Device Name'."""
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
    last_sec_printed = 0
    preview_used = False

    while (time.time() - t0) < seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        vw.write(frame)

        if preview:
            try:
                cv2.imshow("Preview (recording)", frame)
                preview_used = True
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except cv2.error:
                preview = False  # no GUI backend

        now = time.time()
        if now >= next_tick:
            last_sec_printed += 1
            print(last_sec_printed, flush=True)
            next_tick += 1

        time.sleep(max(0, (1.0 / fps) - (time.time() - now)))

    vw.release()
    if preview_used:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"Saved raw video: {out_mp4}")

def run_mouth_crop(raw_mp4, lips_mp4, fps, scale=2.2, size=88):
    """Call your existing mouth_crop_to_mp4.py to produce lips-only mp4 + labels/list.csv."""
    crop_py = Path("mouth_crop_to_mp4.py")
    if not crop_py.exists():
        raise FileNotFoundError("mouth_crop_to_mp4.py not found in current folder.")
    args = [
        sys.executable, str(crop_py),
        "-i", str(raw_mp4),
        "-o", str(lips_mp4),
        "--size", str(size),
        "--scale", str(scale),
        "--fps", str(fps),
        "--root-dir", str(lips_mp4.parent.parent),  # sessions/<ts>/roi_root
        "--dataset", "custom",
        "--clip", lips_mp4.name,
        "--write-list"
    ]
    print("Creating lips-only clip…")
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("mouth_crop_to_mp4 failed.")
    if proc.stdout.strip():
        print(proc.stdout.strip())

def inspect_video(path):
    """Return (ok, frames, fps, w, h) without showing UI."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, 0.0, 0, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return True, frames, fps, w, h

def run_auto_avsr(roi_root, list_csv, ckpt_path, log_dir):
    """Call auto_avsr/eval.py once and return transcript + write logs."""
    eval_py = Path("auto_avsr") / "eval.py"
    if not eval_py.exists():
        raise FileNotFoundError("auto_avsr/eval.py not found beside this script.")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    py_exe = Path(".") / ".venv" / "Scripts" / "python.exe"
    py = str(py_exe) if py_exe.exists() else sys.executable

    args = [
        py, str(eval_py),
        "--modality", "video",
        "--root-dir", str(roi_root),
        "--test-file", str(list_csv.name),  # eval.py resolves under root/labels/
        "--pretrained-model-path", str(ckpt_path),
    ]
    print("Running Auto-AVSR decode…")
    proc = subprocess.run(args, cwd=str(Path(".").resolve()), capture_output=True, text=True)

    # Write logs
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (log_dir / "eval_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

    # Console tail for quick visibility
    lines = (proc.stdout or "").splitlines()
    tail  = "\n".join(lines[-20:])
    if tail.strip():
        print("\n--- eval.py last 20 lines ---")
        print(tail)
        print("--- end ---\n")

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("auto_avsr/eval.py failed.")

    # ===== Robust transcript picker =====
    def pick_transcript(stdout: str) -> str:
        L = [ln.rstrip() for ln in (stdout or "").splitlines() if ln.strip()]
        # 1) Extract any [DECODE] ... that appears anywhere on the line (handles progress-bar prefix)
        decode_re = re.compile(r"\[DECODE\]\s*(.+)$")
        for ln in reversed(L):
            m = decode_re.search(ln)
            if m:
                return " ".join(m.group(1).split())
        # 2) Prediction:/Hypo: anywhere in the line
        for ln in reversed(L):
            m = re.search(r"(?:^|\s)(?:Prediction:|Hypo:)\s*(.+)$", ln, re.IGNORECASE)
            if m:
                return " ".join(m.group(1).split())
        # 3) Otherwise pick the last line that looks like language, ignoring metrics/progress/headers
        skip_tokens = {"wer", "cer", "testing", "dataloader", "metric", "pretrained weights"}
        for ln in reversed(L):
            low = ln.lower()
            if any(tok in low for tok in skip_tokens):
                continue
            if re.search(r"[A-Za-z]", ln) and not re.fullmatch(r"[-=~_]+", ln.strip()):
                return " ".join(ln.split())
        return ""

    return pick_transcript(proc.stdout)

def playback(mp4_path, title, fps):
    # Try OpenCV; if unavailable, open with system default player
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open with OpenCV")
        print(f"Playing: {mp4_path}  (press ESC to close)")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(title, frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    except Exception:
        print(f"OpenCV playback not available — opening {mp4_path} with the system player.")
        p = str(mp4_path)
        if os.name == "nt":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", default="1",
        help='Camera index (e.g. 0/1/2) or device name (e.g. "video=ThinkReality A3")')
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--ckpt", default=os.environ.get("AUTO_AVSR_CKPT", CKPT_DEFAULT))
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-playback", action="store_true")
    ap.add_argument("--scale", type=float, default=2.2, help="mouth bbox expansion (try 2.2–2.6)")
    ap.add_argument("--size", type=int, default=88, help="mouth crop size (e.g., 88 or 112)")
    ap.add_argument("--no-speak", action="store_true", help="do not speak transcript automatically")
    ap.add_argument("--speak-rate", type=int, default=200, help="pyttsx3 words-per-minute")
    args = ap.parse_args()

    cap = open_capture(args.camera_id, args.width, args.height, args.fps)
    if not cap or not cap.isOpened():
        raise SystemExit(f"Cannot open camera: {args.camera_id}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUT_ROOT / ts
    raw_dir = session_dir / "raw"
    roi_root = session_dir / "roi_root"
    logs_dir = session_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (roi_root / "labels").mkdir(parents=True, exist_ok=True)
    (roi_root / "custom").mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_mp4 = raw_dir / f"raw_{ts}.mp4"
    lips_mp4 = roi_root / "custom" / "clip_0001.mp4"
    list_csv = roi_root / "labels" / "list.csv"

    countdown_and_record(cap, raw_mp4, seconds=args.seconds,
                         width=args.width, height=args.height, fps=args.fps,
                         preview=not args.no_preview)
    cap.release()

    run_mouth_crop(raw_mp4, lips_mp4, fps=args.fps, scale=args.scale, size=args.size)

    ok, frames, vfps, vw, vh = inspect_video(lips_mp4)
    with (logs_dir / "roi_video_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"path={lips_mp4}\nframes={frames}\nfps={vfps}\nsize={vw}x{vh}\n")
    print(f"ROI video: frames={frames}, fps={vfps:.2f}, size={vw}x{vh}")

    if not args.no_playback:
        playback(raw_mp4, "Raw (10s)", fps=args.fps)
        playback(lips_mp4, "Lips-only (10s)", fps=args.fps)

    text = run_auto_avsr(roi_root, list_csv, args.ckpt, logs_dir)
    print("\n🗣️ Transcript:", text if text else "(empty)")

    # === auto-speak right after decoding (uses your old TTS) ===
    if text and not args.no_speak:
        try:
            speak_now(text, rate=args.speak_rate)
        except Exception as e:
            print(f"⚠️ TTS error: {e}")

if __name__ == "__main__":
    main()
