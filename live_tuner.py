# live_tuner.py
import cv2, time, argparse, os, sys, json
from pathlib import Path

# Windows-only non-blocking keyboard input
try:
    import msvcrt
except ImportError:
    msvcrt = None  # if not Windows, keys won't work

PRESET_PATH = Path("presets/cam1.json")

def setp(cap, prop, val, name):
    ok = cap.set(prop, val)
    print(f"{name:<22} -> {val:<8} ({'ok' if ok else 'nope'})")
    return ok

def report(cap):
    def get(prop):
        try: return cap.get(prop)
        except: return None
    print("\n--- Camera report ---")
    print(f"auto_exposure        : {get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"exposure             : {get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"gain                 : {get(cv2.CAP_PROP_GAIN)}")
    print(f"auto_white_balance   : {get(cv2.CAP_PROP_AUTO_WB)}")
    print(f"wb_temperature       : {get(cv2.CAP_PROP_WB_TEMPERATURE)}")
    print(f"autofocus            : {get(cv2.CAP_PROP_AUTOFOCUS)}")
    print(f"focus                : {get(cv2.CAP_PROP_FOCUS)}")
    print(f"brightness           : {get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"saturation           : {get(cv2.CAP_PROP_SATURATION)}")
    print(f"sharpness            : {get(cv2.CAP_PROP_SHARPNESS)}")
    print("---------------------\n")

def open_capture(cam_id, width, height, fps, backend="auto"):
    """Try explicit backend order; cam_id can be int or 'video=Name'.
       Returns (cap, used_backend_str) where used_backend_str is 'dshow' or 'msmf' or None."""
    def _mk(arg, be):
        cap = cv2.VideoCapture(arg, be)
        return cap if cap.isOpened() else None

    # Normalize ID
    if isinstance(cam_id, str) and cam_id.isdigit():
        cam_id = int(cam_id)

    # Attempt order
    attempts = []
    if backend.lower() == "dshow":
        attempts = [("dshow", cv2.CAP_DSHOW)]
    elif backend.lower() == "msmf":
        attempts = [("msmf", cv2.CAP_MSMF)]
    else:
        attempts = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]

    cap = None; used = None
    for label, be in attempts:
        if isinstance(cam_id, str):
            name = cam_id if cam_id.lower().startswith("video=") else f"video={cam_id}"
            cap = _mk(name, be); used = label
            if cap: break
        else:
            cap = _mk(cam_id, be); used = label
            if cap: break

    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        print(f"[camera] opened via {used.upper()} ({'name' if isinstance(cam_id,str) else 'index'}: {cam_id})")
    else:
        print(f"[camera] failed: {cam_id} (backend={backend})")
    return cap, used

def save_preset(cap, path=PRESET_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    preset = {
        "CAP_PROP_AUTO_EXPOSURE":  cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "CAP_PROP_EXPOSURE":       cap.get(cv2.CAP_PROP_EXPOSURE),
        "CAP_PROP_GAIN":           cap.get(cv2.CAP_PROP_GAIN),
        "CAP_PROP_AUTO_WB":        cap.get(cv2.CAP_PROP_AUTO_WB),
        "CAP_PROP_WB_TEMPERATURE": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "CAP_PROP_AUTOFOCUS":      cap.get(cv2.CAP_PROP_AUTOFOCUS),
        "CAP_PROP_FOCUS":          cap.get(cv2.CAP_PROP_FOCUS),
        "CAP_PROP_BRIGHTNESS":     cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "CAP_PROP_SATURATION":     cap.get(cv2.CAP_PROP_SATURATION),
        "CAP_PROP_SHARPNESS":      cap.get(cv2.CAP_PROP_SHARPNESS),
    }
    with open(path, "w") as f:
        json.dump(preset, f, indent=2)
    print(f"Saved preset → {path}")

def apply_preset(cap, path=PRESET_PATH):
    if not path.exists():
        print(f"No preset at {path}"); return
    with open(path, "r") as f:
        p = json.load(f)
    # Disable autos first (where relevant), then set manual values
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,   p.get("CAP_PROP_AUTO_EXPOSURE", 0.25))
    cap.set(cv2.CAP_PROP_EXPOSURE,        p.get("CAP_PROP_EXPOSURE", -6))
    cap.set(cv2.CAP_PROP_GAIN,            p.get("CAP_PROP_GAIN", 0))
    cap.set(cv2.CAP_PROP_AUTO_WB,         p.get("CAP_PROP_AUTO_WB", 0))
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,  p.get("CAP_PROP_WB_TEMPERATURE", 4500))
    cap.set(cv2.CAP_PROP_AUTOFOCUS,       p.get("CAP_PROP_AUTOFOCUS", 0))
    cap.set(cv2.CAP_PROP_FOCUS,           p.get("CAP_PROP_FOCUS", 0))
    cap.set(cv2.CAP_PROP_BRIGHTNESS,      p.get("CAP_PROP_BRIGHTNESS", 0))
    cap.set(cv2.CAP_PROP_SATURATION,      p.get("CAP_PROP_SATURATION", 0))
    cap.set(cv2.CAP_PROP_SHARPNESS,       p.get("cv2.CAP_PROP_SHARPNESS", 0))
    print(f"Applied preset from {path}")

def open_property_dialog(cap):
    """Open the native DirectShow property page (Windows, DSHOW only)."""
    try:
        ok = cap.set(cv2.CAP_PROP_SETTINGS, 1)
        print("Opened camera property dialog" if ok else "Property dialog not supported")
    except cv2.error:
        print("Property dialog not supported by this backend/driver.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", default="1",
                    help='Index (e.g. 1) or device name (e.g. "video=ThinkReality A3")')
    ap.add_argument("--backend", choices=["auto","dshow","msmf"], default="auto",
                    help="Force a specific backend")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--record", type=str, default="", help="Optional MP4 path to save stream")
    ap.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = until q)")
    args = ap.parse_args()

    cap, used_backend = open_capture(args.camera_id, args.width, args.height, args.fps, backend=args.backend)
    if not cap or not cap.isOpened():
        raise SystemExit(f"Cannot open camera: {args.camera_id} (backend={args.backend})")
    print(f"[camera] backend in use: {used_backend}")

    vw = None
    if args.record:
        p = Path(args.record); p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(p), fourcc, args.fps, (args.width, args.height))
        if not vw.isOpened():
            print("Warning: could not open VideoWriter; continuing without recording.")
            vw = None

    print("""\nLive tuner hotkeys:
  e / E  : exposure +1 / -1
  g / G  : gain +1 / -1
  a      : toggle auto-exposure (0.25 manual, 0.75 auto on many DSHOW drivers)
  w      : toggle auto white balance
  ] / [  : white-balance temperature +100 / -100 (if supported)
  f      : toggle autofocus (if supported)
  ) / (  : manual focus +2 / -2 (if supported)
  o      : OPEN native camera property dialog (DSHOW only)
  p      : save preset to presets/cam1.json
  R      : re-apply preset now
  s      : save snapshot JPEG
  r      : report values
  q      : quit
""")

    start = time.time()
    last_info = time.time()
    report(cap)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        if vw is not None:
            vw.write(frame)

        if time.time() - last_info > 5:
            exp = cap.get(cv2.CAP_PROP_EXPOSURE)
            gain = cap.get(cv2.CAP_PROP_GAIN)
            wb   = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
            print(f"[tick] exposure={exp}  gain={gain}  wb_temp={wb}")
            last_info = time.time()

        if args.duration and (time.time() - start) >= args.duration:
            break

        if msvcrt and msvcrt.kbhit():
            ch = msvcrt.getch()
            if not ch: continue
            key = ch.decode(errors="ignore")

            if   key == 'q': break
            elif key == 'r': report(cap)
            elif key == 's':
                name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(name, frame); print(f"Saved {name}")
            elif key == 'p': save_preset(cap)
            elif key == 'R': apply_preset(cap)

            elif key == 'o':
                if used_backend == "dshow":
                    open_property_dialog(cap)
                else:
                    print("Property dialog only available with DSHOW. Re-run with --backend dshow.")

            elif key == 'e':
                cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                setp(cap, cv2.CAP_PROP_EXPOSURE, cur + 1, "exposure")
            elif key == 'E':
                cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                setp(cap, cv2.CAP_PROP_EXPOSURE, cur - 1, "exposure")
            elif key == 'g':
                cur = cap.get(cv2.CAP_PROP_GAIN)
                setp(cap, cv2.CAP_PROP_GAIN, cur + 1, "gain")
            elif key == 'G':
                cur = cap.get(cv2.CAP_PROP_GAIN)
                setp(cap, cv2.CAP_PROP_GAIN, max(0, cur - 1), "gain")
            elif key == 'a':
                cur = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                nxt = 0.75 if cur < 0.5 else 0.25
                setp(cap, cv2.CAP_PROP_AUTO_EXPOSURE, nxt, "auto_exposure")
            elif key == 'w':
                cur = cap.get(cv2.CAP_PROP_AUTO_WB)
                nxt = 0 if cur >= 0.5 else 1
                setp(cap, cv2.CAP_PROP_AUTO_WB, nxt, "auto_white_balance")
            elif key == ']':
                cur = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
                if cur and cur > 0:
                    setp(cap, cv2.CAP_PROP_WB_TEMPERATURE, cur + 100, "wb_temperature")
                else:
                    print("wb_temperature not supported")
            elif key == '[':
                cur = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
                if cur and cur > 0:
                    setp(cap, cv2.CAP_PROP_WB_TEMPERATURE, max(2800, cur - 100), "wb_temperature")
                else:
                    print("wb_temperature not supported")
            elif key == 'f':
                cur = cap.get(cv2.CAP_PROP_AUTOFOCUS)
                nxt = 0 if cur >= 0.5 else 1
                setp(cap, cv2.CAP_PROP_AUTOFOCUS, nxt, "autofocus")
            elif key == ')':
                cur = cap.get(cv2.CAP_PROP_FOCUS)
                setp(cap, cv2.CAP_PROP_FOCUS, min(255, cur + 2), "focus")
            elif key == '(':
                cur = cap.get(cv2.CAP_PROP_FOCUS)
                setp(cap, cv2.CAP_PROP_FOCUS, max(0, cur - 2), "focus")

    cap.release()
    if vw is not None: vw.release()
    print("Closed camera.")

if __name__ == "__main__":
    main()
