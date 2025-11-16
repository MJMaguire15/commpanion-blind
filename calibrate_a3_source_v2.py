import cv2, time, argparse, math
import numpy as np

def setp(cap, prop, val, name):
    ok = cap.set(prop, val)
    print(f"{name:<26} -> {val:<8} ({'ok' if ok else 'nope'})")
    return ok

def sharpness(frame_bgr):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def rgb_balance_cost(frame_bgr):
    # "Gray target" cost: how far channel means deviate from each other
    r, g, b = [ch.mean() for ch in cv2.split(frame_bgr)]
    return abs(r-g) + abs(b-g)

def open_cap(cam_id, w, h, fps):
    for backend, label in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")]:
        print(f"Opening cam {cam_id} via {label} …")
        cap = cv2.VideoCapture(cam_id, backend)
        if not cap.isOpened():
            print("  could not open"); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        return cap, label
    return None, None

def wait_frames(cap, n=10):
    for _ in range(n):
        cap.read()

def autofocus_cycle(cap, settle_seconds=2.0, sweep_if_supported=True):
    print("\n== Autofocus ==")
    has_af = setp(cap, cv2.CAP_PROP_AUTOFOCUS, 1, "autofocus ON")
    time.sleep(0.2)
    wait_frames(cap, 5)

    # monitor sharpness for a short settle period
    if has_af:
        t0 = time.time()
        best_s = -1.0
        while time.time() - t0 < settle_seconds:
            ok, f = cap.read()
            if not ok: continue
            s = sharpness(f)
            if s > best_s: best_s = s
            time.sleep(0.03)
        # lock focus where it settled
        setp(cap, cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus OFF (lock)")
    else:
        print("autofocus prop not supported; trying manual sweep if possible.")

    # If manual focus prop exists, sweep to maximize sharpness
    try:
        cur = cap.get(cv2.CAP_PROP_FOCUS)
        if math.isnan(cur): raise ValueError
        if sweep_if_supported:
            print("Manual focus sweep …")
            # Range conventions differ: try [0..255] and [0..1]
            candidates = []
            for rng in [(0,255,15), (0,1,0.05)]:
                lo, hi, step = rng
                vals = np.arange(lo, hi+1e-9, step)
                scores = []
                for v in vals:
                    setp(cap, cv2.CAP_PROP_FOCUS, float(v), f"focus {v}")
                    time.sleep(0.05)
                    wait_frames(cap, 2)
                    ok, f = cap.read()
                    if not ok: continue
                    scores.append((sharpness(f), float(v)))
                if scores:
                    best = max(scores, key=lambda x: x[0])
                    candidates.append(best)
            if candidates:
                best = max(candidates, key=lambda x: x[0])
                setp(cap, cv2.CAP_PROP_FOCUS, best[1], f"focus LOCK @ {best[1]}")
    except Exception:
        pass

def white_balance_cycle(cap, settle_seconds=1.0, fine_tune=True):
    print("\n== White balance ==")
    # Let AWB run briefly
    setp(cap, cv2.CAP_PROP_AUTO_WB, 1, "auto_white_balance ON")
    wait_frames(cap, 5)
    time.sleep(settle_seconds)

    ok, frame = cap.read()
    if not ok:
        print("  could not read frame to evaluate WB")
        return

    # lock AWB
    setp(cap, cv2.CAP_PROP_AUTO_WB, 0, "auto_white_balance OFF (lock)")
    # if driver exposes temperature, try small search around current to minimize RGB cost
    temp = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    if not math.isnan(temp) and fine_tune:
        print(f"  base wb_temp={temp}")
        # Try a few nearby temps (K)
        tries = [int(temp + d) for d in (-1000,-500,-250,0,250,500,1000)]
        best = None
        for t in tries:
            ok1 = setp(cap, cv2.CAP_PROP_WB_TEMPERATURE, int(t), f"wb_temperature {t}")
            time.sleep(0.05)
            wait_frames(cap, 2)
            ok2, f = cap.read()
            if not ok2: continue
            cost = rgb_balance_cost(f)
            if best is None or cost < best[0]:
                best = (cost, t)
        if best:
            setp(cap, cv2.CAP_PROP_WB_TEMPERATURE, best[1], f"wb_temperature LOCK {best[1]}")
    else:
        print("  wb_temperature not exposed by driver; AWB locked without temp.")

def exposure_cycle(cap, fps, target_exp_log2=-6, max_gain=6):
    print("\n== Exposure/Gain ==")
    # Manual exposure (0.25 on DSHOW, 0 or 1 on MSMF vary; we try 0.25 first)
    setp(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure MANUAL (0.25)")
    setp(cap, cv2.CAP_PROP_EXPOSURE, target_exp_log2, "exposure (log2-ish)")
    setp(cap, cv2.CAP_PROP_GAIN, 0, "gain 0")
    wait_frames(cap, 5)

    # Nudge exposure/gain around target to avoid clipping and hit mid-gray
    def luma_stats(bgr):
        y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float32)/255.0
        return float(y.mean()), float((y>0.98).mean()*100), float((y<0.02).mean()*100)

    target = 0.45
    for _ in range(30):
        ok, f = cap.read()
        if not ok: continue
        mean, pct_hi, pct_lo = luma_stats(f)
        if pct_hi > 2.0:
            # highlights clipped -> reduce exposure
            cur = cap.get(cv2.CAP_PROP_EXPOSURE)
            setp(cap, cv2.CAP_PROP_EXPOSURE, cur-1, "exposure--")
        elif mean < target*0.95:
            # too dark -> raise exposure up to about -4, then add gain
            cur_e = cap.get(cv2.CAP_PROP_EXPOSURE)
            if cur_e < -4:
                setp(cap, cv2.CAP_PROP_EXPOSURE, cur_e+1, "exposure++")
            else:
                cur_g = cap.get(cv2.CAP_PROP_GAIN)
                if cur_g < max_gain:
                    setp(cap, cv2.CAP_PROP_GAIN, cur_g+1, "gain++")
                else:
                    break
        elif mean > target*1.05:
            # too bright -> reduce gain first, then exposure
            cur_g = cap.get(cv2.CAP_PROP_GAIN)
            if cur_g > 0:
                setp(cap, cv2.CAP_PROP_GAIN, cur_g-1, "gain--")
            else:
                cur_e = cap.get(cv2.CAP_PROP_EXPOSURE)
                setp(cap, cv2.CAP_PROP_EXPOSURE, cur_e-1, "exposure--")
        else:
            break
        time.sleep(0.06)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", default="1")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    cam_id = int(args.camera_id) if str(args.camera_id).isdigit() else args.camera_id
    cap, backend = open_cap(cam_id, args.width, args.height, args.fps)
    if not cap:
        raise SystemExit("Cannot open camera.")

    exposure_cycle(cap, fps=args.fps, target_exp_log2=-6, max_gain=6)
    autofocus_cycle(cap, settle_seconds=2.0, sweep_if_supported=True)
    white_balance_cycle(cap, settle_seconds=1.0, fine_tune=True)

    # Snapshot after calibration
    ok, frame = cap.read()
    if ok:
        cv2.imwrite("calibration_snapshot.jpg", frame)
        print("\nSaved: calibration_snapshot.jpg")

    print("\n== Final values ==")
    for prop, name in [
        (cv2.CAP_PROP_AUTO_EXPOSURE, "auto_exposure"),
        (cv2.CAP_PROP_EXPOSURE,      "exposure"),
        (cv2.CAP_PROP_GAIN,          "gain"),
        (cv2.CAP_PROP_AUTOFOCUS,     "autofocus"),
        (cv2.CAP_PROP_FOCUS,         "focus"),
        (cv2.CAP_PROP_AUTO_WB,       "auto_wb"),
        (cv2.CAP_PROP_WB_TEMPERATURE,"wb_temp"),
    ]:
        try:
            print(f"{name:<18} = {cap.get(prop)}")
        except Exception:
            pass

    cap.release()
    print(f"\nbackend: {backend}")
    print("Re-run this before each recording, or copy the setp(...) calls into your capture script.")

if __name__ == "__main__":
    main()
