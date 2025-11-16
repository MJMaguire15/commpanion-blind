import cv2, time, argparse
import numpy as np

def setp(cap, prop, val, name):
    ok = cap.set(prop, val)
    print(f"{name:<24} -> {val:<8} ({'ok' if ok else 'nope'})")
    return ok

def get_luma_stats(frame_bgr):
    y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float32)/255.0
    mean = float(y.mean())
    pct_hi = float((y > 0.98).mean()*100.0)   # highlight clipping %
    pct_lo = float((y < 0.02).mean()*100.0)   # black clipping %
    sharp = cv2.Laplacian(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return mean, pct_hi, pct_lo, sharp

def open_cap(cam_id, width, height, fps):
    # Try MSMF then DSHOW
    for backend,name in [(cv2.CAP_MSMF,"MSMF"), (cv2.CAP_DSHOW,"DSHOW")]:
        print(f"Opening cam {cam_id} via {name}…")
        cap = cv2.VideoCapture(cam_id, backend)
        if not cap.isOpened(): 
            print("  could not open")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        return cap, name
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", default="1")           # your A3
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--target", type=float, default=0.45) # target luma (0..1)
    ap.add_argument("--max-hi", type=float, default=1.5)  # allowed % highlights
    ap.add_argument("--max-gain", type=float, default=8)  # upper bound for CAP_PROP_GAIN
    args = ap.parse_args()

    # Normalize "video=Name" vs index
    cam_id = int(args.camera_id) if args.camera_id.isdigit() else args.camera_id

    cap, backend = open_cap(cam_id, args.width, args.height, args.fps)
    if not cap: 
        raise SystemExit("Cannot open camera.")

    print("\n== Enabling autos to get a baseline ==")
    # Auto-WB ON (then we may lock)
    setp(cap, cv2.CAP_PROP_AUTO_WB, 1, "auto_white_balance")

    # Exposure: go MANUAL first (values are driver-specific; DSHOW uses negatives)
    # 0.25 = manual (DSHOW), 0.75 = auto; MSMF often uses 0/1.
    ok_autoexp = setp(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure (manual)")

    # Start with a sane exposure for 25 fps: -6 (~1/64s) or -5 (~1/32s)
    # Must be <= 1/25s (40 ms) to avoid motion smear.
    exp = -6
    setp(cap, cv2.CAP_PROP_EXPOSURE, exp, "exposure (log2)")
    setp(cap, cv2.CAP_PROP_GAIN, 0, "gain")

    # Let settings settle
    time.sleep(0.5)

    # Read a few frames to stabilize
    for _ in range(5):
        cap.read()

    print("\n== Auto-dial exposure (and gain if needed) ==")
    # Iteratively adjust exposure/gain to hit target mean without clipping
    best = None
    for _ in range(60):  # ~2–3 seconds of tuning
        ok, frame = cap.read()
        if not ok: 
            time.sleep(0.02); continue
        mean, pct_hi, pct_lo, sharp = get_luma_stats(frame)

        # Save best (closest to target) with minimal highlight clipping
        score = abs(mean - args.target) + (pct_hi > args.max_hi)*10
        if best is None or score < best[0]:
            best = (score, exp, float(cap.get(cv2.CAP_PROP_GAIN)), mean, pct_hi, sharp)

        # Decide adjustment
        if pct_hi > args.max_hi:
            # too many highlights -> reduce exposure
            exp -= 1
            setp(cap, cv2.CAP_PROP_EXPOSURE, exp, "exposure (log2)")
        else:
            if mean < args.target * 0.95:
                # too dark -> try raise exposure (towards -5/-4), else add small gain
                if exp < -4:  # do not exceed ~1/16s at 25fps
                    exp += 1
                    setp(cap, cv2.CAP_PROP_EXPOSURE, exp, "exposure (log2)")
                else:
                    g = min(args.max_gain, cap.get(cv2.CAP_PROP_GAIN) + 1)
                    setp(cap, cv2.CAP_PROP_GAIN, g, "gain")
            elif mean > args.target * 1.05:
                # too bright -> reduce exposure or gain
                cur_gain = cap.get(cv2.CAP_PROP_GAIN)
                if cur_gain > 0:
                    setp(cap, cv2.CAP_PROP_GAIN, max(0, cur_gain - 1), "gain")
                else:
                    exp -= 1
                    setp(cap, cv2.CAP_PROP_EXPOSURE, exp, "exposure (log2)")
            else:
                # within band; stop early
                break

        time.sleep(0.06)

    # Lock white balance if supported (turn auto off but keep current temp)
    print("\n== Lock white balance if possible ==")
    # Many drivers ignore WB temperature, but try:
    wb_temp = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    setp(cap, cv2.CAP_PROP_AUTO_WB, 0, "auto_white_balance off")
    if wb_temp and wb_temp > 0:
        setp(cap, cv2.CAP_PROP_WB_TEMPERATURE, wb_temp, "wb_temperature (lock)")
    else:
        print("wb_temperature not supported; leaving AWB locked without temp.")

    # Report final
    print("\n== Final settings ==")
    print(f"backend:                 {backend}")
    print(f"exposure (log2):         {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"auto_exposure:           {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"gain:                    {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"auto_white_balance:      {cap.get(cv2.CAP_PROP_AUTO_WB)}")
    print(f"wb_temperature:          {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}")

    # Grab and save a test frame for visual check
    ok, frame = cap.read()
    if ok:
        cv2.imwrite("calibration_snapshot.jpg", frame)
        print("Saved: calibration_snapshot.jpg")

    cap.release()
    print("\nTip: reapply these settings before recording (run this script first or copy the setp() calls).")

if __name__ == "__main__":
    main()
