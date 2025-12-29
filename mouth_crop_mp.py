# mouth_crop_mp.py
import os, argparse, cv2, math
#from mediapipe.python.solutions.face_mesh import FaceMesh, FACEMESH_LIPS

def square_expand(x0, y0, x1, y1, scale, W, H):
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w  = (x1 - x0) * scale
    h  = (y1 - y0) * scale
    s  = max(w, h)
    X0 = int(max(0, cx - s/2)); Y0 = int(max(0, cy - s/2))
    X1 = int(min(W, cx + s/2));  Y1 = int(min(H, cy + s/2))
    return X0, Y0, X1, Y1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input mp4")
    ap.add_argument("-o", "--outdir", required=True, help="output folder for ROI frames")
    ap.add_argument("--size", type=int, default=88, help="ROI size (default 88x88)")
    ap.add_argument("--scale", type=float, default=1.8, help="bbox scale around lips")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    lips_pairs = mp_face_mesh.FACEMESH_LIPS
    lip_idx = sorted({i for a,b in lips_pairs for i in (a,b)})

    cap = cv2.VideoCapture(args.input, cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")

    fm = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            xs = [lm[i].x * W for i in lip_idx]
            ys = [lm[i].y * H for i in lip_idx]
            x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
            x1, y1 = min(W, int(max(xs))),  min(H, int(max(ys)))
            X0, Y0, X1, Y1 = square_expand(x0, y0, x1, y1, args.scale, W, H)
            crop = frame[Y0:Y1, X0:X1]
            if crop.size == 0:
                roi = cv2.resize(frame, (args.size, args.size))
            else:
                roi = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
        else:
            roi = cv2.resize(frame, (args.size, args.size))

        cv2.imwrite(os.path.join(args.outdir, f"{idx:06d}.png"), roi)
        idx += 1

    cap.release()
    print(f"Saved {idx-1} frames to {args.outdir}")

if __name__ == "__main__":
    main()
