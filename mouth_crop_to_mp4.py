# mouth_crop_to_mp4.py
import os, argparse, math, cv2
#from mediapipe.python.solutions.face_mesh import FaceMesh, FACEMESH_LIPS

def square_expand(x0,y0,x1,y1,scale,W,H):
    cx=(x0+x1)/2; cy=(y0+y1)/2
    s=max((x1-x0)*scale,(y1-y0)*scale)
    X0=int(max(0, cx - s/2)); Y0=int(max(0, cy - s/2))
    X1=int(min(W, cx + s/2)); Y1=int(min(H, cy + s/2))
    return X0,Y0,X1,Y1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--input",required=True,help="source video (e.g. input.mp4)")
    ap.add_argument("-o","--outmp4",required=True,help="lips-only mp4 (e.g. roi_root/custom/clip_0001.mp4)")
    ap.add_argument("--size",type=int,default=88,help="ROI size (default 88)")
    ap.add_argument("--scale",type=float,default=1.8,help="bbox scale (default 1.8)")
    ap.add_argument("--fps",type=float,default=0,help="override fps (0 = use input fps)")
    # optional: auto-write list.csv
    ap.add_argument("--root-dir",default="",help="roi root (e.g. roi_root)")
    ap.add_argument("--dataset",default="custom",help="dataset name (default custom)")
    ap.add_argument("--clip",default="clip_0001.mp4",help="relative clip name inside dataset")
    ap.add_argument("--write-list",action="store_true",help="write labels/list.csv for Auto-AVSR")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.outmp4), exist_ok=True)
    cap=cv2.VideoCapture(args.input, cv2.CAP_ANY)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.input}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps and args.fps>0 else (in_fps if in_fps and in_fps>1 else 25)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    vw=cv2.VideoWriter(args.outmp4, fourcc, fps, (args.size,args.size))

    fm=FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,
                min_detection_confidence=0.5,min_tracking_confidence=0.5)
    lip_idx = sorted({i for a,b in FACEMESH_LIPS for i in (a,b)})

    n=0
    while True:
        ok, frame=cap.read()
        if not ok: break
        H,W=frame.shape[:2]
        res=fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark
            xs=[lm[i].x*W for i in lip_idx]; ys=[lm[i].y*H for i in lip_idx]
            X0,Y0,X1,Y1=square_expand(int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys)),args.scale,W,H)
            crop=frame[Y0:Y1, X0:X1]
            if crop.size: roi=cv2.resize(crop,(args.size,args.size),cv2.INTER_AREA)
            else: roi=cv2.resize(frame,(args.size,args.size))
        else:
            roi=cv2.resize(frame,(args.size,args.size))
        vw.write(roi); n+=1

    cap.release(); vw.release()
    print(f"Wrote {args.outmp4}  frames={n}  fps={fps}")

    # optional: write roi_root/labels/list.csv
    if args.write_list:
        if not args.root_dir:
            raise SystemExit("--write-list needs --root-dir (e.g. roi_root)")
        labels_dir = os.path.join(args.root_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        line = f"{args.dataset},{os.path.basename(args.clip)},{n},-1"
        with open(os.path.join(labels_dir,"list.csv"), "w", encoding="utf-8") as f:
            f.write(line+"\n")
        print(f"Wrote {os.path.join(labels_dir,'list.csv')} -> {line}")

if __name__ == "__main__":
    main()
