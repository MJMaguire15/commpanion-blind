# stream_cpu_lowlatency.py
WINDOW_SECONDS = 1.0
STEP_SECONDS   = 0.4
FPS = 25
ROI_SIZE = 88

WINDOW_FRAMES = int(WINDOW_SECONDS * FPS)
STEP_FRAMES   = int(STEP_SECONDS * FPS)

engine = load_model(CKPT)

buffer = deque(maxlen=WINDOW_FRAMES)
frames_since_decode = 0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    roi = extract_mouth_roi(frame)
    buffer.append(roi)
    frames_since_decode += 1

    if len(buffer) == WINDOW_FRAMES and frames_since_decode >= STEP_FRAMES:
        frames_since_decode = 0

        clip = np.stack(buffer, axis=0)
        clip = clip[:, :, :, None]
        clip = torch.from_numpy(clip).to(dtype=torch.float32).div_(255.0)

        t0 = time.time()
        text = run(engine, clip)
        dt = (time.time() - t0) * 1000

        if text:
            print(f"[{dt:6.1f} ms] {text}")
