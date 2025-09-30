import cv2
for i in [0,1,2]:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    ret, frame = (False, None)
    if ok:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        ret, frame = cap.read()
    print(f"index {i} -> open={ok}, frame={frame.shape if (ret and frame is not None) else None}")
    cap.release()
