from time import monotonic, sleep, monotonic_ns
import cv2

class CameraSource:
    """
    Live camera as a frame source.
    cam_index: use your A3 camera index (e.g., 2). Set buffersize small if supported.
    """
    def __init__(self, cam_index=0, target_fps=None):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # not all drivers support this
        except Exception:
            pass
        self.target_dt = (1.0/target_fps) if target_fps else None
        self._last = None

    def read(self):
        if self.target_dt and self._last:
            wait = self.target_dt - (monotonic() - self._last)
            if wait > 0: sleep(wait)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last = monotonic()
        return (monotonic_ns(), frame)

import cv2
from time import monotonic, sleep, monotonic_ns

class FileStreamingSource:
    """
    Play a file as a finite stream of frames.
    - If realtime=True: pace according to timestamps/FPS.
    - If realtime=False: no sleeping, run as fast as possible.
    """
    def __init__(self, path, realtime=True):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.realtime = realtime

        # FPS & frame count for pacing/EOF
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 1e-3:
            self.fps = 25.0
        self.dt = 1.0 / self.fps

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_idx = 0

        self.start_wall = None
        self.start_ms = None
        self.prev_ms = -1.0

    def read(self):
        # Hard stop: if we've already returned all frames, end
        if self.frame_count and self.frame_idx >= self.frame_count:
            return None

        ok, frame = self.cap.read()
        if not ok:
            return None  # EOF or error

        now = monotonic()

        if self.realtime:
            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            use_pos = (
                pos_ms is not None
                and pos_ms > 0.0
                and (self.prev_ms < 0.0 or pos_ms >= self.prev_ms)
            )
            if self.start_wall is None:
                self.start_wall = now
                self.start_ms = pos_ms if use_pos else 0.0
                self.frame_idx = 0

            if use_pos:
                target = self.start_wall + max(0.0, (pos_ms - self.start_ms) / 1000.0)
                self.prev_ms = pos_ms
            else:
                # Fallback: pace by frame index and fps
                target = self.start_wall + self.frame_idx * self.dt

            delay = target - now
            if delay > 0:
                sleep(delay)

        # Bump frame counter after successful read
        self.frame_idx += 1
        return (monotonic_ns(), frame)
        
