import pathlib
import cv2
import time
import queue
import threading
import numpy as np
from typing import Callable, Optional, Deque, List
from collections import deque

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None  # runtime check will warn

def ctc_greedy_decode(logits: np.ndarray, labels: str) -> str:
    if logits.ndim == 3:
        logits = logits[0]
    pred = logits.argmax(axis=-1)
    blank = 0
    out = []
    prev = None
    for p in pred:
        if p != blank and p != prev:
            out.append(labels[p])
        prev = p
    return "".join(out)

class MouthCropper:
    def __init__(self, roi_size: int = 112):
        if mp is None:
            raise RuntimeError("mediapipe is not installed. Please install mediapipe to use lip reading.")
        self.roi_size = roi_size
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mouth_indices = list(set(list(range(61, 88)) + [0]))

    def get_mouth_roi(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        xs, ys = [], []
        for idx in self.mouth_indices:
            pt = lm.landmark[idx]
            xs.append(int(pt.x * w))
            ys.append(int(pt.y * h))
        x1, x2 = max(min(xs) - 16, 0), min(max(xs) + 16, w)
        y1, y2 = max(min(ys) - 16, 0), min(max(ys) + 16, h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.roi_size, self.roi_size), interpolation=cv2.INTER_AREA)
        norm = (resized.astype(np.float32) / 127.5) - 1.0
        return norm

class LipReader:
    """
    Streams frames from a camera, extracts a mouth ROI sequence, runs a lipreading model,
    and calls back with decoded text in (near) real-time.
    """
    def __init__(
        self,
        camera_id: int = 0,
        backend: str = "torchscript",
        window_frames: int = 29,
        stride: int = 5,
        model_path: Optional[str] = "models/lipreader.pt",
        labels_path: Optional[str] = "models/lipreader.labels",
        on_text: Optional[Callable[[str], None]] = None,
    ):
        self.camera_id = camera_id
        self.window_frames = window_frames
        self.stride = stride
        self.on_text = on_text
        self.model_path = model_path
        self.labels_path = labels_path
        self.backend = backend

        self.cap = None
        self.cropper = None

        self.model = None
        self.labels = " abcdefghijklmnopqrstuvwxyz'"

        self._thread = None
        self._running = threading.Event()
        self._buffer = deque(maxlen=window_frames)

    def _load_model(self):
        global torch
        if self.backend == "torchscript":
            if torch is None:
                raise RuntimeError("PyTorch is not installed. Install torch/torchvision to run the lipreading model.")
            if self.model_path and pathlib.Path(self.model_path).exists():
                try:
                    self.model = torch.jit.load(self.model_path, map_location="cpu").eval()
                except Exception as e:
                    print(f"[LipReader] Failed to load model at {self.model_path}: {e}")
                    self.model = None
            else:
                print(f"[LipReader] No model found at {self.model_path}. Running in 'passthrough' mode.")
                self.model = None

        if self.labels_path and pathlib.Path(self.labels_path).exists():
            self.labels = pathlib.Path(self.labels_path).read_text().strip()
            if not self.labels.startswith(" "):
                self.labels = " " + self.labels

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def _run(self):
        self._load_model()
        self.cropper = MouthCropper()

        # robust open by MSMF name or index
        if isinstance(self.camera_id, str):
            name = self.camera_id if self.camera_id.lower().startswith('video=') else f'video={self.camera_id}'
            self.cap = cv2.VideoCapture(name, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                try: self.cap.release()
                except Exception: pass
                self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                try: self.cap.release()
                except Exception: pass
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"[LipReader] Cannot open camera {self.camera_id}")
            return

        last_infer_frame_idx = -self.window_frames
        frame_idx = 0

        while self._running.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            roi = self.cropper.get_mouth_roi(frame)
            if roi is None:
                frame_idx += 1
                continue

            self._buffer.append(roi)

            if len(self._buffer) == self.window_frames and (frame_idx - last_infer_frame_idx) >= self.stride:
                window = np.stack(list(self._buffer), axis=0)  # [T, H, W]
                text = self._infer(window)
                last_infer_frame_idx = frame_idx
                if text and self.on_text:
                    self.on_text(text)

            frame_idx += 1

    def _infer(self, window: np.ndarray) -> str:
        """
        window: [T, H, W] normalized frames in [-1, 1]
        Returns decoded text or empty string.
        """
        if getattr(self, "backend", "torchscript") == "avhubert":
            try:
                from avhubert_runner import decode as avhubert_decode
            except Exception as e:
                print(f"[LipReader] AV-HuBERT backend unavailable: {e}")
                return ""
            try:
                return avhubert_decode(window)
            except Exception as e:
                print(f"[LipReader] AV-HuBERT decode error: {e}")
                return ""

        if self.model is None:
            return ""  # No-op without a model

        try:
            import torch
            import torch.nn.functional as F
        except Exception as e:
            print(f"[LipReader] Torch backend unavailable: {e}")
            return ""

        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0).unsqueeze(2)  # [1, T, 1, H, W]
            x = x.float()
            try:
                logits = self.model(x)  # expected to return [B, T, C]
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
                text = ctc_greedy_decode(probs, self.labels)
                text = " ".join(text.split())
                return text
            except Exception as e:
                print(f"[LipReader] Inference error: {e}")
                return ""
