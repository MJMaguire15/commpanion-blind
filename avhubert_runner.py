# avhubert_runner.py
"""
Subprocess adapter for LipReader(backend="avhubert").

Reuses your existing Auto-AVSR pipeline:
  - Writes snippet window [T,H,W] to an mp4 (mouth-only)
  - Writes labels/list.csv (single-line)
  - Calls auto_avsr/eval.py with your checkpoint
  - Parses stdout and returns the decoded text
  - Logs full stdout/stderr to .tmp_avsr_runner/logs/

Optional env vars:
  AUTO_AVSR_CKPT    -> path to your .pth checkpoint
  AUTO_AVSR_ROOT    -> where temp roi_root lives (default: ./.tmp_avsr_runner)
  AUTO_AVSR_PY      -> Python to call eval.py (default: .venv/Scripts/python.exe or "python")
  AUTO_AVSR_DATASET -> dataset folder name (default: "custom")
  AUTO_AVSR_CLIP    -> clip name (default: "clip_0001.mp4")
"""

import os, re, cv2, subprocess, numpy as np
from pathlib import Path

DEFAULT_CKPT = r".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"
DEFAULT_ROOT = Path(".") / ".tmp_avsr_runner"
DEFAULT_DATASET = os.getenv("AUTO_AVSR_DATASET", "custom")
DEFAULT_CLIP = os.getenv("AUTO_AVSR_CLIP", "clip_0001.mp4")
FPS = 25
OUT_SIZE = 88

_initialized = False
_root = None
_labels_csv = None
_py = None
_ckpt = None
_clip_path = None
_log_dir = None

def _which_python():
    venv = Path(".") / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return str(venv)
    env_py = os.getenv("AUTO_AVSR_PY")
    if env_py:
        return env_py
    return "python"

def _init_once():
    global _initialized, _root, _labels_csv, _py, _ckpt, _clip_path, _log_dir
    if _initialized:
        return
    _py = _which_python()
    _ckpt = os.getenv("AUTO_AVSR_CKPT", DEFAULT_CKPT)
    _root = Path(os.getenv("AUTO_AVSR_ROOT", DEFAULT_ROOT)).resolve()
    (_root / "labels").mkdir(parents=True, exist_ok=True)
    (_root / DEFAULT_DATASET).mkdir(parents=True, exist_ok=True)
    _labels_csv = _root / "labels" / "list.csv"
    _clip_path = _root / DEFAULT_DATASET / DEFAULT_CLIP
    _log_dir = _root / "logs"; _log_dir.mkdir(parents=True, exist_ok=True)
    if not Path("auto_avsr").exists():
        raise RuntimeError("auto_avsr folder not found beside this script.")
    if not Path(_ckpt).exists():
        raise RuntimeError(f"Checkpoint not found: {_ckpt}")
    _initialized = True

def _write_mp4_from_window(window: np.ndarray, dst_mp4: Path):
    T, H, W = window.shape
    if window.dtype != np.uint8:
        x = ((window + 1.0) * 127.5).astype(np.float32)
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = window
    if (H, W) != (OUT_SIZE, OUT_SIZE):
        x = np.stack([cv2.resize(f, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA) for f in x], axis=0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(dst_mp4), fourcc, FPS, (OUT_SIZE, OUT_SIZE))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {dst_mp4}")
    for i in range(x.shape[0]):
        f = x[i]
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        vw.write(f)
    vw.release()

def _write_list_csv(num_frames: int):
    line = f"{DEFAULT_DATASET},{DEFAULT_CLIP},{num_frames},-1\n"
    _labels_csv.write_text(line, encoding="utf-8")

def _parse_eval_stdout(stdout: str) -> str:
    if not stdout:
        return ""
    L = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(L):
        if ln.startswith("[DECODE]"):
            return " ".join(ln.replace("[DECODE]", "", 1).strip().split())
    for ln in reversed(L):
        low = ln.lower()
        if low.startswith("prediction:") or low.startswith("hypo:"):
            return " ".join(ln.split(":", 1)[-1].split())
    bad = {"wer", "cer", "testing", "dataloader", "metric"}
    for ln in reversed(L):
        low = ln.lower()
        if any(tok in low for tok in bad):
            continue
        if re.search(r"[A-Za-z]", ln) and not re.fullmatch(r"[-=~_]+", ln):
            return " ".join(ln.split())
    return ""

def decode(window: np.ndarray) -> str:
    _init_once()
    try:
        _write_mp4_from_window(window, _clip_path)
        _write_list_csv(num_frames=window.shape[0])
    except Exception as e:
        print(f"[avhubert_runner] Input prep failed: {e}")
        return ""
    eval_py = Path("auto_avsr") / "eval.py"
    if not eval_py.exists():
        print("[avhubert_runner] auto_avsr/eval.py not found.")
        return ""
    args = [
        _py, str(eval_py),
        "--modality", "video",
        "--root-dir", str(_root),
        "--test-file", str(_labels_csv.name),
        "--pretrained-model-path", str(_ckpt),
    ]
    try:
        proc = subprocess.run(
            args, cwd=str(Path(".").resolve()),
            capture_output=True, text=True, timeout=180
        )
    except Exception as e:
        print(f"[avhubert_runner] eval.py invocation failed: {e}")
        return ""
    (_log_dir / "last_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (_log_dir / "last_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        print(f"[avhubert_runner] eval.py error:\n{proc.stderr}")
        return ""
    return _parse_eval_stdout(proc.stdout)
