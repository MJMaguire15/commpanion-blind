"""
clip_farm_and_run.py

Modes
-----

1) One-shot clip test (default)
   - Records N seconds from camera.
   - Runs mouth_crop_to_mp4 + auto_avsr/eval.py once.
   - Optionally computes WER/CER vs full reference (SRT if given, otherwise .txt).
   - Logs to: sessions/<ts>/logs/metrics_clip.csv

   Example:
       python clip_farm_and_run.py --camera-id 1 --seconds 10

2) Continuous chunked stream (sliding window)
   - Continuously captures frames from camera.
   - Every step_seconds, dumps the last window_seconds into an MP4.
   - Runs mouth_crop_to_mp4 + auto_avsr/eval.py for each chunk.
   - If --srt is supplied, uses SRT timestamps to compute WER/CER per chunk.
   - Builds a de-overlapped full transcript across all chunks.
   - Saves: sessions/<ts>/full_predicted_transcript_continuous.txt
   - Logs per-chunk metrics to: sessions/<ts>/logs/metrics_chunks.csv

   Example:
       python clip_farm_and_run.py --camera-id 1 --continuous \
           --window-seconds 5.0 --step-seconds 2.5 \
           --srt "Top 10 Small Talk Questions and Answers [dEZkN0_6R1c].en_cleaned.srt"

3) Offline benchmarking on a saved video
   - Takes a full video file (e.g. .webm from YouTube) and an SRT.
   - Splits into non-overlapping windows of length window_seconds.
   - Runs mouth_crop_to_mp4 + auto_avsr/eval.py for each window.
   - Uses SRT timestamps to get reference text and compute WER/CER per chunk.
   - Concatenates all chunk hypotheses into a full transcript (no overlap).
   - Saves: sessions/<ts>/full_predicted_transcript_benchmark.txt
   - Logs per-chunk metrics to: sessions/<ts>/logs/metrics_benchmark.csv
   - Also prints global WER/CER vs the full SRT text if available.

   Example:
       python clip_farm_and_run.py \
           --benchmark-video "Top 10 Small Talk Questions and Answers [dEZkN0_6R1c].webm" \
           --srt "Top 10 Small Talk Questions and Answers [dEZkN0_6R1c].en_cleaned.srt" \
           --window-seconds 5.0 --no-preview --no-playback --no-speak
"""

import os
import sys
import time
import cv2
import argparse
import subprocess
import datetime
import re
import csv
from pathlib import Path

# --- defaults ---
DEFAULT_FPS = 25
DEFAULT_W, DEFAULT_H = 1280, 720
OUT_ROOT = Path("sessions")
CKPT_DEFAULT = r".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"
REF_DEFAULT = Path("benchmarking/refs/vanessa_clear_confident.norm.txt")

# ---- TTS (your existing tts._TTS) ----
_TTS_ENGINE = None
def speak_now(text: str, rate: int = 200):
    """Speak using your existing pyttsx3-based TTS (Windows default output)."""
    global _TTS_ENGINE
    text = (text or "").strip()
    if not text:
        return
    if _TTS_ENGINE is None:
        from tts import _TTS  # your original class
        _TTS_ENGINE = _TTS(rate=rate)
    _TTS_ENGINE.start(text)


# ========= Metrics helpers =========

def _norm_text(s: str) -> str:
    """Simple normalisation for WER/CER: lowercase, strip punctuation, collapse spaces."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _levenshtein(a, b):
    """Levenshtein distance on sequences a, b (list of tokens or chars)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def compute_global_wer_cer(ref_text: str, hyp_text: str):
    """Compute WER and CER for entire reference vs hypothesis."""
    ref_norm = _norm_text(ref_text)
    hyp_norm = _norm_text(hyp_text)

    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()
    word_dist = _levenshtein(ref_tokens, hyp_tokens)
    wer = word_dist / max(1, len(ref_tokens))

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    char_dist = _levenshtein(ref_chars, hyp_chars)
    cer = char_dist / max(1, len(ref_chars))

    return wer, cer


class TranscriptAligner:
    """
    Greedy, sequential aligner for chunked decoding using a plain-text reference.

    - Loads a full reference transcript.
    - For each chunk hypothesis, finds the best-matching segment near the current position.
    - Advances the position so segments don't overlap.
    """

    def __init__(self, ref_text: str):
        self.ref_norm = _norm_text(ref_text)
        self.ref_tokens = self.ref_norm.split()
        self.pos = 0  # current token index in ref_tokens

    def match_chunk(self, hyp_text: str):
        """
        Given a chunk hypothesis, return:
            (ref_segment_text, wer, cer)
        If no reference remains or hyp is empty, returns ("", None, None).
        """
        hyp_norm = _norm_text(hyp_text)
        hyp_tokens = hyp_norm.split()
        if not hyp_tokens or self.pos >= len(self.ref_tokens):
            return "", None, None

        L = len(hyp_tokens)
        remaining = len(self.ref_tokens) - self.pos
        if remaining <= 0:
            return "", None, None

        min_len = max(1, int(L * 0.5))
        max_len = max(min(remaining, int(L * 1.5) + 4), min_len)

        best_dist = None
        best_len = None
        best_tokens = None

        for seg_len in range(min_len, max_len + 1):
            seg_tokens = self.ref_tokens[self.pos:self.pos + seg_len]
            dist = _levenshtein(hyp_tokens, seg_tokens)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_len = seg_len
                best_tokens = seg_tokens

        if best_tokens is None or best_len is None:
            return "", None, None

        self.pos += best_len
        ref_seg_text = " ".join(best_tokens)
        wer = best_dist / max(1, len(best_tokens))

        ref_chars = list("".join(best_tokens))
        hyp_chars = list("".join(hyp_tokens))
        char_dist = _levenshtein(ref_chars, hyp_chars)
        cer = char_dist / max(1, len(ref_chars))

        return ref_seg_text, wer, cer


# ========= SRT helpers =========

def _parse_srt_time(t: str) -> float:
    """Convert 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' into seconds as float."""
    t = t.replace('.', ',')
    hms, ms = t.split(',')
    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s) + int(ms) / 1000.0


def load_srt(path: Path):
    """
    Parse an SRT file into a list of (start_s, end_s, text).
    Multi-line subtitle text is joined with spaces.
    """
    entries = []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip('\n') for ln in raw.splitlines()]

    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue

        # Optional index line
        if lines[i].strip().isdigit():
            i += 1

        if i >= n:
            break

        timing = lines[i].strip()
        i += 1
        if '-->' not in timing:
            continue

        start_str, end_str = [p.strip() for p in timing.split('-->')]
        start_s = _parse_srt_time(start_str)
        end_s   = _parse_srt_time(end_str)

        texts = []
        while i < n and lines[i].strip():
            texts.append(lines[i].strip())
            i += 1

        # Join and lightly normalise spaces
        text = " ".join(texts)
        if text:
            entries.append((start_s, end_s, text))

    return entries


def srt_text_for_window(entries, t0: float, t1: float) -> str:
    """
    Return concatenated subtitle text overlapping [t0, t1], with some
    light deduplication and a boundary fallback.

    Overlap condition: entry_end > t0 and entry_start < t1.
    If nothing matches, we optionally pick entries close to boundaries.
    """
    out_segments = []

    for start_s, end_s, text in entries:
        if end_s <= t0:
            continue
        if start_s >= t1:
            break
        text_norm = re.sub(r"\s+", " ", text).strip()
        if text_norm and text_norm not in out_segments:
            out_segments.append(text_norm)

    if not out_segments:
        # Fallback: pick entries very close to window edges
        for start_s, end_s, text in entries:
            if abs(start_s - t0) < 1.0 or abs(end_s - t1) < 1.0:
                text_norm = re.sub(r"\s+", " ", text).strip()
                if text_norm and text_norm not in out_segments:
                    out_segments.append(text_norm)

    return " ".join(out_segments)


# ========= Core video / AVSR helpers =========

def open_capture(cam_id, width, height, fps):
    """Open by MSMF then DSHOW, supports int index or 'video=Device Name'."""
    def _mk(arg, backend):
        cap = cv2.VideoCapture(arg, backend)
        return cap if cap.isOpened() else None

    if isinstance(cam_id, str) and cam_id.isdigit():
        cam_id = int(cam_id)

    if isinstance(cam_id, str):
        name = cam_id if cam_id.lower().startswith("video=") else f"video={cam_id}"
        cap = _mk(name, cv2.CAP_MSMF) or _mk(name, cv2.CAP_DSHOW)
    else:
        cap = _mk(cam_id, cv2.CAP_MSMF) or _mk(cam_id, cv2.CAP_DSHOW)

    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


def countdown_and_record(cap, out_mp4, seconds, width, height, fps, preview=True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    print("Get ready…")
    for t in (3, 2, 1):
        print(t, flush=True)
        time.sleep(1)
    print("Recording…", flush=True)

    t0 = time.time()
    next_tick = t0 + 1
    last_sec_printed = 0
    preview_used = False

    while (time.time() - t0) < seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        vw.write(frame)

        if preview:
            try:
                cv2.imshow("Preview (recording)", frame)
                preview_used = True
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except cv2.error:
                preview = False

        now = time.time()
        if now >= next_tick:
            last_sec_printed += 1
            print(last_sec_printed, flush=True)
            next_tick += 1

        time.sleep(max(0, (1.0 / fps) - (time.time() - now)))

    vw.release()
    if preview_used:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"Saved raw video: {out_mp4}")


def run_mouth_crop(raw_mp4, lips_mp4, fps, scale=2.2, size=88):
    """Call mouth_crop_to_mp4.py to produce lips-only mp4 + labels/list.csv."""
    crop_py = Path("mouth_crop_to_mp4.py")
    if not crop_py.exists():
        raise FileNotFoundError("mouth_crop_to_mp4.py not found in current folder.")
    args = [
        sys.executable, str(crop_py),
        "-i", str(raw_mp4),
        "-o", str(lips_mp4),
        "--size", str(size),
        "--scale", str(scale),
        "--fps", str(fps),
        "--root-dir", str(lips_mp4.parent.parent),
        "--dataset", "custom",
        "--clip", lips_mp4.name,
        "--write-list",
    ]
    print("Creating lips-only clip…")
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("mouth_crop_to_mp4 failed.")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def inspect_video(path):
    """Return (ok, frames, fps, w, h) without showing UI."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, 0.0, 0, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return True, frames, fps, w, h


def run_auto_avsr(roi_root, list_csv, ckpt_path, log_dir):
    """Call auto_avsr/eval.py once and return transcript + write logs."""
    eval_py = Path("auto_avsr") / "eval.py"
    if not eval_py.exists():
        raise FileNotFoundError("auto_avsr/eval.py not found beside this script.")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    py_exe = Path(".") / ".venv" / "Scripts" / "python.exe"
    py = str(py_exe) if py_exe.exists() else sys.executable

    args = [
        py, str(eval_py),
        "--modality", "video",
        "--root-dir", str(roi_root),
        "--test-file", str(list_csv.name),
        "--pretrained-model-path", str(ckpt_path),
    ]
    print("Running Auto-AVSR decode…")
    proc = subprocess.run(args, cwd=str(Path(".").resolve()),
                          capture_output=True, text=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (log_dir / "eval_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

    lines = (proc.stdout or "").splitlines()
    tail  = "\n".join(lines[-20:])
    if tail.strip():
        print("\n--- eval.py last 20 lines ---")
        print(tail)
        print("--- end ---\n")

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("auto_avsr/eval.py failed.")

    def pick_transcript(stdout: str) -> str:
        L = [ln.rstrip() for ln in (stdout or "").splitlines() if ln.strip()]
        decode_re = re.compile(r"\[DECODE\]\s*(.+)$")
        for ln in reversed(L):
            m = decode_re.search(ln)
            if m:
                return " ".join(m.group(1).split())
        for ln in reversed(L):
            m = re.search(r"(?:^|\s)(?:Prediction:|Hypo:)\s*(.+)$", ln, re.IGNORECASE)
            if m:
                return " ".join(m.group(1).split())
        skip_tokens = {"wer", "cer", "testing", "dataloader", "metric", "pretrained weights"}
        for ln in reversed(L):
            low = ln.lower()
            if any(tok in low for tok in skip_tokens):
                continue
            if re.search(r"[A-Za-z]", ln) and not re.fullmatch(r"[-=~_]+", ln.strip()):
                return " ".join(ln.split())
        return ""

    return pick_transcript(proc.stdout)


def playback(mp4_path, title, fps):
    """Play mp4 via OpenCV if possible, else via system default player."""
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open with OpenCV")
        print(f"Playing: {mp4_path}  (press ESC to close)")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(title, frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    except Exception:
        print(f"OpenCV playback not available — opening {mp4_path} with the system player.")
        p = str(mp4_path)
        if os.name == "nt":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])


def write_buffer_to_mp4(frames, out_mp4, fps):
    """Write a list of BGR frames to an MP4 file."""
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"Saved window clip: {out_mp4}")


# Helper for continuous transcript concat (avoid overlap duplication)
def _norm_for_concat(s: str) -> str:
    """Normalize text for overlap-based concatenation (keep spaces, strip punctuation)."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ========= Continuous streaming mode =========

def stream_and_decode(cap,
                      args,
                      session_dir: Path,
                      roi_root: Path,
                      logs_dir: Path,
                      ref_aligner: TranscriptAligner | None,
                      metrics_csv: Path | None,
                      srt_entries=None):
    """
    Continuous loop:
      - capture frames into a sliding window buffer
      - every step_seconds, dump last window_seconds to mp4
      - run mouth_crop + auto_avsr
      - speak transcript (optional)
      - log per-chunk WER/CER + latency
      - build a de-overlapped full transcript over chunks
    """
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    lips_mp4 = roi_root / "custom" / "clip_0001.mp4"
    list_csv = roi_root / "labels" / "list.csv"

    fps = args.fps
    window_frames = max(1, int(args.window_seconds * fps))
    step_frames   = max(1, int(args.step_seconds * fps))
    if step_frames > window_frames:
        step_frames = window_frames

    print(f"Streaming with window={args.window_seconds:.2f}s "
          f"({window_frames} frames), step={args.step_seconds:.2f}s "
          f"({step_frames} frames).")
    print("Press ESC in the preview window or Ctrl+C in the console to stop.\n")

    frame_buffer = []
    frames_since_decode = 0
    chunk_idx = 1
    preview = not args.no_preview
    frame_count = 0

    # transcript building
    all_chunk_hyps = []
    continuous_transcript = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            frame_count += 1
            frame_buffer.append(frame)
            if len(frame_buffer) > window_frames:
                frame_buffer.pop(0)

            if preview:
                try:
                    cv2.imshow("Preview (stream)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        print("ESC pressed — stopping stream.")
                        break
                except cv2.error:
                    preview = False

            frames_since_decode += 1

            if len(frame_buffer) >= window_frames and frames_since_decode >= step_frames:
                frames_since_decode = 0
                raw_mp4 = raw_dir / f"raw_window_{chunk_idx:04d}.mp4"
                log_dir = logs_dir / f"chunk_{chunk_idx:04d}"
                this_chunk = chunk_idx
                chunk_idx += 1

                frames_to_write = list(frame_buffer)

                chunk_end_time = frame_count / fps
                chunk_start_time = max(0.0, chunk_end_time - args.window_seconds)

                print(f"\n=== Chunk {this_chunk} "
                      f"({chunk_start_time:.2f}s–{chunk_end_time:.2f}s) ===")
                write_buffer_to_mp4(frames_to_write, raw_mp4, fps)

                run_mouth_crop(raw_mp4, lips_mp4, fps=fps,
                               scale=args.scale, size=args.size)

                ok_roi, frames_roi, vfps, vw, vh = inspect_video(lips_mp4)
                with (logs_dir / "roi_video_info.txt").open("w", encoding="utf-8") as f:
                    f.write(f"path={lips_mp4}\nframes={frames_roi}\n"
                            f"fps={vfps}\nsize={vw}x{vh}\n")
                print(f"ROI video: frames={frames_roi}, fps={vfps:.2f}, size={vw}x{vh}")

                t0 = time.time()
                text = run_auto_avsr(roi_root, list_csv, args.ckpt, log_dir)
                t1 = time.time()
                latency_ms = (t1 - t0) * 1000.0

                print(f"\n🗣️ [Chunk {this_chunk}] Transcript:",
                      text if text else "(empty)")
                print(f"⏱️  [Chunk {this_chunk}] Latency: {latency_ms:.1f} ms")

                # Store raw chunk hyp
                all_chunk_hyps.append(text or "")

                # Build continuous transcript with overlap handling
                if text:
                    new_norm = _norm_for_concat(text)
                    if not continuous_transcript:
                        continuous_transcript.append(new_norm)
                    else:
                        prev = continuous_transcript[-1]
                        max_overlap_len = 0
                        max_k = min(len(prev), len(new_norm))
                        for k in range(1, max_k + 1):
                            if prev.endswith(new_norm[:k]):
                                max_overlap_len = k
                        non_overlap = new_norm[max_overlap_len:].lstrip()
                        if non_overlap:
                            continuous_transcript.append(non_overlap)

                # WER/CER
                ref_seg = ""
                wer_val = None
                cer_val = None

                if text:
                    if srt_entries:
                        ref_seg = srt_text_for_window(srt_entries,
                                                      chunk_start_time,
                                                      chunk_end_time)
                        if ref_seg.strip():
                            wer_val, cer_val = compute_global_wer_cer(ref_seg, text)
                            print(f"📊 [Chunk {this_chunk}] WER={wer_val:.3f}, CER={cer_val:.3f}")
                        else:
                            print(f"📊 [Chunk {this_chunk}] No SRT text for this window.")
                    elif ref_aligner is not None:
                        ref_seg, wer_val, cer_val = ref_aligner.match_chunk(text)
                        if wer_val is not None:
                            print(f"📊 [Chunk {this_chunk}] WER={wer_val:.3f}, CER={cer_val:.3f}")
                        else:
                            print(f"📊 [Chunk {this_chunk}] No reference segment available.")
                    else:
                        print(f"📊 [Chunk {this_chunk}] No reference for WER/CER.")

                if text and not args.no_speak:
                    try:
                        speak_now(text, rate=args.speak_rate)
                    except Exception as e:
                        print(f"⚠️ TTS error: {e}")

                if metrics_csv is not None:
                    is_new = not metrics_csv.exists()
                    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if is_new:
                            writer.writerow([
                                "mode", "session_ts", "chunk_idx",
                                "chunk_start_s", "chunk_end_s",
                                "window_s", "step_s",
                                "latency_ms",
                                "hyp_text", "ref_text",
                                "wer", "cer",
                            ])
                        writer.writerow([
                            "continuous",
                            session_dir.name,
                            this_chunk,
                            f"{chunk_start_time:.3f}",
                            f"{chunk_end_time:.3f}",
                            f"{args.window_seconds:.3f}",
                            f"{args.step_seconds:.3f}",
                            f"{latency_ms:.3f}",
                            text or "",
                            ref_seg or "",
                            "" if wer_val is None else f"{wer_val:.6f}",
                            "" if cer_val is None else f"{cer_val:.6f}",
                        ])

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — stopping stream.")

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # Save continuous full transcript
    full_transcript = " ".join(continuous_transcript).strip()
    out_path = session_dir / "full_predicted_transcript_continuous.txt"
    out_path.write_text(full_transcript, encoding="utf-8")
    print(f"\n📄 Saved continuous full predicted transcript to:\n  {out_path}\n")


# ========= Offline benchmarking mode (video + SRT) =========

def benchmark_video_with_srt(video_path: Path,
                             args,
                             session_dir: Path,
                             roi_root: Path,
                             logs_dir: Path,
                             srt_entries,
                             srt_full_text: str | None):
    """
    Offline benchmarking:
      - Reads the given video file.
      - Splits it into non-overlapping windows of args.window_seconds.
      - For each window:
          * writes a raw MP4 chunk
          * runs mouth_crop_to_mp4
          * runs auto_avsr/eval.py
          * gets SRT text for that time range
          * computes WER/CER
      - Logs per-chunk metrics to metrics_benchmark.csv
      - Concatenates all hyps into full_predicted_transcript_benchmark.txt
      - Computes global WER/CER vs the full SRT text if available.
    """
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    lips_mp4 = roi_root / "custom" / "clip_0001.mp4"
    list_csv = roi_root / "labels" / "list.csv"
    metrics_csv = logs_dir / "metrics_benchmark.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video file: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
    if vid_fps <= 0:
        vid_fps = float(args.fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / vid_fps if vid_fps > 0 else 0.0
    print(f"Benchmarking video: {video_path}")
    print(f"  fps={vid_fps:.3f}, frames={total_frames}, duration={duration_s:.2f}s")

    window_frames = max(1, int(args.window_seconds * vid_fps))
    # non-overlapping windows
    step_frames = window_frames

    frame_idx = 0
    chunk_idx = 1
    all_hyp_texts = []

    is_new = not metrics_csv.exists()
    with metrics_csv.open("a", newline="", encoding="utf-8") as f_metrics:
        writer = csv.writer(f_metrics)
        if is_new:
            writer.writerow([
                "mode", "session_ts", "chunk_idx",
                "chunk_start_s", "chunk_end_s",
                "window_s",
                "latency_ms",
                "hyp_text", "ref_text",
                "wer", "cer",
            ])

        while True:
            frames = []
            start_frame_idx = frame_idx

            for _ in range(window_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                frame_idx += 1

            if not frames:
                break

            chunk_start_s = start_frame_idx / vid_fps
            chunk_end_s = frame_idx / vid_fps

            raw_mp4 = raw_dir / f"raw_window_{chunk_idx:04d}.mp4"
            log_dir = logs_dir / f"chunk_{chunk_idx:04d}"
            this_chunk = chunk_idx
            chunk_idx += 1

            print(f"\n=== Benchmark Chunk {this_chunk} "
                  f"({chunk_start_s:.2f}s–{chunk_end_s:.2f}s) ===")
            write_buffer_to_mp4(frames, raw_mp4, vid_fps)

            run_mouth_crop(raw_mp4, lips_mp4, fps=vid_fps,
                           scale=args.scale, size=args.size)

            ok_roi, frames_roi, vfps, vw, vh = inspect_video(lips_mp4)
            with (logs_dir / "roi_video_info.txt").open("w", encoding="utf-8") as f_info:
                f_info.write(f"path={lips_mp4}\nframes={frames_roi}\n"
                             f"fps={vfps}\nsize={vw}x{vh}\n")
            print(f"ROI video: frames={frames_roi}, fps={vfps:.2f}, size={vw}x{vh}")

            t0 = time.time()
            hyp_text = run_auto_avsr(roi_root, list_csv, args.ckpt, log_dir)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            print(f"\n🗣️ [Chunk {this_chunk}] Transcript:",
                  hyp_text if hyp_text else "(empty)")
            print(f"⏱️  [Chunk {this_chunk}] Latency: {latency_ms:.1f} ms")

            all_hyp_texts.append(hyp_text or "")

            ref_seg = ""
            wer_val = None
            cer_val = None
            if srt_entries and hyp_text:
                ref_seg = srt_text_for_window(srt_entries,
                                              chunk_start_s, chunk_end_s)
                if ref_seg.strip():
                    wer_val, cer_val = compute_global_wer_cer(ref_seg, hyp_text)
                    print(f"📊 [Chunk {this_chunk}] WER={wer_val:.3f}, CER={cer_val:.3f}")
                else:
                    print(f"📊 [Chunk {this_chunk}] No SRT text for this window.")
            else:
                print(f"📊 [Chunk {this_chunk}] No SRT entries or empty hyp; WER/CER skipped.")

            writer.writerow([
                "benchmark",
                session_dir.name,
                this_chunk,
                f"{chunk_start_s:.3f}",
                f"{chunk_end_s:.3f}",
                f"{args.window_seconds:.3f}",
                f"{latency_ms:.3f}",
                hyp_text or "",
                ref_seg or "",
                "" if wer_val is None else f"{wer_val:.6f}",
                "" if cer_val is None else f"{cer_val:.6f}",
            ])

    cap.release()

    # Save full predicted transcript for benchmark (simple concatenation)
    # (non-overlapping windows so no overlap logic needed)
    full_pred = " ".join(_norm_for_concat(t) for t in all_hyp_texts if t).strip()
    full_pred_path = session_dir / "full_predicted_transcript_benchmark.txt"
    full_pred_path.write_text(full_pred, encoding="utf-8")
    print(f"\n📄 Saved benchmark full predicted transcript to:\n  {full_pred_path}\n")

    # Global WER/CER for the whole video if we have full SRT text
    if srt_full_text and full_pred:
        global_wer, global_cer = compute_global_wer_cer(srt_full_text, full_pred)
        print("\n===== Global Benchmark Metrics (vs SRT) =====")
        print(f"Global WER={global_wer:.3f}, CER={global_cer:.3f}")
        print("=============================================\n")
    else:
        print("\nNo global WER/CER computed (missing SRT or no hypotheses).\n")


# ========= main =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", default="1",
        help='Camera index (e.g. 0/1/2) or device name (e.g. "video=ThinkReality A3")')
    ap.add_argument("--seconds", type=int, default=10,
        help="Duration in seconds for one-shot mode.")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--ckpt", default=os.environ.get("AUTO_AVSR_CKPT", CKPT_DEFAULT))
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-playback", action="store_true")
    ap.add_argument("--scale", type=float, default=2.2,
                    help="mouth bbox expansion (try 2.2–2.6)")
    ap.add_argument("--size", type=int, default=88,
                    help="mouth crop size (e.g., 88 or 112)")
    ap.add_argument("--no-speak", action="store_true",
                    help="do not speak transcript automatically")
    ap.add_argument("--speak-rate", type=int, default=200,
                    help="pyttsx3 words-per-minute")

    ap.add_argument("--continuous", action="store_true",
                    help="Enable continuous streaming + sliding-window decode.")
    ap.add_argument("--window-seconds", type=float, default=5.0,
                    help="Window length (seconds).")
    ap.add_argument("--step-seconds", type=float, default=2.5,
                    help="Step (seconds) between decodes in continuous mode.")

    ap.add_argument("--ref", type=str, default=str(REF_DEFAULT),
                    help="Path to reference transcript .txt for WER/CER (fallback if no SRT).")
    ap.add_argument("--srt", type=str, default="",
                    help="Path to .srt subtitle file for timestamp-aligned WER/CER.")
    ap.add_argument("--benchmark-video", type=str, default="",
                    help="Path to a full video file for offline benchmarking with SRT.")

    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUT_ROOT / ts
    raw_dir = session_dir / "raw"
    roi_root = session_dir / "roi_root"
    logs_dir = session_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (roi_root / "labels").mkdir(parents=True, exist_ok=True)
    (roi_root / "custom").mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load SRT if provided
    srt_entries = None
    srt_full_text = None
    if args.srt:
        srt_path = Path(args.srt)
        if srt_path.exists():
            print(f"Using SRT subtitles: {srt_path}")
            srt_entries = load_srt(srt_path)
            srt_full_text = " ".join(text for _, _, text in srt_entries)
        else:
            print(f"⚠️ SRT file not found at {srt_path}, SRT-based WER/CER will be skipped.")

    # Load plain-text reference if present and no SRT
    ref_path = Path(args.ref) if args.ref else None
    ref_text_full = None
    ref_aligner = None
    if ref_path is not None and ref_path.exists():
        print(f"Using reference transcript: {ref_path}")
        ref_text_full = ref_path.read_text(encoding="utf-8", errors="ignore")
        if not srt_entries:
            ref_aligner = TranscriptAligner(ref_text_full)
    else:
        if ref_path is not None and not ref_path.exists():
            print(f"⚠️ Reference transcript not found at {ref_path}; "
                  f"txt-based WER/CER will be skipped.")

    # Offline benchmarking mode (no camera)
    if args.benchmark_video:
        video_path = Path(args.benchmark_video)
        if not video_path.exists():
            raise SystemExit(f"benchmark_video file not found: {video_path}")
        benchmark_video_with_srt(video_path, args,
                                 session_dir, roi_root, logs_dir,
                                 srt_entries, srt_full_text)
        return

    # Camera-based modes
    cap = open_capture(args.camera_id, args.width, args.height, args.fps)
    if not cap or not cap.isOpened():
        raise SystemExit(f"Cannot open camera: {args.camera_id}")

    if args.continuous:
        metrics_chunks_csv = logs_dir / "metrics_chunks.csv"
        stream_and_decode(cap, args, session_dir, roi_root, logs_dir,
                          ref_aligner, metrics_chunks_csv, srt_entries)
        cap.release()
        return

    # One-shot mode
    raw_mp4 = raw_dir / f"raw_{ts}.mp4"
    lips_mp4 = roi_root / "custom" / "clip_0001.mp4"
    list_csv = roi_root / "labels" / "list.csv"

    countdown_and_record(cap, raw_mp4, seconds=args.seconds,
                         width=args.width, height=args.height, fps=args.fps,
                         preview=not args.no_preview)
    cap.release()

    run_mouth_crop(raw_mp4, lips_mp4, fps=args.fps, scale=args.scale, size=args.size)

    ok, frames, vfps, vw, vh = inspect_video(lips_mp4)
    with (logs_dir / "roi_video_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"path={lips_mp4}\nframes={frames}\nfps={vfps}\nsize={vw}x{vh}\n")
    print(f"ROI video: frames={frames}, fps={vfps:.2f}, size={vw}x{vh}")

    if not args.no_playback:
        playback(raw_mp4, f"Raw ({args.seconds}s)", fps=args.fps)
        playback(lips_mp4, f"Lips-only ({args.seconds}s)", fps=args.fps)

    t0 = time.time()
    text = run_auto_avsr(roi_root, list_csv, args.ckpt, logs_dir)
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0

    print("\n🗣️ Transcript:", text if text else "(empty)")
    print(f"⏱️ Latency: {latency_ms:.1f} ms")

    if text and not args.no_speak:
        try:
            speak_now(text, rate=args.speak_rate)
        except Exception as e:
            print(f"⚠️ TTS error: {e}")

    # Whole-clip WER/CER (prefer SRT full text if available)
    wer_val = None
    cer_val = None
    full_ref = srt_full_text or ref_text_full
    if full_ref and text:
        wer_val, cer_val = compute_global_wer_cer(full_ref, text)
        print(f"📊 Clip WER={wer_val:.3f}, CER={cer_val:.3f}")

    metrics_clip_csv = logs_dir / "metrics_clip.csv"
    is_new = not metrics_clip_csv.exists()
    with metrics_clip_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "session_ts", "seconds",
                "latency_ms",
                "hyp_text",
                "wer", "cer",
            ])
        writer.writerow([
            session_dir.name,
            args.seconds,
            f"{latency_ms:.3f}",
            text or "",
            "" if wer_val is None else f"{wer_val:.6f}",
            "" if cer_val is None else f"{cer_val:.6f}",
        ])


if __name__ == "__main__":
    main()
