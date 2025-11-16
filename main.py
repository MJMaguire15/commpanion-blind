import time
import re
import threading
from pathlib import Path
import subprocess
import sys
import os

from llm import LMStudioResponder
from wakeword_detector import WakeWordDetector
from stt import SpeechToTextApplication
from recorder import AudioRecorder
from classify import IntentClassifier
from tts import talk_stream
from blip import BlipModel
from yolov8Objects import locate_objects_in_frame
from ocr import DocTROCR
from camera_manager import CameraManager
from lipreader import LipReader

# ----------------- CONFIG -----------------
# "pipeline" => run record_10s_and_run.py when user says "start lip reading"
# "stream"   => use your LipReader().start() continuous mode
LIPREAD_MODE = os.environ.get("LIPREAD_MODE", "pipeline").lower()  # "pipeline" or "stream"

# Resolve paths relative to this file (NOT the shell CWD)
HERE = Path(__file__).resolve().parent
CLIP_SCRIPT = (HERE / "record_10s_and_run.py").resolve()

# Optional camera override (else your script’s default --camera-id=1 is used)
CAMERA_ID_DEFAULT = os.environ.get("AUTO_AVSR_CAMERA_ID")  # e.g. "1", "2"
# ------------------------------------------

class VoiceAssistant:
    def __init__(self):
        self.recorder   = AudioRecorder()
        self.llm        = LMStudioResponder()
        self.classifier = IntentClassifier()
        self.blip       = BlipModel()
        self.ocr        = DocTROCR()

        cam_id = 2  # only used for the streaming LipReader path
        self.camera_manager = CameraManager(camera_id=cam_id, image_dir="image")

        self._last_lip_text: str = ""
        self._last_cmd_text: str = ""   # keep last recognized command string
        self._command_lock = threading.Lock()

        self.lip_reader = LipReader(
            camera_id=cam_id,
            backend="torchscript",
            on_text=self.on_lip_text
        )

        # Prefer the A3 mic if available
        mic_index = None
        try:
            from utils.audio_devices import find_input_device_index
            mic_index = find_input_device_index()
            if mic_index is not None and hasattr(self.recorder, "set_microphone"):
                self.recorder.set_microphone(mic_index)
                print(f"🎙️ Using Lenovo A3 microphone (index {mic_index}) for STT/recorder")
            else:
                print("🎙️ A3 mic not found — using system default for STT/recorder")
        except Exception as e:
            print(f"⚠️ Mic selection error: {e}")

        # Make short commands easier to catch
        try:
            if hasattr(self.recorder, "set_silence_settings"):
                self.recorder.set_silence_settings(threshold=400, duration=0.7)
        except Exception:
            pass

        # STT files saved next to main.py
        self.records_dir = HERE / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.stt = SpeechToTextApplication(audio_records_path=self.records_dir)
        try:
            if mic_index is not None and hasattr(self.stt, "set_microphone"):
                self.stt.set_microphone(mic_index)
                print(f"🎙️ STT bound to mic index {mic_index}")
        except Exception:
            pass

        # Wake word
        self.detector = WakeWordDetector(
            wakeword_models=[str(HERE / "models" / "hey_lucy.onnx")],
            inference_framework="onnx",
            threshold=0.25,
            input_device_index=mic_index,
            min_trigger_interval=1.0
        )
        self.detector.register_callback("hey_lucy", self.on_wake_word_detected)

        # Intents
        self._re_start  = re.compile(r"\b(start|begin|go|resume|turn on|enable)\b.*\b(lip|read)\b", re.I)
        self._re_stop   = re.compile(r"\b(stop|halt|cancel|quit|exit|turn off|disable)\b", re.I)
        self._re_repeat = re.compile(r"\b(repeat|again|say (that|it) again|one more time)\b", re.I)
        self._re_status = re.compile(r"\b(status|state|are you (running|on)|is it (running|on))\b", re.I)
        self._re_record = re.compile(r"\b(record|capture|clip|read the tv|read tv)\b", re.I)
        self._re_secs   = re.compile(r"\b(\d{1,3})\s*(sec|secs|second|seconds)\b", re.I)

    # ---------- TTS ----------
    def _speak(self, text: str):
        try:
            from utils.tts_out import speak_on_device
            speak_on_device(text, device_query=("voice-audio", "qualcomm", "a3"))
        except Exception:
            talk_stream(text)

    # ---------- LipReader callback ----------
    def on_lip_text(self, text: str):
        if not text:
            return
        self._last_lip_text = text
        self._speak(text)

    # ---------- Wake word flow ----------
    def on_wake_word_detected(self, *args, **kwargs):
        threading.Thread(target=self._handle_command_session, daemon=True).start()

    def _handle_command_session(self):
        if self._command_lock.locked():
            return
        with self._command_lock:
            print("🟢 Wake word 'Lucy' detected!")
            self._speak("Yes?")

            try:
                print("⏸️ Pausing wake-word detector for command capture…")
                self.detector.pause()
            except Exception as e:
                print(f"ℹ️ Could not pause detector cleanly: {e}")

            try:
                print("🎧 Listening for Lucy command… (e.g. 'start lip reading', 'record a 10 second clip', 'stop', 'status')")
                text = self._capture_command_then_transcribe(max_seconds=9.0).strip()
                self._last_cmd_text = text
                if not text:
                    print("🙈 No text recognized.")
                    self._speak("Never mind.")
                else:
                    print(f"🗣️ Command heard: {text}")
                    low = self._normalize(text)

                    if self._re_start.search(low) or re.fullmatch(r"(start|begin|go|resume)", low):
                        self._intent_start_lipreading()
                    elif self._re_record.search(low):
                        seconds = self._extract_seconds(low, default=10)
                        self._intent_record_clip(seconds=seconds)
                    elif self._re_stop.search(low):
                        self._intent_stop_lipreading()
                    elif self._re_repeat.search(low):
                        self._intent_repeat()
                    elif self._re_status.search(low):
                        self._intent_status()
                    else:
                        self._speak("Sorry, I didn't catch that.")
            finally:
                try:
                    print("▶️ Resuming wake-word detector…")
                    self.detector.resume()
                except Exception as e:
                    print(f"⚠️ Could not restart detector: {e}")

    # ---------- STT capture ----------
    def _capture_command_then_transcribe(self, max_seconds: float) -> str:
        started = False
        try:
            started = self.recorder.start_recording()
        except Exception as e:
            print(f"⚠️ recorder.start_recording() failed: {e}")

        if not started:
            print("⚠️ Could not start recording (no mic configured?)")
            return ""

        t0 = time.time()
        while getattr(self.recorder, "is_recording", False) and (time.time() - t0) < max_seconds:
            time.sleep(0.05)

        if getattr(self.recorder, "is_recording", False):
            self.recorder.stop_recording()

        wav_path = self.records_dir / f"lucy_cmd_{int(time.time())}.wav"
        ok = False
        try:
            ok = self.recorder.save_recording(str(wav_path))
        except Exception as e:
            print(f"⚠️ save_recording() failed: {e}")

        if not ok:
            print("⚠️ Failed to save command WAV")
            return ""

        print(f"💾 Captured command WAV: {wav_path}")

        try:
            text = self.stt.transcribe()
            if isinstance(text, dict):
                text = text.get("text", "")
            return text if isinstance(text, str) else str(text)
        except Exception as e:
            print(f"⚠️ STT.transcribe() failed: {e}")
            return ""

    # ---------- helpers ----------
    def _normalize(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _extract_seconds(self, text: str, default: int = 10) -> int:
        m = self._re_secs.search(text)
        if m:
            try:
                n = int(m.group(1))
                return max(1, min(120, n))
            except Exception:
                pass
        words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"fifteen":15,"twenty":20,"thirty":30}
        for w, n in words.items():
            if f"{w} second" in text or f"{w}-second" in text:
                return n
        return default

    # ---------- intents ----------
    def _intent_start_lipreading(self):
        # If in pipeline mode, treat "start lip reading" as "record a clip"
        if LIPREAD_MODE == "pipeline":
            seconds = self._extract_seconds(self._last_cmd_text or "", default=10)
            print(f"🔁 LIPREAD_MODE=pipeline → triggering record_10s_and_run.py for {seconds}s")
            self._intent_record_clip(seconds=seconds)
            return

        # Otherwise, run your streaming LipReader like before
        try:
            if hasattr(self.lip_reader, "is_running") and getattr(self.lip_reader, "is_running"):
                self._speak("Lip reading is already running.")
                return
            self.lip_reader.start()
            self._speak("Starting lip reading now.")
            print("✅ lip_reader.start() called")
        except Exception as e:
            print(f"⚠️ Lip reader start error: {e}")
            self._speak("I couldn't start lip reading.")

    def _intent_stop_lipreading(self):
        try:
            if hasattr(self.lip_reader, "stop"):
                self.lip_reader.stop()
                self._speak("Stopped lip reading.")
                print("✅ lip_reader.stop() called")
            else:
                self._speak("Stopping is not supported.")
        except Exception as e:
            print(f"⚠️ Lip reader stop error: {e}")
            self._speak("I couldn't stop lip reading.")

    def _intent_repeat(self):
        if self._last_lip_text:
            self._speak(self._last_lip_text)
        else:
            self._speak("I have nothing to repeat yet.")

    def _intent_status(self):
        running = getattr(self.lip_reader, "is_running", None)
        if running is True:
            self._speak("Lip reader is running.")
        elif running is False:
            self._speak("Lip reader is stopped.")
        else:
            self._speak("I cannot determine the lip reader state.")

    # ---------- run the clip pipeline (LIVE output; correct CWD; UTF-8 env) ----------
    def _intent_record_clip(self, seconds: int = 10):
        if not CLIP_SCRIPT.exists():
            self._speak("The clip recorder script is missing.")
            print(f"❌ Not found: {CLIP_SCRIPT}")
            return

        # Stop streaming lip_reader to avoid camera contention (harmless if not running)
        try:
            if hasattr(self.lip_reader, "stop"):
                self.lip_reader.stop()
        except Exception:
            pass

        # Pause wake-word while clip runs
        paused = False
        try:
            self.detector.pause()
            paused = True
        except Exception:
            pass

        self._speak(f"Recording a {seconds} second clip.")

        py_exe = HERE / ".venv" / "Scripts" / "python.exe"
        py = str(py_exe) if py_exe.exists() else sys.executable
        cmd = [py, "-u", str(CLIP_SCRIPT), "--seconds", str(seconds), "--no-playback"]
        if CAMERA_ID_DEFAULT:
            cmd += ["--camera-id", str(CAMERA_ID_DEFAULT)]

        print(f"▶️ LAUNCHING CLIP SCRIPT AT: {CLIP_SCRIPT}")
        print("▶️ CWD:", HERE)
        print("▶️ CMD:", " ".join(cmd))

        # ✅ Force UTF-8 so the child can print emojis without crashing
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        collected_stdout_lines = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            def _drain(pipe, is_err=False):
                for line in iter(pipe.readline, ''):
                    line = line.rstrip("\n")
                    if is_err:
                        print(f"[clip][err] {line}")
                    else:
                        print(f"[clip] {line}")
                        collected_stdout_lines.append(line)
                pipe.close()

            t_out = threading.Thread(target=_drain, args=(proc.stdout, False), daemon=True)
            t_err = threading.Thread(target=_drain, args=(proc.stderr, True), daemon=True)
            t_out.start(); t_err.start()

            ret = proc.wait()
            t_out.join(timeout=1.0)
            t_err.join(timeout=1.0)

            if ret != 0:
                self._speak("The clip recorder failed.")
                print(f"❌ Script returned {ret}")
            else:
                out = "\n".join(collected_stdout_lines)
                text = self._pick_transcript(out)
                self._speak(text or "Clip finished.")
        except Exception as e:
            print(f"⚠️ Error running clip script: {e}")
            self._speak("I couldn't complete the clip.")
        finally:
            if paused:
                try:
                    print("▶️ Resuming wake-word detector…")
                    self.detector.resume()
                except Exception as e:
                    print(f"⚠️ Could not restart detector: {e}")

    def _pick_transcript(self, stdout: str) -> str:
        L = [ln.rstrip() for ln in (stdout or "").splitlines() if ln.strip()]
        for ln in reversed(L):
            if "Transcript:" in ln:
                return ln.split("Transcript:", 1)[-1].strip()
        decode_re = re.compile(r"\[DECODE\]\s*(.+)$")
        for ln in reversed(L):
            m = decode_re.search(ln)
            if m:
                return " ".join(m.group(1).split())
        for ln in reversed(L):
            m = re.search(r"(?:^|\s)(?:Prediction:|Hypo:)\s*(.+)$", ln, re.IGNORECASE)
            if m:
                return " ".join(m.group(1).split())
        return ""

    # ---------- lifecycle ----------
    def run(self):
        print("🚀 Starting assistant...")
        print(f"🔧 LIPREAD_MODE = {LIPREAD_MODE}  (pipeline → record_10s_and_run.py on 'start lip reading')")
        try:
            self.detector.start()
            print("🟢 Wake-word detector running — say 'hey lucy'")
        except Exception as e:
            print(f"⚠️ Wake-word detector not started: {e}")

        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n🛑 Exiting...")
            try: self.detector.stop()
            except Exception: pass
            try:
                if hasattr(self.lip_reader, "stop"):
                    self.lip_reader.stop()
            except Exception:
                pass


if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
