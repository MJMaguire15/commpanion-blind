import numpy as np
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import download_models
import threading
import queue
import os
import time
from typing import Callable, Optional, Dict, Any
import logging
from pathlib import Path


class WakeWordDetector:
    """Modular wake word detector using openWakeWord."""

    def __init__(
        self,
        wakeword_models: list[str] = None,
        inference_framework: str = 'onnx',
        threshold: float = 0.25,
        chunk_size: int = 1280,
        sample_rate: int = 16000,
        channels: int = 1,
        input_device_index: Optional[int] = None,
        min_trigger_interval: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        self.wakeword_models = wakeword_models or ['hey_jarvis']
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_device_index = input_device_index
        self.min_trigger_interval = min_trigger_interval
        self.logger = logger or logging.getLogger(__name__)

        # -------------------------------------------------
        # FIX: Explicitly point openWakeWord to local ONNX files
        # -------------------------------------------------
        HERE = Path(__file__).resolve().parent
        OWW_MODELS = HERE / "models" / "openwakeword"

        if not OWW_MODELS.exists():
            raise FileNotFoundError(
                f"openWakeWord models directory not found: {OWW_MODELS}"
            )

        self.model = Model(
            wakeword_models=self.wakeword_models,
            inference_framework=inference_framework,

            # REQUIRED because PyPI wheel is missing resources/
            melspec_model_path=str(OWW_MODELS / "melspectrogram.onnx"),
            embedding_model_path=str(OWW_MODELS / "embedding_model.onnx"),
        )

        # -------------------------------------------------
        # Audio setup
        # -------------------------------------------------
        self.audio_format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.stream = None

        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.callbacks: Dict[str, Callable] = {}

        self._last_trigger_ts = 0.0
        self.processing_thread: Optional[threading.Thread] = None

    def register_callback(self, wakeword: str, callback: Callable[[str, float], Any]):
        self.callbacks[wakeword] = callback
        self.logger.info(f"Callback registered for '{wakeword}'")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def _process_audio(self):
        self.logger.info("Starting audio processing")
        while self.is_listening:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                predictions = self.model.predict(audio_array)

                now = time.time()
                for wakeword, score in predictions.items():
                    if (
                        score > self.threshold
                        and (now - self._last_trigger_ts) >= self.min_trigger_interval
                    ):
                        self._last_trigger_ts = now
                        self.logger.info(
                            f"Wake word detected: {wakeword} (score: {score:.2f})"
                        )
                        if wakeword in self.callbacks:
                            try:
                                self.callbacks[wakeword](wakeword, score)
                            except Exception as e:
                                self.logger.error(f"Error in callback: {e}")
                        self.model.reset()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error during audio processing: {e}")

    @classmethod
    def download_models(cls):
        if not os.path.exists("resources/models"):
            print("Downloading and installing openWakeWord")
            download_models()
            print("✅ openWakeWord installed successfully\n")

    def start(self):
        print("Starting wake word detector")
        if self.is_listening:
            self.logger.warning("Detector is already running")
            return

        self.logger.info("Starting wake word detector")
        self.is_listening = True

        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        self.processing_thread = threading.Thread(
            target=self._process_audio, daemon=True, name="oWW-processor"
        )
        self.processing_thread.start()

        self.logger.info("Detector started and listening")

    def _close_stream(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            finally:
                self.stream = None

    def stop(self):
        """Stop listening and tear down safely."""
        if not self.is_listening and self.stream is None:
            return

        self.logger.info("Stopping detector")
        self.is_listening = False

        if self.processing_thread and threading.current_thread() is not self.processing_thread:
            try:
                self.processing_thread.join()
            except Exception:
                pass

        self._close_stream()

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break

        self.logger.info("Detector stopped")

    # Friendly wrappers
    def pause(self):
        self.stop()

    def resume(self):
        self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def cleanup(self):
        self.stop()
