import threading
from llm import LMStudioResponder
from wakeword_detector import WakeWordDetector
import time
from stt import SpeechToTextApplication
from recorder import AudioRecorder
from classify import IntentClassifier
from tts import talk_stream
from blip import BlipModel
from yolov8Objects import locate_objects_in_frame
from ocr import DocTROCR
from camera_manager import CameraManager
from lipreader import LipReader


class VoiceAssistant:
    def __init__(self):
        # Core components
        self.recorder = AudioRecorder()
        self.llm = LMStudioResponder()
        self.classifier = IntentClassifier()
        self.blip = BlipModel()
        self.ocr = DocTROCR()

        # Camera / LipReader targeting Lenovo A3
        A3_CAM = "Lenovo ThinkReality A3 RGB Camera"
        self.camera_manager = CameraManager(camera_id=f"video={A3_CAM}", image_dir="image")
        self.lip_reader = LipReader(camera_id=f"video={A3_CAM}", backend="torchscript", on_text=self.on_lip_text)

        # Wake word
        self.detector = WakeWordDetector(
            wakeword_models=["models\\hey_lucy.onnx"]
        )
        self.detector.register_callback("hey_lucy", self.on_wake_word_detected)

        # Prefer the A3 microphone if present
        try:
            from utils.audio_devices import find_input_device_index
            idx = find_input_device_index()
            if idx is not None and hasattr(self.recorder, "set_microphone"):
                self.recorder.set_microphone(idx)
                print(f"🎙️ Using Lenovo A3 microphone (index {idx})")
            else:
                print("🎙️ A3 mic not found — using system default")
        except Exception as e:
            print(f"⚠️ Mic selection error: {e}")

    def on_lip_text(self, text: str):
        if text:
            try:
                from utils.tts_out import speak_on_device
                speak_on_device(text, device_query=("voice-audio","qualcomm","a3"))
            except Exception:
                talk_stream(text)

    def on_wake_word_detected(self, *args, **kwargs):
        print("🟢 Wake word detected!")

    def run(self):
        print("🚀 Starting assistant...")
        try:
            self.lip_reader.start()
            print("👄 Lip reader started")
        except Exception as e:
            print(f"⚠️ Lip reader not started: {e}")

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
