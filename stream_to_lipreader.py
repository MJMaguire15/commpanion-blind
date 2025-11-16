# stream_to_lipreader.py
from lipreader import LipReader

def on_text_callback(text: str):
    """Called whenever the model produces decoded text."""
    if text:
        print(f"🗣️ {text}")

def main():
    # A3 usually enumerates as camera index 0; if not, try the device name string.
    # e.g., camera_id="ThinkReality A3"
    reader = LipReader(
        camera_id=1,
        backend="avhubert",     # <-- use your avhubert_runner.decode()
        window_frames=40,       # ~1.6 s at 25 fps
        stride=20,              # new decode every ~0.8 s
        on_text=on_text_callback,
    )

    try:
        print("🎥 Starting lip reader — Ctrl+C to stop.")
        reader.start()
        while True:
            pass
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        reader.stop()

if __name__ == "__main__":
    main()
