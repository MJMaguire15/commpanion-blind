# import pyaudio
# import wave
# import time
# import os
# import threading
# from pathlib import Path
# from sttDeaf import SpeechToTextApplication

# # Config
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK_DURATION_SEC = 1  # Smaller = lower latency
# FRAMES_PER_CHUNK = int(RATE / CHUNK * CHUNK_DURATION_SEC)
# AUDIO_DIR = Path("live_chunks")
# AUDIO_DIR.mkdir(exist_ok=True)

# # Init PyAudio and STT
# p = pyaudio.PyAudio()
# stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
# stt_app = SpeechToTextApplication()

# NEON_GREEN = "\033[92m"
# RESET_COLOR = "\033[0m"

# # Lock for concurrency
# transcribe_lock = threading.Lock()


# def record_and_transcribe_loop():
#     try:
#         counter = 0
#         print("Live transcription started. Press Ctrl+C to stop.")
#         while True:
#             frames = []
#             for _ in range(FRAMES_PER_CHUNK):
#                 data = stream.read(CHUNK, exception_on_overflow=False)
#                 frames.append(data)

#             # Save chunk to file
#             chunk_filename = AUDIO_DIR / f"chunk_{counter}.wav"
#             with wave.open(str(chunk_filename), 'wb') as wf:
#                 wf.setnchannels(CHANNELS)
#                 wf.setsampwidth(p.get_sample_size(FORMAT))
#                 wf.setframerate(RATE)
#                 wf.writeframes(b''.join(frames))

#             # Transcribe in a separate thread
#             threading.Thread(
#                 target=transcribe_chunk,
#                 args=(chunk_filename,),
#                 daemon=True
#             ).start()

#             counter += 1

#     except KeyboardInterrupt:
#         print("\n Stopped by user.")
#     finally:
#         stream.stop_stream()
#         stream.close()
#         p.terminate()

# def transcribe_chunk(chunk_path: Path):
#     # The lock is no longer strictly necessary with this change, but it doesn't hurt.
#     with transcribe_lock:
#         try:
#             # Pass the specific path to the transcribe method
#             transcription = stt_app.transcribe(chunk_path)
#             if transcription.strip(): # Only print if there's actual text
#                 print(NEON_GREEN + transcription + RESET_COLOR)
#         except Exception as e:
#             print(f" Transcription error: {e}")
# """
# def transcribe_chunk(chunk_path: Path):
#     with transcribe_lock:
#         try:
#             transcription = stt_app.transcribe()
#             print(NEON_GREEN + transcription + RESET_COLOR)
#         except Exception as e:
#             print(f" Transcription error: {e}")
# """

# if __name__ == "__main__":
#     record_and_transcribe_loop()

import pyaudio
import wave
import os
import threading
from pathlib import Path
from sttDeaf import SpeechToTextApplication # Your modified stt.py


last_transcription = ""
TRANSCRIPT_PATH = Path("live_transcript.txt")


# --- Config ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
AUDIO_DIR = Path("live_chunks")
AUDIO_DIR.mkdir(exist_ok=True)

# --- Overlap/Buffer Config ---
# We'll keep 5 seconds of audio in memory
BUFFER_DURATION_SEC = 5
# We'll add new audio and run transcription every 1 second
RECORD_INTERVAL_SEC = 1

FRAMES_PER_INTERVAL = int(RATE / CHUNK * RECORD_INTERVAL_SEC)
BUFFER_MAX_FRAMES = int(RATE / CHUNK * BUFFER_DURATION_SEC)

# --- Init ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
# Initialize the MODIFIED SpeechToTextApplication
stt_app = SpeechToTextApplication() 

NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def record_and_transcribe_loop():
    try:
        counter = 0
        audio_buffer = []
        print("Live transcription started. Press Ctrl+C to stop.")
        
        while True:
            # Record for the interval duration
            for _ in range(FRAMES_PER_INTERVAL):
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_buffer.append(data)

            # Keep the buffer from growing indefinitely
            if len(audio_buffer) > BUFFER_MAX_FRAMES:
                audio_buffer = audio_buffer[-BUFFER_MAX_FRAMES:]

            # Save the entire current buffer to a file
            chunk_filename = AUDIO_DIR / f"chunk_{counter}.wav"
            with wave.open(str(chunk_filename), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_buffer))

            # Transcribe the buffer file in a separate thread
            threading.Thread(
                target=transcribe_chunk,
                args=(chunk_filename,),
                daemon=True
            ).start()

            counter += 1

    except KeyboardInterrupt:
        print("\n Stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def transcribe_chunk(chunk_path: Path):
    global last_transcription  # So we can update the global copy

    try:
        transcription = stt_app.transcribe(chunk_path).strip()

        if not transcription:
            return

        # De-duplication logic
        if last_transcription and transcription.startswith(last_transcription):
            new_part = transcription[len(last_transcription):].strip()
        else:
            new_part = transcription  # If no match, print all

        if new_part:
            #print(NEON_GREEN + new_part + RESET_COLOR, flush=True)
            import sys

            # Clear line and write updated caption
            sys.stdout.write('\r' + NEON_GREEN + transcription + RESET_COLOR)
            sys.stdout.flush()


            with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
                f.write(new_part + " ")
        last_transcription = transcription  # Update memory

    except Exception as e:
        print(f" Transcription error: {e}")

"""
# This function is the same as in the Step 1 fix
def transcribe_chunk(chunk_path: Path):
    try:
        transcription = stt_app.transcribe(chunk_path)
        if transcription.strip():
            # You will get repeated phrases; you may want to handle this in your application logic
            # For now, we'll just print it.
            print(NEON_GREEN + transcription + RESET_COLOR, flush=True)
    except Exception as e:
        print(f" Transcription error: {e}")
"""

if __name__ == "__main__":
    record_and_transcribe_loop()