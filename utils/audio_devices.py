import pyaudio

def find_input_device_index(keywords=("lenovo","thinkreality","a3","qualcomm")) -> int | None:
    p = pyaudio.PyAudio()
    try:
        kws = tuple(k.lower() for k in keywords)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if info.get("maxInputChannels", 0) > 0 and any(k in name for k in kws):
                return i
    finally:
        p.terminate()
    return None
