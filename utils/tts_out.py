import tempfile, os, wave, pyaudio, pyttsx3

def _find_output_index(query=("qualcomm","a3","voice-audio","lenovo")):
    q = tuple(k.lower() for k in query)
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if info.get("maxOutputChannels", 0) > 0 and any(k in name for k in q):
                return i
    finally:
        p.terminate()
    return None

def speak_on_device(text: str, device_query=("qualcomm","voice-audio","a3")):
    engine = pyttsx3.init()
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    engine.save_to_file(text, tmp_wav)
    engine.runAndWait()

    p = pyaudio.PyAudio()
    try:
        dev_idx = _find_output_index(device_query)
        wf = wave.open(tmp_wav, 'rb')
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        output_device_index=dev_idx)
        data = wf.readframes(4096)
        while data:
            stream.write(data)
            data = wf.readframes(4096)
        stream.stop_stream(); stream.close(); wf.close()
    finally:
        p.terminate()
        try: os.unlink(tmp_wav)
        except: pass
