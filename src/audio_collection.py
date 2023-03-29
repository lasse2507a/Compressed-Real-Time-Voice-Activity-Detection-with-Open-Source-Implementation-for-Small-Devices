import sounddevice as sd
import numpy as np

def record_audio(samplerate, blocksize, recordings, thread_stop_event):
    print("Audio recording started")
    input_stream = sd.InputStream(samplerate=samplerate, blocksize=blocksize, channels=1, dtype=np.float32)
    input_stream.start()
    while not thread_stop_event.is_set():
        recordings.put(np.reshape(input_stream.read(frames=blocksize)[0], 256))
    if thread_stop_event.is_set():
        input_stream.abort()
    print("Recording stopped")
