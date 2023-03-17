import sounddevice as sd
import numpy as np

def record_audio(recording_size, sampling_frequency, recordings, stop_event):
    print("Audio recording started")
    while not stop_event.is_set():
        recordings.append(sd.rec(frames = recording_size,
                                   samplerate = sampling_frequency,
                                   channels = 1,
                                   blocking = True,
                                   dtype = np.float32))
        if stop_event.is_set():
            sd.stop()
    print("Recording stopped")
