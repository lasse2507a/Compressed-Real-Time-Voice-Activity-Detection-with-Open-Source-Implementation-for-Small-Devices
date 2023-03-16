import sounddevice as sd
import numpy as np

def record_audio(recording_size, sampling_frequency, recording_queue):
    print("Audio recordning started")
    recording_queue.put(sd.rec(recording_size,
                       samplerate = sampling_frequency,
                       channels = 1,
                       blocking = False,
                       dtype = np.float32))
