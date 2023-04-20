import os
import scipy.io.wavfile as wavfile

class Preprocessor:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size

    def process(self, data_path):
        for file in os.listdir(data_path):
            current_file = wavfile.read(os.path.join(data_path, file))[1]
