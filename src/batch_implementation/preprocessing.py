import os
import scipy.io.wavfile as wavfile
import scipy.signal as signal

class Preprocessor:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size

    def lp_filter(self):
        # parameters need adjustment
        filter_order = 80
        passband_freq = 6000
        sampling_freq = 48000
        max_passband_ripple = 0.00057565
        max_stopband_attenuation = 1e-4

        nyquist_freq = sampling_freq / 2
        normalized_passband_freq = passband_freq / nyquist_freq
        filter_taps = signal.remez(filter_order+1, [0, normalized_passband_freq, 1], [1, 0], [1/max_passband_ripple, max_stopband_attenuation])

    def process(self, data_path):
        for file in os.listdir(data_path):
            current_file = wavfile.read(os.path.join(data_path, file))[1]
