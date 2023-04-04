import numpy as np

class SineWaveGenerator:
    def __init__(self, N, fs):
        self.N = N
        self.fs = fs

    def time(self):
        return np.linspace(0, self.N/self.fs, self.N)

    def frequency(self):
        return np.linspace(0, self.fs/2, int(self.N/2))

    def single_sine_wave(self, freq=100, phase=0):
        """
        Signal consisting of a single sine wave with specified frequency and phase.
        """
        t = np.arange(self.N)/self.fs
        amp = 2 * np.pi * t
        return np.sin(amp * freq + phase)

    def five_sine_wave(self, freq1=1000, freq2=2000, freq3=3000, freq4=4000,
                       phase1=0, phase2=0, phase3=0, phase4=0):
        """
        Signal consisting of four sine waves with specified frequencies, phases, and amount of points.
        """
        t = np.arange(self.N)/self.fs
        amp = 2 * np.pi * t
        x1 = np.sin(amp * freq1 + phase1)
        x2 = np.sin(amp * freq2 + phase2)
        x3 = np.sin(amp * freq3 + phase3)
        x4 = np.sin(amp * freq4 + phase4)
        return x1 + x2 + x3 + x4
