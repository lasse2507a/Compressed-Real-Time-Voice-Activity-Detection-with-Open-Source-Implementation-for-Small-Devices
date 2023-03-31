import numpy as np

class SineWaveGenerator:
    def __init__(self, J=12, fs=2**11):
        self.J = J
        self.fs = fs

    def single_sine_wave(self, freq=100, phase=0):
        """
        Signal consisting of a single sine wave with specified frequency and phase.
        """
        N = 2**self.J
        t = np.arange(N)/self.fs
        A = 2 * np.pi * t
        x = np.sin(A * freq + phase)
        return x

    def five_sine_wave(self, blocksize, freq1=1000, freq2=2000, freq3=3000, freq4=4000,
                       phase1=0, phase2=0, phase3=0, phase4=0):
        """
        Signal consisting of four sine waves with specified frequencies, phases, and amount of points.
        """
        N = blocksize
        t = np.arange(N)/self.fs
        A = 2 * np.pi * t
        x1 = np.sin(A * freq1 + phase1)
        x2 = np.sin(A * freq2 + phase2)
        x3 = np.sin(A * freq3 + phase3)
        x4 = np.sin(A * freq4 + phase4)
        x_sum = x1 + x2 + x3 + x4
        return x_sum
