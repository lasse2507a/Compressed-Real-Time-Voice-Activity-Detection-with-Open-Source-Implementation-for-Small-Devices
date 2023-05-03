import numpy as np
import matplotlib.pyplot as plt


class SineWaveGenerator:
    def __init__(self, N, fs):
        """
        Initialize a SineWaveGenerator instance.
        Args: N (int): Number of points in the generated signal.
              fs (float): Sampling frequency of the generated signal.
        """
        self.N = N
        self.fs = fs


    def time(self):
        '''
        Return: (numpy.ndarray): Array of time values.
        '''
        return np.linspace(0, self.N/self.fs, self.N)


    def frequency(self):
        '''
        Return: (numpy.ndarray): Array of frequency values.
        '''
        return np.linspace(0, self.fs/2, int(self.N/2))


    def single_sine_wave(self, freq=100, phase=0):
        """
        Generate a signal consisting of a single sine wave with specified frequency and phase.
        Args: freq (float): Frequency of the sine wave (default is 100).
              phase (float): Phase of the sine wave (default is 0).
        Return: (numpy.ndarray): Array of signal values.
        """
        t = np.arange(self.N)/self.fs
        amp = 2 * np.pi * t
        return np.sin(amp * freq + phase)


    def four_sine_waves(self, freq1=1000, freq2=2000, freq3=3000, freq4=4000, phase1=0, phase2=0, phase3=0, phase4=0):
        """
        Generate a signal consisting of four sine waves with specified frequencies and phases.
        Args: freq1 (float): Frequency of the first sine wave (default is 1000).
              freq2 (float): Frequency of the second sine wave (default is 2000).
              freq3 (float): Frequency of the third sine wave (default is 3000).
              freq4 (float): Frequency of the fourth sine wave (default is 4000).
              phase1 (float): Phase of the first sine wave (default is 0).
              phase2 (float): Phase of the second sine wave (default is 0).
              phase3 (float): Phase of the third sine wave (default is 0).
              phase4 (float): Phase of the fourth sine wave (default is 0).
        Return: (numpy.ndarray): Array of signal values.
        """
        t = np.arange(self.N)/self.fs
        amp = 2 * np.pi * t
        x1 = np.sin(amp * freq1 + phase1)
        x2 = np.sin(amp * freq2 + phase2)
        x3 = np.sin(amp * freq3 + phase3)
        x4 = np.sin(amp * freq4 + phase4)
        return x1 + x2 + x3 + x4


if __name__ == "__main__":
    samplerate = 16000
    size = 17200

    time = SineWaveGenerator(N=size, fs=samplerate).time()
    freqs = SineWaveGenerator(N=size, fs=samplerate).frequency()
    signal = SineWaveGenerator(N=size, fs=samplerate).four_sine_waves()
    signal_fft = np.abs(np.fft.fft(signal))[:int(size/2)]
    signal_fft2 = np.abs(np.fft.fft(signal))**2

    plt.figure(figsize=(16,9))
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, ".-")
    plt.subplot(3, 1, 2)
    plt.plot(freqs, signal_fft, ".-")
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0, samplerate, size), signal_fft2, ".-")
    plt.show()
