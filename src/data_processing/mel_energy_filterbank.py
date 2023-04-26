import numpy as np

class MelEnergyFilterbank:
    def __init__(self, nFFT, nMels, fs):
        self.freq_l = 300
        self.freq_h = 8000
        self.nFFT = nFFT
        self.nMels = nMels
        self.fs = fs
        self.filterbank = np.zeros((self.nMels, nFFT//2))

        lowerMel = 1125 * np.log(1 + self.freq_l/700)
        higherMel = 1125 * np.log(1 + self.freq_h/700)
        melBand = np.linspace(lowerMel, higherMel, self.nMels+2)
        freqBand = 700 * (np.exp(melBand/1125) - 1)
        f = np.floor((nFFT + 1) * freqBand/fs).astype(int)

        for m in range(1, self.nMels+1):
            for k in range(0, nFFT//2):
                if k > f[m-1] and k <= f[m]:
                    self.filterbank[m-1,k] = (k - f[m-1])/(f[m] - f[m-1])
                elif k > f[m] and k <= f[m+1]:
                    self.filterbank[m-1,k] = (f[m+1] - k)/(f[m+1] - f[m])

        self.filterbank = np.transpose(self.filterbank)
        self.melEnergy = np.zeros((self.nMels, 1))

    def compute(self, fftPower):
        for i in range(0, self.nMels):
            self.melEnergy[i] = np.log(np.sum(self.filterbank[:,i] * fftPower[0:self.nFFT//2]))
        return self.melEnergy.flatten()
