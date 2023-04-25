import numpy as np

class MelEnergyFilterbank:
    def __init__(self, freq_l, freq_h, nFFT, nFilt, fs):
        self.freq_l = freq_l
        self.freq_h = freq_h
        self.nFFT = nFFT
        self.nFilt = nFilt
        self.fs = fs
        self.filterbank = np.zeros((nFilt, nFFT//2))

        lowerMel = 1125 * np.log(1 + freq_l/700)
        higherMel = 1125 * np.log(1 + freq_h/700)
        melBand = np.linspace(lowerMel, higherMel, nFilt+2)
        freqBand = 700*(np.exp(melBand/1125) - 1)
        f = np.floor((nFFT + 1) * freqBand/fs).astype(int)

        for m in range(1, nFilt+1):
            for k in range(0, nFFT//2):
                if k > f[m-1] and k <= f[m]:
                    self.filterbank[m-1,k] = (k - f[m-1])/(f[m] - f[m-1])
                elif k > f[m] and k <= f[m+1]:
                    self.filterbank[m-1,k] = (f[m+1] - k)/(f[m+1] - f[m])

        self.filterbank = np.transpose(self.filterbank)
        self.melEnergy = np.zeros((nFilt, 1))

    def compute(self, fftPower):
        for i in range(0, self.nFilt):
            self.melEnergy[i] = np.log(np.sum(self.filterbank[:,i] * fftPower[0:self.nFFT//2]))
        return self.melEnergy
