import os
from queue import Queue
import numpy as np
#import matplotlib.pyplot as plt
import librosa

class Preprocessor:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size
        self.hop_length = int(size/2)
        self.FFT_size = int(2**np.ceil(np.log2(self.size)))
        self.window = np.hanning(self.size)
        self.frames_MFSC = Queue(maxsize=40)
        self.mel_size = 40

    def process(self, data_path):
        print("preprocessing started")
        os.makedirs(f"{data_path}\\mfsc_window_{self.size}samples", exist_ok = True)

        for file in os.listdir(data_path):
            file_name = file.replace(".wav", "")
            j = 1
            self.frames_MFSC.queue.clear()
            if file.endswith(".wav"):
                current_file, _ = librosa.load(os.path.join(data_path, file), sr=self.samplerate)
                num_frames = int(np.floor((len(current_file) - self.hop_length) / self.hop_length))
                print(str(file) + " loaded, num_frames: " + str(num_frames))
                for i in range(num_frames):
                    start = int(i * self.hop_length)
                    end = start + self.size
                    frame = current_file[start:end] * self.window
                    zero_padded_frame = np.zeros(self.FFT_size)
                    zero_padded_frame[:self.size] = frame
                    melspectrogram = librosa.feature.melspectrogram(y=zero_padded_frame, sr=self.samplerate, window='boxcar', n_fft=self.FFT_size, hop_length=self.hop_length,
                                                                    center=False, n_mels=self.mel_size, fmin=300, fmax=8000)
                    self.frames_MFSC.put(librosa.power_to_db(melspectrogram, ref=np.max))
                    if self.frames_MFSC.qsize() == 40:
                        picture_MFSC = np.concatenate(list(self.frames_MFSC.queue), axis=1)
                        np.save(f"{data_path}\\mfsc_window_{self.size}samples\\mfsc{j}_{file_name}.npy", picture_MFSC)
                        print(f"{j} picture saved")
                        j += 1
                        for i in range(5):
                            self.frames_MFSC.get()

                # plt.figure(figsize=(10, 4))
                # librosa.display.specshow(picture_MFSC, x_axis='time', y_axis='mel', cmap='coolwarm')
                # plt.colorbar()
                # plt.title('MFSC Features')
                # plt.tight_layout()
                # plt.show()
