from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import os
import numpy as np
import librosa


class Preprocessor:
    def __init__(self, samplerate, size):
        """
        Initializes a Preprocessor instance.
        Args: samplerate (int): Sampling rate of the audio.
              size (int): Window size in samples.
        """
        self.samplerate = samplerate
        self.size = size
        self.hop_length = int(size/2)
        self.FFT_size = int(2**np.ceil(np.log2(self.size)))
        self.window = np.hanning(self.size)
        self.mel_size = 40


    def process(self, data_path):
        """
        Preprocesses audio files located in the given directory using multithreading.
        Args: data_path (str): Path to directory containing audio files.
        """
        print("Preprocessing started")
        os.makedirs(f"{data_path}\\mfsc_window_{self.size}samples", exist_ok = True)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for file in os.listdir(data_path):
                if file.endswith(".wav"):
                    executor.submit(self._process_file, file, data_path)
        print("Preprocessing finished")


    def _process_file(self, file, data_path):
        """
        Helper function to preprocess a single audio file.
        Args: file (str): Name of the audio file to preprocess.
              data_path (str): Path to directory containing audio files.
        """
        file_name = file.replace(".wav", "")
        frame_num = 1
        frames_MFSC = Queue(maxsize=40)
        frames_MFSC.queue.clear()
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
                frames_MFSC.put(librosa.power_to_db(melspectrogram, ref=np.max))
                if frames_MFSC.full():
                    image_MFSC = np.concatenate(list(frames_MFSC.queue), axis=1)
                    np.save(f"{data_path}\\mfsc_window_{self.size}samples\\{file_name}_mfsc{frame_num}.npy", image_MFSC)
                    frame_num += 1
                    for _ in range(5):
                        frames_MFSC.get()


if __name__ == "__main__":
    preprocessor = Preprocessor(samplerate=16000, size=400)
    preprocessor.process('data\\output\\training_clip_len_17200samples')
    preprocessor.process('data\\output\\validation_clip_len_17200samples')
