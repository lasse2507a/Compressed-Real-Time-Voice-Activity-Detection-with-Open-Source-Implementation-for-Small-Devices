import csv
import os
import numpy as np
import scipy.io.wavfile as wavfile

class DataHandler:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.names = []
        self.start_times = []
        self.end_times = []
        self.labels = []
        self.indices_correct_size = []
        self.indices_current_file = []
        self.is_same_file = False

    def load_csv(self, path = 'data\\ava_speech_labels_v1.csv'):
        with open(path, encoding = "utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            for row in csv_reader:
                self.names.append(row[0])
                self.start_times.append(float(row[1]))
                self.end_times.append(float(row[2]))
                self.labels.append(int(row[3]))

    def create_data(self, size):
        for k, start_time in enumerate(self.start_times):
            if ((self.end_times[k] - start_time) * self.samplerate) >= size:
                self.indices_correct_size.append(k)
        print("indices of intervals with correct size: " + str(len(self.indices_correct_size)))

        os.makedirs(f"data\\output\\training_clip_len_{size}samples", exist_ok = True)
        os.makedirs(f"data\\output\\test_clip_len_{size}samples", exist_ok = True)

        for file in os.listdir('data\\input'):
            current_file = wavfile.read(os.path.join('data\\input', file))[1]
            l = 0
            for i in self.indices_correct_size:
                if 'data\\input\\' + self.names[i] + '.wav' == os.path.join('data\\input', file):
                    self.indices_current_file.append(i)
            total_number_of_clips = 0
            for j in self.indices_current_file:
                number_of_clips = int(np.floor(((self.end_times[j] - self.start_times[j]) * self.samplerate) / size))
                total_number_of_clips += number_of_clips
                begin = self.start_times[j] * self.samplerate
                for m in range(number_of_clips):
                    end = begin + size
                    clip = current_file[int(begin):int(end)]
                    l += 1
                    if l % 2 == 0:
                        wavfile.write(f"data\\output\\training_clip_len_{size}samples\\{j+1},{m+1}_{self.names[j]}_{self.labels[j]}.wav",
                                    self.samplerate, np.array(clip, dtype=np.int16))
                    else:
                        wavfile.write(f"data\\output\\test_clip_len_{size}samples\\{j+1},{m+1}_{self.names[j]}_{self.labels[j]}.wav",
                                    self.samplerate, np.array(clip, dtype=np.int16))
                    begin = end
            print(os.path.join('data\\input', file) + " indicies: " + str(len(self.indices_current_file)) + " total clips: " + str(total_number_of_clips))
            self.indices_current_file = []
