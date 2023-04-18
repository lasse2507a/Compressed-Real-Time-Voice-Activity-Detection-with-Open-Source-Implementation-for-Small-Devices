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
            print(len(self.names))
            print(len(self.start_times))
            print(len(self.end_times))
            print(len(self.labels))

    def create_data(self, size = 44100*3):
        for k, start_time in enumerate(self.start_times):
            if ((self.end_times[k] - start_time) * self.samplerate) >= size:
                self.indices_correct_size.append(k)
        print(len(self.indices_correct_size))

        os.makedirs(f"data\\output\\training_{size/self.samplerate:.3f}s", exist_ok = True)
        os.makedirs(f"data\\output\\test_{size/self.samplerate:.3f}s", exist_ok = True)

        for file in os.listdir('data\\input'):
            current_file = wavfile.read(os.path.join('data\\input', file))[1]
            l = 0
            for i in self.indices_correct_size:
                if 'data\\input\\' + self.names[i] + '.wav' == os.path.join('data\\input', file):
                    self.indices_current_file.append(i)
                #    self.is_same_file = True
                #elif self.is_same_file:
                #    self.is_same_file = False
                #    break
            for j in self.indices_current_file:
                begin = self.start_times[j] * self.samplerate
                end = self.start_times[j] * self.samplerate + size
                clip = current_file[int(begin):int(end)]
                l += 1
                if l % 2 == 0:
                    wavfile.write(f"data\\output\\training_{size/self.samplerate:.3f}s\\{j+1}_{self.names[j]}_{self.labels[j]}.wav",
                                  self.samplerate, np.array(clip, dtype=np.float32))
                else:
                    wavfile.write(f"data\\output\\test_{size/self.samplerate:.3f}s\\{j+1}_{self.names[j]}_{self.labels[j]}.wav",
                                  self.samplerate, np.array(clip, dtype=np.float32))

if __name__ == '__main__':
    datahandler = DataHandler(44100)
    datahandler.load_csv()
    datahandler.create_data()
