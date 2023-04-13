import csv
import os
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

    def create_data(self, size = 48000):
        for k in range(len(self.start_times)):
            if (self.end_times[k] - self.start_times[k]) * self.samplerate >= size:
                self.indices_correct_size.append(k)

        os.makedirs(f"data\\training_data_{size}", exist_ok = True)
        os.makedirs(f"data\\test_data_{size}", exist_ok = True)

        for file in os.listdir('data\\audio'):
            with open(file, 'r', encoding = "utf-8") as current_file:
                for i in self.indices_correct_size:
                    if self.names[i] == current_file.name:
                        self.indices_current_file.append(self.indices_correct_size[i])
                        self.is_same_file = True
                    elif self.is_same_file:
                        break
                for j in self.indices_current_file:
                    begin = self.start_times[j] * self.samplerate
                    end = self.start_times[j] * self.samplerate + size
                    if j % 2 == 0:
                        wavfile.write(f"data\\training_data_{size}\\{j}_{current_file.name}_training_{size}", self.samplerate, current_file[begin:end])
                    else:
                        wavfile.write(f"data\\test_data_{size}\\{j}_{current_file.name}_test_{size}", self.samplerate, current_file[begin:end])

datahandler = DataHandler(16000)
datahandler.load_csv()
datahandler.create_data()
