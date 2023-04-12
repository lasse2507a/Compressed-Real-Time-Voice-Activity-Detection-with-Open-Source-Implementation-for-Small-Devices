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
        self.indices = []

    def load_csv(self, path = 'data\\data_test.csv'):
        with open(path, encoding = "utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            for row in csv_reader:
                self.names.append(row[0])
                self.start_times.append(float(row[1]))
                self.end_times.append(float(row[2]))
                self.labels.append(int(row[3]))

    def create_data(self, size = 48000):
        for i in self.start_times:
            if (self.end_times[i] - self.start_times[i]) * self.samplerate > size:
                self.indices.append(i)

        os.makedirs("data\\training_data")
        os.makedirs("data\\test_data")

        for file in os.listdir('data\\audio'):
            current_file = open(file, 'r', encoding = "utf-8")
            for index in self.indices:
                if current_file.name == self.names[index]:
                    begin = self.start_times[index] * self.samplerate
                    end = self.start_times[index] * self.samplerate + size
                    if index % 2 == 0:
                        wavfile.write(f"data\\training_data\\{index}_{current_file.name}_training", self.samplerate, current_file[begin:end])
                    else:
                        wavfile.write(f"data\\test_data\\{index}_{current_file.name}_test", self.samplerate, current_file[begin:end])

datahandler = DataHandler(16000)
datahandler.load_csv()
