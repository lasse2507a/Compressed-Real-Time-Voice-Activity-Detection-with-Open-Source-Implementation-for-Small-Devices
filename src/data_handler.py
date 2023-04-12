import csv
import os
import numpy as np

class DataHandler:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.names = []
        self.start_times = []
        self.end_times = []
        self.labels = []
        self.indices = []

    def load_csv(self, path = 'data\\data_test.csv'):
        with open(path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            for row in csv_reader:
                self.names.append(row[0])
                self.start_times.append(float(row[1]))
                self.end_times.append(float(row[2]))
                self.labels.append(int(row[3]))

    def create_data(self, size = 48000, make_test_data = True):
        for i in self.start_times:
            if (self.end_times[i] - self.start_times[i]) * self.samplerate > size:
                self.indices.append(i)

        if make_test_data:
            os.makedirs("data\\training_data")
            os.makedirs("data\\test_data")
        else:
            os.makedirs("data\\training_data")

        for file in os.listdir('data\\audio'):
            open(file, 'r')





datahandler = DataHandler(16000)
datahandler.load_csv()
