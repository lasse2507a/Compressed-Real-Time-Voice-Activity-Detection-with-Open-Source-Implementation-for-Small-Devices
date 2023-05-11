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


    def load_csv(self, path):
        """
        Load metadata from a CSV file. The format has to be identical to data\\metadata\\ava_speech_labels_v1.csv.
        Args: path (str): The path to the CSV file containing the metadata.
        """
        with open(path, encoding = "utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter = ',')
            for row in csv_reader:
                self.names.append(row[0])
                self.start_times.append(float(row[1]))
                self.end_times.append(float(row[2]))
                self.labels.append(int(row[3]))


    def create_data(self, size):
        """
        Create all possible audio clips of a given size for each interval from the input data.
        Args: size (int): The size of the clips in samples.
        """
        for k, start_time in enumerate(self.start_times):
            if ((self.end_times[k] - start_time) * self.samplerate) >= size:
                self.indices_correct_size.append(k)
        print("indices of intervals with correct size: " + str(len(self.indices_correct_size)))

        os.makedirs(f"data\\output\\training_clip_len_{size}samples", exist_ok = True)
        os.makedirs(f"data\\output\\validation_clip_len_{size}samples", exist_ok = True)

        for file in os.listdir('data\\input'):
            current_file = wavfile.read(os.path.join('data\\input', file))[1]
            dest_split = 0
            for index in self.indices_correct_size:
                if 'data\\input\\' + self.names[index] + '.wav' == os.path.join('data\\input', file):
                    self.indices_current_file.append(index)
            total_num_of_clips = 0
            for index_current_file in self.indices_current_file:
                num_of_clips = int(np.floor(((self.end_times[index_current_file] - self.start_times[index_current_file]) * self.samplerate) / size))
                total_num_of_clips += num_of_clips
                begin = self.start_times[index_current_file] * self.samplerate
                for clip_num in range(num_of_clips):
                    end = begin + size
                    clip = current_file[int(begin):int(end)]
                    dest_split += 1
                    if dest_split % 2 == 0:
                        wavfile.write(f"data\\output\\training_clip_len_{size}samples\\{index_current_file+1},{clip_num+1}_{self.names[index_current_file]}_{self.labels[index_current_file]}.wav",
                                      self.samplerate, np.array(clip, dtype=np.int16))
                    else:
                        wavfile.write(f"data\\output\\validation_clip_len_{size}samples\\{index_current_file+1},{clip_num+1}_{self.names[index_current_file]}_{self.labels[index_current_file]}.wav",
                                      self.samplerate, np.array(clip, dtype=np.int16))
                    begin = end
            print(os.path.join('data\\input', file) + " indicies: " + str(len(self.indices_current_file)) + " total clips: " + str(total_num_of_clips))
            self.indices_current_file = []


if __name__ == '__main__':
    datahandler = DataHandler(samplerate = 16000)
    datahandler.load_csv(path = 'data\\metadata\\ava_speech_labels_v1.csv')
    datahandler.create_data(size = 400)
