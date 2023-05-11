import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np


class DataLoader:
    def __init__(self, path):
        self.path = path


    def load_data_parallel(self):
        """
        Load data from path in parrallel.
        Return: (numpy.ndarray): Data consisting of images with specific shape of (1, 40, 40).
                (numpy.ndarray): Array of labels.
        """
        data = []
        labels = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            files = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith(".npy")]
            for batch in executor.map(self._load_data, files, chunksize=1000):
                file_data, label = batch
                data.extend(file_data)
                labels.append(label)
        data = np.array(data)
        labels = np.array(labels)
        print("data loaded successfully from path: " + str(self.path))
        return data, labels


    def _load_data(self, file):
        """
        Helper function load a single file.
        Args: file (str): Name of the audio file to load.
        """
        file_data = np.load(file)
        label = int(os.path.basename(file).split("_")[-2])
        return np.reshape(file_data, (1, 40, 40)), label
