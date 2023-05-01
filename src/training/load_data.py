import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

class LoadData:
    """
        A class for loading data from NumPy files in parallel.
    Attributes:
        path (str): The path to the directory containing the NumPy files.
    Methods:
        load_data_parallel: Loads all the NumPy files in the directory specified by
                            `path` in parallel using a process pool executor. Returns
                            a tuple of two NumPy arrays: `data` and `labels`.
    """
    def __init__(self, path):
        """
        Constructs a new LoadData object with the specified path.
        Args:
            path (str): The path to the directory containing the NumPy files.
        """
        self.path = path

    def load_data_parallel(self):
        """
        Loads all the NumPy files in the directory specified by `path` in parallel
        using a process pool executor. Each NumPy file should contain a 3D array of
        shape (40, 40, num_channels) and have a filename of the form `file_label.npy`.
        The label of each file is extracted from the filename and stored in the `labels`
        array. Returns a tuple of two NumPy arrays: `data` and `labels`.
        Returns:
            tuple: A tuple of two NumPy arrays: `data` and `labels`. `data` is a 4D array
            of shape (num_files, 1, 40, 40, num_channels), where `num_files` is the number
            of files in the directory. `labels` is a 1D array of shape (num_files,) containing
            the integer labels of the files.
        """
        data = []
        labels = []
        num_files = 0
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            files = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith(".npy")]
            for batch in executor.map(self._load_data, files, chunksize=1000):
                file_data, label = batch
                data.extend(file_data)
                labels.append(label)
                num_files += 1
                if num_files % 10000 == 0:
                    print(f"loaded {num_files} files out of {len(files)}")
        data = np.array(data)
        labels = np.array(labels)
        print("data loaded successfully from path: " + str(self.path))
        return data, labels

    def _load_data(self, file):
        """
        Loads a single NumPy file and extracts its label from the filename.
        Args:
            file (str): The path to the NumPy file.
        Returns:
            tuple: A tuple of two NumPy arrays: `file_data` and `label`. `file_data` is
            a 3D array of shape (40, 40, num_channels) containing the data from the NumPy
            file. `label` is an integer label extracted from the filename.
        """
        file_data = np.load(file)
        label = int(os.path.basename(file).split("_")[-2])
        return np.reshape(file_data, (1, 40, 40)), label
