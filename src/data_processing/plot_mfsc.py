import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

def plot_mfsc(data_path, image_destination_path):
    """
    Plot MFSC (Mel-Frequency Spectral Coefficients) features for a given directory of files and save the output images.
    Args: data_path (str): The directory path containing the MFSC features files.
          image_destination_path (str): The directory path to save the output images.
    """
    for i, file in enumerate(os.listdir(data_path)):
        image_MFSC = np.load(os.path.join(data_path, file))
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(image_MFSC, y_coords=librosa.mel_frequencies(n_mels=40, fmin=300, fmax=8000), x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.colorbar()
        plt.title('MFSC Features')
        plt.tight_layout()
        file_name = f'{file}_mfsc_example{i+1}.png'
        plt.savefig(image_destination_path + file_name)


if __name__ == '__main__':
    plot_mfsc(data_path='data\\output\\preprocess_test\\mfsc_examples',
              image_destination_path='data\\output\\preprocess_test\\images')
