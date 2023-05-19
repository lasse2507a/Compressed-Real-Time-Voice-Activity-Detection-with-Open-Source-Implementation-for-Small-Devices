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
        plt.figure(figsize=(8, 6))
        librosa.display.specshow(image_MFSC, y_coords=librosa.mel_frequencies(n_mels=40, fmin=300, fmax=8000), x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.colorbar(format='%+2.f dB')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        file_name = f'{file}_mfsc_example{i+1}.png'
        plt.savefig(image_destination_path + file_name)


def plot_mel_filterbank(image_destination_path):
    fs = 16000
    flow = 300
    fhigh = 8000
    n_mels = 40

    mel_low = 2595 * np.log10(1 + flow / 700)
    mel_high = 2595 * np.log10(1 + fhigh / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fft_bins = np.floor((fs + 1) * hz_points / fs).astype(int)
    filterbank = np.zeros((n_mels, int(np.floor(fs / 2 + 1))))

    for m in range(1, n_mels + 1):
        f_m_minus = fft_bins[m - 1]
        f_m = fft_bins[m]
        f_m_plus = fft_bins[m + 1]
        filterbank[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
        filterbank[m - 1, f_m:f_m_plus] = 1 - (np.arange(f_m, f_m_plus) - f_m) / (f_m_plus - f_m)

    plt.figure(figsize=(9, 6))
    plt.tight_layout()
    plt.xlim(flow, fhigh)
    plt.ylim(0.01, 1)
    plt.title('Mel Filterbank')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.plot(filterbank[:40, :].T, '#1f77b4')  # Plot the first 40 filters
    file_name = 'mel_filterbank_.png'
    plt.savefig(image_destination_path + file_name)


def plot_mel_scale_vs_frequency(image_destination_path):
    def mel_scale(frequency):
        return 2595 * np.log10(1 + frequency / 700)

    def inverse_mel_scale(mel):
        return 700 * (10**(mel / 2595) - 1)

    # Generate frequencies from 0 to 8000 Hz
    frequencies = np.linspace(300, 8000, 1000)

    # Compute mel scale values for the frequencies
    mel_values = mel_scale(frequencies)

    # Plot the relation between mel scale and frequency
    plt.figure(figsize=(7, 6))
    plt.plot(frequencies, mel_values)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mel Scale')
    plt.title('Mel Scale vs. Frequency')

    # Set the x-axis tick labels to display the edge frequencies
    x_ticks = [300, 800, 2000, 5000, 8000]
    x_tick_labels = ['300', '800', '2000', '5000', '8000']
    plt.xticks(x_ticks, x_tick_labels)

    # Add ticks on the y-axis for the corresponding y-values of 300 and 8000
    y_ticks = [mel_scale(300), mel_scale(800), mel_scale(2000), mel_scale(5000), mel_scale(8000)]
    y_tick_labels = ['{:.2f}'.format(mel_scale(300)), '{:.2f}'.format(mel_scale(800)), '{:.2f}'.format(mel_scale(2000)), '{:.2f}'.format(mel_scale(5000)), '{:.2f}'.format(mel_scale(8000))]
    plt.yticks(y_ticks, y_tick_labels)
    plt.grid()
    plt.tight_layout()
    file_name = 'mel_scale_vs_frequency.png'
    plt.savefig(image_destination_path + file_name)


def plot_power_spectrum(data_path, image_destination_path):
    for i, file in enumerate(os.listdir(data_path)):
        if file.endswith('.wav'):
            data, _ = librosa.load(os.path.join(data_path, file))
            data = data[:16000]
            power_spectrum = (np.abs(np.fft.fft(data))**2)[:int(8000)]
            plt.figure(figsize=(7, 6))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude')
            plt.title('Power Spectrum')
            plt.tight_layout()
            plt.plot(np.linspace(0, 8000, 8000), power_spectrum)
            file_name = f'{file}_power_spectrum_{i+1}.png'
            plt.savefig(image_destination_path + file_name)


def plot_waveform(data_path, image_destination_path):
    for i, file in enumerate(os.listdir(data_path)):
        if file.endswith('.wav'):
            data, _ = librosa.load(os.path.join(data_path, file))
            plt.figure(figsize=(7, 6))
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.title('Waveform')
            plt.tight_layout()
            plt.plot(np.linspace(0, 1, 16000), data[:16000])
            file_name = f'{file}_waveform_{i+1}.png'
            plt.savefig(image_destination_path + file_name)


if __name__ == '__main__':
    # plot_mfsc(data_path='data/output/preprocess_test/mfsc_window_400samples',
    #           image_destination_path='data/output/preprocess_test/images/')
    # plot_waveform(data_path='data/output/preprocess_test',
    #               image_destination_path='data/output/preprocess_test/images/')
    # plot_power_spectrum(data_path='data/output/preprocess_test',
    #                     image_destination_path='data/output/preprocess_test/images/')
    #plot_mel_filterbank(image_destination_path='data/output/preprocess_test/images/')
    plot_mel_scale_vs_frequency(image_destination_path='data/output/preprocess_test/images/')
