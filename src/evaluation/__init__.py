from concurrent.futures import ThreadPoolExecutor
import os
import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
from sklearn.metrics import roc_auc_score
from evaluation.data_loader import DataLoader
from evaluation.confusion_matrix import confusion_matrix
from evaluation.metrics import *


def predictions():
    data_loader_mfsc = DataLoader("data\\output\\prediction_audio_clip_2\\mfsc_400samples")
    data, labels = data_loader_mfsc.load_data_parallel()

    model = tf.keras.models.load_model("models\\cnn_model_original_25(12,8,5).h5")
    preds = model.predict(x=data, verbose=1)

    return labels, preds


# Normalisering af data
# def load_file(file):
#     file_data, _ = librosa.load(file, sr=16000, mono=True)
#     scaler = MinMaxScaler()
#     normalized_data = scaler.fit_transform(file_data.reshape(-1, 1)).reshape(-1)
#     clips = np.frombuffer(normalized_data, dtype=np.int16).reshape(-1, 2)
#     label = int(os.path.basename(file).split("_")[-1][0])
#     return clips, label



def load_file(file):
    file_data, _ = librosa.load(file, sr=16000, mono=True)
    clips = np.frombuffer(file_data, dtype=np.int16).reshape(-1, 2)
    label = int(os.path.basename(file).split("_")[-1][0])
    return clips, label


def predictions_webrtc():
    labels = []
    path = "data/output/prediction_audio_clip_2/audio_clip_2_480samples"
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".wav")]
    preds = []

    batch_size = 100
    num_batches = int(np.floor(len(files) / batch_size))

    with ThreadPoolExecutor() as executor:
        for batch_idx in range(num_batches):
            batch_files = files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            results = list(executor.map(load_file, batch_files))
            clips, batch_labels = zip(*results)
            labels += batch_labels
            clips = np.array(clips)
            clips = np.reshape(clips, (batch_size, -1))
            print(f'batch {batch_idx+1}/{num_batches}')

            model_webrtc = webrtcvad.Vad()

            batch_preds = []
            for i in range(4):
                model_webrtc.set_mode(i)
                mode_preds = [model_webrtc.is_speech(buf=clip, sample_rate=16000) for clip in clips]
                batch_preds.append(mode_preds)

            preds.append(np.array(batch_preds))

    return labels, np.concatenate(preds, axis=1)


def precision_recall_plot(labels, preds, N=100, is_webrtc=False):
    precisions = []
    recalls = []

    if is_webrtc:
        for i in range(4):
            cm = confusion_matrix(labels, preds[i])
            precisions.append(precision(cm))
            recalls.append(recall(cm))
            print(precision(cm))
            print(recall(cm))
    else:
        thresholds = np.linspace(0, 1, N)
        for threshold in thresholds:
            cm = confusion_matrix(labels, preds, threshold)
            precisions.append(precision(cm))
            recalls.append(recall(cm))


    plt.plot(recalls, precisions)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def auc_roc(labels, preds, N=100):
    thresholds = np.linspace(0, 1, N)
    tpr = []
    fpr = []

    for threshold in thresholds:
        cm = confusion_matrix(labels, preds, threshold)
        tpr.append(recall(cm))
        fpr.append(fp_rate(cm))

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    auc_value = roc_auc_score(labels, preds)
    plt.title(f"ROC Curve (AUC = {auc_value:.4f})")
    plt.show()
