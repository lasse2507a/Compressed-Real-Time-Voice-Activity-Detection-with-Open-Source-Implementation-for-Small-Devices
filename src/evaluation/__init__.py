from concurrent.futures import ThreadPoolExecutor
import os
import librosa
import tensorflow as tf
import numpy as np
import webrtcvad
from evaluation.data_loader import DataLoader
from evaluation.metrics import *


def execute_evaluation():
    labels_original, preds_original = predictions('cnn_model_original_25(12,8,5).h5')
    labels_model_v4, preds_model_v4 = predictions('cnn_model_v4_25(12,8,5).h5')
    labels_model_v4_lite, preds_model_v4_lite = predictions('cnn_model_v4_25(12,8,5).tflite')
    #labels_webrtc, preds_webrtc = predictions_webrtc('data/output/prediction_audio_clip_2/audio_clip_2_480samples')
    labels_silero = np.load('data/output/prediction_audio_clip_2/other_model_predictions/silero_labels17200.npy')
    preds_silero = np.load('data/output/prediction_audio_clip_2/other_model_predictions/silero_preds17200.npy')

    labels_list = [labels_model_v4, labels_original, labels_silero, labels_model_v4_lite,labels_model_v4_latency_lite, labels_model_v4_size_lite]
    preds_list = [preds_model_v4, preds_original, preds_silero, preds_model_v4_lite, preds_model_v4_latency_lite, preds_model_v4_size_lite]
    model_names = ['Modified VAD', 'Original VAD', 'Silero VAD', 'Modified VAD Lite', 'Modified VAD Latency Lite', 'Modified VAD Size Lite']

    precision_recall_plot(labels_list, preds_list, model_names)
    auc_roc_plot(labels_list, preds_list, model_names)


def predictions(model_name):
    data_loader_mfsc = DataLoader("data/output/prediction_audio_clip_2/mfsc_400samples")
    data, labels = data_loader_mfsc.load_data_parallel()
    preds = []

    if model_name.split('.')[-1] == 'tflite':
        for image in data:
            interpreter = tf.lite.Interpreter(model_path=f"models/{model_name}")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
            interpreter.set_tensor(input_details['index'], image.astype(np.float32))
            interpreter.invoke()
            output_details = interpreter.get_output_details()[0]
            preds.append(interpreter.get_tensor(output_details['index']).ravel())
    elif model_name.split('.')[-1] == 'h5':
        model = tf.keras.models.load_model(f"models/{model_name}")
        preds = model.predict(x=data, verbose=1)
        print(f'finished predictions with {model_name}')
    else:
        print(f'{model_name} wrong format')

    return labels, preds


def predictions_webrtc(path):
    def _load_file(file):
        file_data, _ = librosa.load(file, sr=16000, mono=True)
        clips = np.frombuffer(file_data, dtype=np.int16).reshape(-1, 2)
        label = int(os.path.basename(file).split("_")[-1][0])
        return clips, label


    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".wav")]
    labels = []
    preds = []

    batch_size = 100
    num_batches = int(np.floor(len(files) / batch_size))

    model_webrtc = webrtcvad.Vad()

    with ThreadPoolExecutor() as executor:
        for batch_idx in range(num_batches):
            batch_files = files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            results = list(executor.map(_load_file, batch_files))
            clips, batch_labels = zip(*results)
            labels += batch_labels
            clips = np.array(clips)
            clips = np.reshape(clips, (batch_size, -1))

            batch_preds = []
            for i in range(4):
                model_webrtc.set_mode(i)
                mode_preds = [model_webrtc.is_speech(buf=clip, sample_rate=16000) for clip in clips]
                batch_preds.append(mode_preds)

            preds.append(np.array(batch_preds))

    print('finished predictions with WebRTC VAD')

    return labels, np.concatenate(preds, axis=1)
