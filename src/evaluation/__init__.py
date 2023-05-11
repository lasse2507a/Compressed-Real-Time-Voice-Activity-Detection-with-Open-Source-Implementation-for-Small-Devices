import tensorflow as tf
import numpy as np
from evaluation.data_loader import DataLoader
from evaluation.metrics import *


def execute_evaluation():
    labels_original, preds_original = calc_preds('cnn_model_original_25(12,8,5).h5')
    labels_model_v4, preds_model_v4 = calc_preds('cnn_model_v4_25(12,8,5).h5')
    labels_model_v4_lite, preds_model_v4_lite = calc_preds('cnn_model_v4_25(12,8,5).tflite')
    labels_silero = np.load('data/output/prediction_audio_clip_2/other_model_predictions/silero_labels17200.npy')
    preds_silero = np.load('data/output/prediction_audio_clip_2/other_model_predictions/silero_preds17200.npy')

    labels_list = [labels_model_v4, labels_original, labels_silero, labels_model_v4_lite]
    preds_list = [preds_model_v4, preds_original, preds_silero, preds_model_v4_lite]
    model_names = ['Modified VAD', 'Original VAD', 'Silero VAD', 'Modified VAD Lite']

    precision_recall_plot(labels_list, preds_list, model_names)
    auc_roc_plot(labels_list, preds_list, model_names)


def calc_preds(model_name):
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
