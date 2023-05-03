import numpy as np


def confusion_matrix(labels, predictions, threshold=0.5):
    labels = np.array(labels, dtype=np.float32)
    predictions = np.array(predictions, dtype=np.float32)

    for i, prediction in enumerate(predictions):
        if prediction >= threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i, prediction in enumerate(predictions):
        if prediction == 1 and labels[i] == 1:
            tp += 1
        elif prediction == 0 and labels[i] == 0:
            tn += 1
        elif prediction == 1 and labels[i] == 0:
            fp += 1
        elif prediction == 0 and labels[i] == 1:
            fn += 1

    return {'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn}
