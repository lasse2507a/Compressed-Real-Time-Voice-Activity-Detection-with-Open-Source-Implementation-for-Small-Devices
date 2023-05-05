import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
#import real_time_implementation.__init__ as real_time_implementation
#import training.__init__ as training
import evaluation.__init__ as evaluation

if __name__ == '__main__':
    #real_time_implementation.real_time_implementation()
    #training.execute_training()

    # y, y_ = evaluation.predictions_webrtc()
    # import matplotlib.pyplot as plt
    # plt.plot(y_[0], label='Mode 0')
    # plt.plot(y_[1], label='Mode 1')
    # plt.plot(y_[2], label='Mode 2')
    # plt.plot(y_[3], label='Mode 3')
    # plt.xlim(-0.1, 30000)
    # plt.ylim(-0.1, 1.1)
    # plt.legend()
    # plt.show()

    y1, y1_ = evaluation.predictions()

    y2 = np.load('data/output/prediction_audio_clip_2/silero_labels.npy')
    y2_ = np.load('data/output/prediction_audio_clip_2/silero_preds.npy')

    fpr1, tpr1 = evaluation.auc_roc(y1, y1_)
    fpr2, tpr2 = evaluation.auc_roc(y2, y2_)

    plt.plot(fpr1, tpr1)
    plt.plot(fpr2, tpr2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    auc_value1 = roc_auc_score(y1, y1_)
    auc_value2 = roc_auc_score(y2, y2_)
    plt.title(f"ROC Curve (AUC = {auc_value1:.4f}, AUC = {auc_value2:.4f})")
    plt.show()

    # recalls1, precisions1 = evaluation.precision_recall_plot(y1, y1_)
    # recalls2, precisions2 = evaluation.precision_recall_plot(y2, y2_)

    # plt.plot(recalls1, precisions1)
    # plt.plot(recalls2, precisions2)
    # #plt.xlim(0, 1.05)
    # #plt.ylim(0, 1.05)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.show()
