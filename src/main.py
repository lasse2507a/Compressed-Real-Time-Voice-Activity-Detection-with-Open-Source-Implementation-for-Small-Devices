#import real_time_implementation.__init__ as real_time_implementation
#import training.__init__ as training
import evaluation.__init__ as evaluation

if __name__ == '__main__':
    #real_time_implementation.real_time_implementation()
    #training.execute_training()
    y, y_ = evaluation.predictions_webrtc()
    import matplotlib.pyplot as plt
    plt.plot(y_[0], label='Mode 0')
    plt.plot(y_[1], label='Mode 1')
    plt.plot(y_[2], label='Mode 2')
    plt.plot(y_[3], label='Mode 3')

    plt.legend()
    plt.show()
    #evaluation.precision_recall_plot(y, y_, is_webrtc=True)
