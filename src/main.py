#import real_time_implementation.__init__ as real_time_implementation
#import training.__init__ as training
import evaluation.__init__ as evaluation

if __name__ == '__main__':
    #real_time_implementation.real_time_implementation()
    #training.execute_training()
    y, y_ = evaluation.predictions()
    evaluation.precision_recall_plot(y, y_)
    evaluation.auc_roc(y, y_)
