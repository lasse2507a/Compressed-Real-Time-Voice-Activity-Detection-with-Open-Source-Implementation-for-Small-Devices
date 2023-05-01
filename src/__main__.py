from data_processing.__init__ import frequency_test
from data_processing.__init__ import data_generation
from data_processing.__init__ import preprocessing
from real_time_implementation.__init__ import real_time_implementation
from training.__init__ import execute_training
from training.__init__ import visualize_model

if __name__ == '__main__':
    #data_generation()
    #frequency_test()
    #preprocessing()
    #real_time_implementation()
    #visualize_model()
    execute_training(training_data_path='data\\output\\training_clip_len_17200samples\\mfsc_window_400samples',
                     validation_data_path='data\\output\\test_clip_len_17200samples\\mfsc_window_400samples')
