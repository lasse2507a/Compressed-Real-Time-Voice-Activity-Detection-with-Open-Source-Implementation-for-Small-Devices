#from data_processing.__init__ import frequency_test
from data_processing.__init__ import data_generation
#from data_processing.__init__ import preprocessing
#from data_processing.__init__ import plot_mfsc
#from real_time_implementation.__init__ import real_time_implementation
#from training.__init__ import visualize_model
#from training.__init__ import execute_training

if __name__ == '__main__':
    #frequency_test()
    data_generation()
    #preprocessing()
    #plot_mfsc()
    #real_time_implementation()
    #visualize_model()
    #execute_training(training_data_path='data\\output\\training_1000_images',
                     #validation_data_path='data\\output\\validation_1000_images')


    import tensorflow as tf
    print("GPU: " + str(len(tf.config.list_physical_devices('GPU'))))
