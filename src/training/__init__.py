import tensorflow as tf
import visualkeras
from training.cnn_model_original import cnn_model_original
from training.cnn_model_v2 import cnn_model_v2
from training.data_generator import DataGenerator

def execute_training():
    training_data_path='data/output/training_clip_len_17200samples/mfsc_window_400samples',
    validation_data_path='data/output/validation_clip_len_17200samples/mfsc_window_400samples'
    model_name = 'cnn_model_v2'
    batch_size = 256
    epochs = 100

    model = cnn_model_v2()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    training_data = DataGenerator(training_data_path, batch_size)
    validation_data = DataGenerator(validation_data_path, batch_size)

    def scheduler(epoch, lr):
        if epoch == 1:
            return lr * 0.1
        elif epoch == 10:
            return lr * 0.1
        else:
            return lr

    model.fit(x=training_data,
              validation_data=validation_data,
              epochs=epochs,
              verbose=1,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}.h5',
                                                            monitor='val_binary_accuracy',
                                                            save_best_only=True,),
                         tf.keras.callbacks.TensorBoard(),
                         tf.keras.callbacks.LearningRateScheduler(scheduler)],)

def visualize_model():
    model = cnn_model_original()

    visualkeras.layered_view(model,
                            legend=True,
                            to_file='model.png').show()
