import tensorflow as tf
from training.generator import Generator
from training.architecture_models import *


def execute_training():
    training_data_path='data/output/training_clip_len_17200samples/mfsc_window_400samples'
    validation_data_path='data/output/validation_clip_len_17200samples/mfsc_window_400samples'
    model_name = 'cnn_model_v4_25(12,8,5)'
    batch_size = 256
    epochs = 25

    model = cnn_model_v4()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    training_data = Generator(training_data_path, batch_size)
    validation_data = Generator(validation_data_path, batch_size)

    def scheduler(epoch, lr):
        if epoch == 12:
            return lr * 0.1
        elif epoch == 20:
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
