import tensorflow as tf
import visualkeras
from training.cnn_model_original import cnn_model_original
from training.data_generator import DataGenerator

def execute_training(training_data_path='data/output/training_clip_len_17200samples/mfsc_window_400samples',
                     validation_data_path='data/output/validation_clip_len_17200samples/mfsc_window_400samples',
                     model_name = 'model_original'):

    model = cnn_model_original()

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    batch_size = 256
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
              epochs=120,
              verbose=1,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}.h5',
                                                            monitor='val_binary_accuracy',
                                                            save_best_only=True,),
                         tf.keras.callbacks.TensorBoard(),
                         tf.keras.callbacks.LearningRateScheduler(scheduler)],)

def visualize_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=40, kernel_size=[5,5], strides=(2,2), padding='SAME', activation='relu', input_shape=(40,40,1)),
        tf.keras.layers.Conv2D(filters=20, kernel_size=[5,5], strides=(2,2), padding='SAME', activation='relu'),
        tf.keras.layers.Conv2D(filters=10, kernel_size=[5,5], strides=(2,2), padding='SAME', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dropout(rate=0.75),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    visualkeras.layered_view(model,
                            legend=True,
                            to_file='model.png'
                            ).show()
