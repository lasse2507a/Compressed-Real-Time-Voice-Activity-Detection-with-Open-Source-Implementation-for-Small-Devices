import tensorflow as tf


def cnn_model_original(K=40, L=20, M=10, N=100, keep_prob=0.75):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=K, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu', input_shape=(40,40,1)),
        tf.keras.layers.Conv2D(filters=L, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=M, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N, activation='relu'),
        tf.keras.layers.Dropout(rate=keep_prob),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def cnn_model_v2(K=40, L=20, M=10, N=100, keep_prob=0.75):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=K, kernel_size=[5,5], strides=(2,2), padding='same', input_shape=(40,40,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters=L, kernel_size=[5,5], strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters=M, kernel_size=[5,5], strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N, activation='relu'),
        tf.keras.layers.Dropout(rate=keep_prob),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def cnn_model_v3(K=40, L=20, M=10, N=100, keep_prob=0.75):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=K, kernel_size=[3,3], strides=(2,2), padding='same', activation='relu', input_shape=(40,40,1)),
        tf.keras.layers.Conv2D(filters=L, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=M, kernel_size=[7,7], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N, activation='relu'),
        tf.keras.layers.Dropout(rate=keep_prob),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def cnn_model_v4(K=40, L=20, M=10, N=100, keep_prob=0.75):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=K, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu', input_shape=(40,40,1)),
        tf.keras.layers.Conv2D(filters=L, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=M, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=5, kernel_size=[5,5], strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N, activation='relu'),
        tf.keras.layers.Dropout(rate=keep_prob),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
