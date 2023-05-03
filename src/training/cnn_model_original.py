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
