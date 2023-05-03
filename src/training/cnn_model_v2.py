import tensorflow as tf

def cnn_model_v2(keep_prob=0.75):
    return tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=[3,3], padding='same', activation='relu', input_shape=(40,40,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3], padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=keep_prob),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
