import tensorflow as tf

class CNNModel(tf.keras.Model):
    def __init__(self, K=40, L=20, M=10, N=100, keep_prob=0.75):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=K, kernel_size=[5,5], strides=(2,2), padding='SAME', activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=L, kernel_size=[5,5], strides=(2,2), padding='SAME', activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=M, kernel_size=[5,5], strides=(2,2), padding='SAME', activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=N, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=keep_prob)
        self.dense2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        output = self.dense2(x)
        return output
