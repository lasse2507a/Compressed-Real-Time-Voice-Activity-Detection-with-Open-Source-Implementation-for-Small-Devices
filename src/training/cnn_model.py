import tensorflow as tf

class CNNModel(tf.keras.Model):
    def __init__(self, K, L, M, N, classes, div, batch_size, keep_prob, learning_rate):
        super(CNNModel, self).__init__()
        self.K = K
        self.L = L
        self.M = M
        self.N = N
        self.classes = classes
        self.div = div
        self.batch_size = batch_size

        with tf.name_scope("hyperparameters"):
            self.learning_rate = tf.Variable(tf.cast(learning_rate, dtype=tf.float32))
            self.keep_prob = tf.Variable(keep_prob)

        with tf.name_scope("inputs"):
            self.x = tf.Variable(initial_value=tf.zeros([self.batch_size, 40, 40]), shape=tf.TensorShape([self.batch_size, 40, 40]), name="x-input")
            self.x_image = tf.reshape(self.x, [-1, 40, 40, 1], name="x-image")
            self.Y_ = tf.Variable(initial_value=tf.zeros([self.batch_size, 40, 40]), shape=tf.TensorShape([self.batch_size, 40, 40]), name="y-input")

        with tf.name_scope("model"):
            self.W1 = tf.Variable(tf.random.truncated_normal([5,5,1,self.K], stddev=0.05))
            self.B1 = tf.Variable(tf.ones([self.K])/self.div)
            self.Y1 = tf.nn.relu(tf.nn.conv2d(self.x_image, self.W1, strides=[1,2,2,1], padding='SAME') + self.B1)

            self.W2 = tf.Variable(tf.random.truncated_normal([5,5,self.K,self.L], stddev=0.05))
            self.B2 = tf.Variable(tf.ones([self.L])/self.div)
            self.Y2 = tf.nn.relu(tf.nn.conv2d(self.Y1, self.W2, strides=[1,2,2,1], padding='SAME') + self.B2)

            self.W3 = tf.Variable(tf.random.truncated_normal([5,5,self.L,self.M], stddev=0.05))
            self.B3 = tf.Variable(tf.ones([self.M])/self.div)
            self.Y3 = tf.nn.relu(tf.nn.conv2d(self.Y2, self.W3, strides=[1,2,2,1], padding='SAME') + self.B3)

            self.YY = tf.reshape(self.Y3, shape=[-1, 5*5*self.M])

            self.W4 = tf.Variable(tf.random.truncated_normal([5*5*self.M, self.N], stddev=0.05))
            self.B4 = tf.Variable(tf.ones([self.N])/self.div)
            self.Yf = tf.nn.relu(tf.matmul(self.YY, self.W4) + self.B4)
            self.Y4 = tf.nn.dropout(self.Yf, self.keep_prob)

            self.W5 = tf.Variable(tf.random.truncated_normal([self.N, 2], stddev=0.05))
            self.B5 = tf.Variable(tf.ones([2])/self.div)
            self.Y = tf.nn.softmax(tf.matmul(self.Y4, self.W5) + self.B5)

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def call(self, inputs, training=None, mask=None):
        self.x.assign(inputs['x'])
        self.Y_.assign(inputs['y'])
        self.learning_rate.assign(inputs['learning_rate'])
        self.keep_prob.assign(inputs['keep_prob'])

        cross_entropy = -tf.reduce_sum(self.Y_*tf.math.log(tf.clip_by_value(self.Y, 1e-10, 1.0)))
        is_correct = tf.equal(tf.argmax(self.Y,1), tf.argmax(self.Y_,1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        if training:
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            optimizer.minimize(cross_entropy, var_list=self.trainable_variables)
            return cross_entropy, accuracy
        else:
            return cross_entropy, accuracy
