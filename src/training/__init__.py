import os
import numpy as np
import tensorflow as tf
import visualkeras
from keras.utils import plot_model
from training.cnn_model import CNNModel

def execute_training(data_path):
    training_data = np.empty((0, 40, 40))
    training_labels = np.empty((0,))
    for file in os.listdir(data_path):
        file_data = np.load(os.path.join('data\\output\\training_clip_len_17200samples\\mfsc_window_400samples', file))
        training_data = np.concatenate([training_data, file_data], axis=0)
        label = int(file.split("_")[-2])
        training_labels = np.concatenate(training_labels, label)

    model = CNNModel(K=40, L=20, M=10, N=100, classes=2, div=10, batch_size=25000, keep_prob=0.75, learning_rate=np.hstack((1e-3*np.ones(6),
                                                                                                                            1e-4*np.ones(4),
                                                                                                                            1e-5*np.ones(2))))

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, 'models\\model_1\\checkpoints', max_to_keep=5)

    # Train the model
    num_epochs = 12
    batch_size = 25000
    num_batches = len(training_data) // batch_size
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            x_batch = training_data[batch*batch_size:(batch+1)*batch_size]
            y_batch = training_labels[batch*batch_size:(batch+1)*batch_size]

            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = model.loss_fn(y_batch, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            model.metric.update_state(y_batch, logits)

        # Print the loss and accuracy for the epoch
        print(f'Epoch {epoch+1}/{num_epochs}: Loss={loss.numpy()}, Accuracy={model.metric.result().numpy()}')

        # Save the model's weights and optimizer state to disk
        manager.save()

        # Reset the metric for the next epoch
        model.metric.reset_states()

def visualize_model():
    model = CNNModel(K=40, L=20, M=10, N=100, classes=2, div=10, batch_size=25000, keep_prob=0.75, learning_rate=np.hstack((1e-3*np.ones(6),
                                                                                                                            1e-4*np.ones(4),
                                                                                                                            1e-5*np.ones(2))))
    visualkeras.layered_view(model, to_file='model_architecture.png').show()
