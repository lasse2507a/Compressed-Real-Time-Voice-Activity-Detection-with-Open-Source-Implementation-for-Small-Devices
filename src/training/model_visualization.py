import tensorflow as tf
import visualkeras


def visualize_model(model_name):
    model = tf.keras.models.load_model(f'models/{model_name}')
    visualkeras.layered_view(model, legend=True, to_file=f'images\\{model_name.split(".")[0]}.png').show() # Fix scaling


if __name__ == '__main__':
    visualize_model('cnn_model_v4_25(12,8,5).h5')
