import tensorflow as tf
import visualkeras


def visualize_model(model_name):
    model = tf.keras.models.load_model(f'models\\{model_name}')
    tf.keras.utils.plot_model(model, to_file=f'images\\{model_name.split(".")[0]}_table.png', show_shapes=True, show_layer_names=True)
    visualkeras.layered_view(model, legend=True, scale_z=1, scale_xy=9, to_file=f'images\\{model_name.split(".")[0]}_figure.png')


if __name__ == '__main__':
    visualize_model('cnn_model_original_25(12,8,5).h5')
