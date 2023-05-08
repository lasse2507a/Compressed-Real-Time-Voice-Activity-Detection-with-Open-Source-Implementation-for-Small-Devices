import tensorflow as tf


def compress_model(model_name):
    model = tf.keras.models.load_model(f'models/{model_name}.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tf_flite_model = converter.convert()

    tf.io.write_file(f'models/{model_name}.tflite', tf_flite_model)

if __name__ == '__main__':
    compress_model('cnn_model_v4_25(12,8,5)')
