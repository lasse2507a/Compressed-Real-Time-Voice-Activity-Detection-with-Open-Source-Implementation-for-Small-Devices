import tensorflow as tf

def compress_model(model_name):
    model = tf.keras.models.load_model(f'models/{model_name}.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8

    tf_flite_model = converter.convert()

    tf.io.write_file(f'models/{model_name}.tflite', tf_flite_model)
