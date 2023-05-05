import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model into memory
interpreter = Interpreter('my_model.tflite')
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data
input_data = np.random.rand(1, 40, 40, 1).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Use output data for further processing or analysis
