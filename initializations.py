import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32,seed=10)
    return tf.Variable(initial, name=name)
