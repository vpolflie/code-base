"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

# Internal imports


class ExpandAs(layers.Layer):
    def __init__(self, name, n_repeats, axis, **kwargs):
        super(ExpandAs, self).__init__(name=name, **kwargs)
        self.n_repeats = n_repeats
        self.axis = axis

    def build(self, input_shape):
        self.expander = layers.Lambda(
            lambda x, reps:
            K.repeat_elements(x, reps, axis=self.axis),
            arguments={'reps': self.n_repeats},
            name=self.name + '_Lambda'
        )

    def call(self, inputs):
        return self.expander(inputs)


def expand_as(tensor, rep, ax, name):
    repeated_tensor = tf.keras.layers.Lambda(
        lambda x, repnum:
        K.repeat_elements(x, repnum, axis=ax),
        arguments={'repnum': rep},
        name=name
    )(tensor)
    return repeated_tensor


def set_tf_config():
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
