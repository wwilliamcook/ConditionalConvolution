"""Implementation of the CondConv layer in Tensorflow Keras.
"""


import tensorflow as tf
import tensorflow.keras.layers as kl


class CondConv(kl.Layer):
    """Wraps a Conv layer to make a conditional convolution layer.

    A conditional convolution is identical to a standard convolution except
    that the output depends on an additional batch of vectors.
    """
    def __init__(self, conv_layer, activation=None):
        """Conditional convolution layer.

        :param: conv_layer: instance of any kind of keras convolution layer
        :param: activation: activation function
        """
        super(CondConv, self).__init__()
        self._conv_layer = conv_layer
        self._dense_layer = kl.Dense(conv_layer.filters, use_bias=False)
        self._activation = kl.Activation(activation)

    def call(self, inputs):
        images, vectors = inputs  # Two input tensors

        y = self._conv_layer(images) + self._dense_layer(vectors)
        y = self._activation(y)

        return y


def WrapCondConv(conv_layer_class, *args, **kwargs):
    """Convenience function for creating a CondConv layer.

    :param: conv_layer_class: tensorflow keras class for internal Conv layer
    """
    if 'activation' in kwargs:
        activation = kwargs['activation']
        kwargs['activation'] = None
    else:
        activation = None
    return CondConv(conv_layer_class(*args, **kwargs), activation=activation)


def CondConv1D(*args, **kwargs):
    """Returns a 1D conditional convolution layer.
    """
    return WrapCondConv(kl.Conv1D, *args, **kwargs)


def CondConv2D(*args, **kwargs):
    """Returns a 2D conditional convolution layer.
    """
    return WrapCondConv(kl.Conv2D, *args, **kwargs)


def CondConv3D(*args, **kwargs):
    """Returns a 3D conditional convolution layer.
    """
    return WrapCondConv(kl.Conv3D, *args, **kwargs)


class CreativeNoise(kl.Layer):
    """Adds gaussian noise to the input.

    Designed to introduce controlled non-deterministic behavior to the input
    by allowing the magnitude of the noise to be trained while also adding
    a loss to keep the magnitude from being trained to zero.
    """
    def __init__(self, loss_rate=1.0):
        """Adds trainable gaussian noise to the input.

        :param float: loss_rate: amount to scale nonzero-noise loss by
        """
        super(CreativeNoise, self).__init__()
        self._loss_rate = loss_rate

    def build(self, input_shape):
        # Add a trainable weight to multiply the noise by
        self._stddev = self.add_weight(shape=input_shape[1:],
                                       initializer='ones',
                                       trainable=True)

    def call(self, inputs):
        # Create a loss to keep the noise from being removed completely
        self.add_loss(-self._loss_rate * tf.reduce_sum(
            tf.math.log(tf.math.abs(self._stddev))))

        noise = tf.random.normal(tf.shape(inputs)) * self._stddev

        return inputs + noise
