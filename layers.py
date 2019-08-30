"""Implementation of a queryable convolutional layer in Tensorflow 2.0 Keras.
"""


import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np


__author__ = "Weston Cook"
__credits__ = ["Weston Cook"]
__license__ = "Not yet licensed"
__email__ = "wwcook@ucsc.edu"
__status__ = "Prototype"
__data__ = "29 Aug. 2019"


class QueryableConv2D(kl.Layer):
    """Implements a "queryable convolution" layer.

    A queryable convolution is identical to a standard
    convolution except that an additional "query" vector
    is convatenated depthwise onto the image patches tensor,
    as well as the addition of hidden layers.
    """
    def __init__(self, units, ksizes, strides=None, rates=None,
                 padding='VALID', activation=None):
        """Constructs a "queryable convolution" layer.

        Args:
            units: Integer specifying the depth of the output image.
            ksize: [height, width].
            strides: [height, width].
            rates: [height, width].
            padding: 'VALID' or 'SAME'.
            activation: Activation to use for output layer.
            hidden_units: List of integers specifying depths of hidden layers.
            hidden_activations: List of activations to use for hidden layers.
        """
        super(QueryableConv2D, self).__init__()
        if strides is None:
            strides = [1, 1, 1, 1]
        if rates is None:
            rates = [1, 1, 1, 1]
        if np.shape(units) == ():
            units = [units]
        self.units = units
        if np.shape(activation) == ():
            activation = [activation]
        else:
            if len(activation) != len(units):
                raise ValueError('hidden_units and hidden_activations must have the same length.')
        self.activations = [kl.Activation(a) for a in activation]
        self.ksizes = [1, ksizes[0], ksizes[1], 1]
        self.strides = strides
        self.rates = rates
        self.padding = padding

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('Invalid arguments: expected [images, queries].')
        im_batch_size, im_h, im_w, im_d = input_shape[0]
        q_batch_size, self.queries_d = input_shape[1]
        if im_batch_size != q_batch_size:
            raise ValueError('images and queries must have the same batch size.')
        # Calculate the depth of the augmented patches tensor
        self.patches_d = self.ksizes[1]*self.ksizes[2]*im_d + self.queries_d
        # Create the weights and biases for each layer
        self.layer_weights = []
        W = self.add_weight(shape=[self.patches_d, self.units[0]],
                            initializer='random_normal',
                            trainable=True,
                            name='W1')
        b = self.add_weight(shape=[self.units[0]],
                            initializer='random_normal',
                            trainable=True,
                            name='b1')
        self.layer_weights.append([W, b])
        for i in range(1, len(self.units)):
            W = self.add_weight(shape=[self.units[i - 1],
                                       self.units[i]],
                                initializer='random_normal',
                                trainable=True,
                                name='W%d' % (i + 1))
            b = self.add_weight(shape=[self.units[i]],
                                initializer='random_normal',
                                trainable=True,
                                name='b%d' % (i + 1))
            self.layer_weights.append([W, b])

    def call(self, inputs):
        images, queries = inputs  # Two input tensors
        # Separate the image into patches for performing convolution
        patches = tf.image.extract_patches(images=images,
                                           sizes=self.ksizes,
                                           strides=self.strides,
                                           rates=self.rates,
                                           padding=self.padding)
        # Reshape and broadcast queries
        #queries = tf.expand_dims(tf.expand_dims(queries, 1), 1)
        reshape_shape = [1, 1, queries.shape[1]]
        queries = kl.Reshape(reshape_shape)(queries)
        queries = tf.broadcast_to(
            queries,
            tf.where([True, False, False, True],
                     tf.shape(queries), tf.shape(patches)))
        # Augment the patches with the queries
        y = tf.concat([patches, queries], axis=3)
        # Perform the matmul + bias + activation for each layer
        for ((W, b), activation) in zip(self.layer_weights, self.activations):
            y = activation(tf.einsum('bijd,de -> bije', y, W) + b)

        return y

def build_model(img_shape, query_dims):
    # Define input tensors
    x = tf.keras.Input(shape=img_shape)
    Q = tf.keras.Input(shape=[query_dims])
    # Build model
    y = QueryableConv2D([64, 8], [5, 5],
                        activation=['relu', 'relu'])([x, Q])
    return tf.keras.Model(inputs=[x, Q], outputs=y)

model = build_model([32, 32, 3], 15)

x = np.random.random(size=(8, 32, 32, 3))
Q = np.random.random(size=(8, 15))
y = model([x, Q])
print(y.shape)
