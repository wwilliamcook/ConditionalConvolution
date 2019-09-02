"""Dynamic pix2pix using queryable convolutions.
"""

import tensorflow as tf
import tensorflow.keras.layers as kl  # pylint: disable=import-error
import numpy as np

from layers import QueryableConv2D


__author__ = "Weston Cook"
__license__ = "Not yet licensed"
__email__ = "wwcook@ucsc.edu"
__status__ = "Prototype"
__data__ = "31 Aug. 2019"


def build_models(image_shape, vocab_size, embedding_dims, lstm_units, bidirectional_lstm=False):
    """Builds a model that takes an image and a sequence of vocab indices and generates an image.

    Args:
        image_shape: 3-tuple/list [image height, image width, image depth].
        vocab_size: integer specifying the size of the input vocabulary.
        embedding_dims: integer specifying the size of the vocab embeddings.
        lstm_units: integer/list specifying units of LSTM layer(s).
        bidirectional_lstm: boolean use bidirectional wrapper for LSTMs.
    
    Returns:
        A tf keras model.
    """
    if len(image_shape) != 3:
        raise ValueError('image_shape must be a 3-tuple of [height, width, depth].')
    # Convert lstm_units to a list of ints if it is just one int
    try:
        assert int(lstm_units) == lstm_units
        lstm_units = [int(lstm_units)]
    except:
        pass
    # Create inputs
    image = tf.keras.Input(shape=image_shape)
    indices = tf.keras.Input(shape=[None])
    # Build text encoder
    q = kl.Embedding(vocab_size + 1, embedding_dims)(indices)  # + 1 to account for unknown tokens
    for units in lstm_units[:-1]:  # Need to return sequences for all but last LSTM layer
        if bidirectional_lstm:
            q = kl.Bidirectional(kl.LSTM(units, return_sequences=True))(q)
        else:
            q = kl.LSTM(units)(q)
    if bidirectional_lstm:
        q = kl.Bidirectional(kl.LSTM(lstm_units[-1]))(q)
    else:
        q = kl.LSTM(lstm_units[-1])(q)
    # Build image translation model
    x = [image]
    x.append(QueryableConv2D(8, [5, 5], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(12, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(16, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(24, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(32, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(48, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(64, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(96, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(128, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(192, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))
    x.append(QueryableConv2D(256, [15, 15], activation='tanh', padding='SAME')([x[-1], q]))

    y = QueryableConv2D(256, [15, 15], activation='tanh', padding='SAME')([x[-1], q]) + x[-1]
    y = QueryableConv2D(192, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-2]
    y = QueryableConv2D(128, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-3]
    y = QueryableConv2D(96, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-4]
    y = QueryableConv2D(64, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-5]
    y = QueryableConv2D(48, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-6]
    y = QueryableConv2D(32, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-7]
    y = QueryableConv2D(24, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-8]
    y = QueryableConv2D(16, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-9]
    y = QueryableConv2D(12, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-10]
    y = QueryableConv2D(8, [15, 15], activation='tanh', padding='SAME')([y, q]) + x[-11]
    y = QueryableConv2D(image_shape[2], [5, 5], activation='sigmoid', padding='SAME')([y, q])# + x[-12]

    image_transform_model = tf.keras.Model(inputs=[image, indices], outputs=y)

    generator_output = tf.keras.Input(shape=image_shape)

    # Build discriminator model
    dy = tf.concat([image, generator_output], axis=3)
    dy = QueryableConv2D(8, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(12, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(16, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(24, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(32, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(48, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(64, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(96, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(128, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(192, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = QueryableConv2D(256, [5, 5], activation='relu', padding='SAME')([dy, q])
    dy = kl.GlobalMaxPooling2D()(dy)
    dy = kl.Dense(128, activation='relu')(dy)
    dy = kl.Dense(64, activation='relu')(dy)
    dy = kl.Dense(32, activation='relu')(dy)
    dy = kl.Dense(16, activation='relu')(dy)
    dy = kl.Dense(1, activation='sigmoid')(dy)

    discriminator_model = tf.keras.Model(inputs=[image, indices, generator_output], outputs=dy)

    generator_train_model = tf.keras.Model(inputs=[image, indices], outputs=discriminator_model([image, indices, y]))

    return image_transform_model, discriminator_model, generator_train_model
