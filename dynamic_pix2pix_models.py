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
    x.append(QueryableConv2D(8, [5, 5], use_bias=False, padding='SAME')([x[-1], q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(12, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(16, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(24, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(32, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(48, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(64, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(96, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(128, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(192, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))
    x.append(QueryableConv2D(256, [15, 15], use_bias=False, padding='SAME')([x_, q]))
    x_ = kl.LeakyReLU(kl.BatchNormalization()(x[-1]))

    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(256, [15, 15], use_bias=False, padding='SAME')([x_, q]) + x[-1]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(192, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-2]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(128, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-3]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(96, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-4]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(64, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-5]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(48, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-6]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(32, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-7]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(24, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-8]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(16, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-9]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(12, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-10]))
    y = kl.LeakyReLU()(kl.BatchNormalization()(QueryableConv2D(8, [15, 15], use_bias=False, padding='SAME')([y, q]) + x[-11]))
    y = kl.Activation('sigmoid')(QueryableConv2D(image_shape[2], [5, 5], padding='SAME')([y, q]) + x[-12])

    image_transform_model = tf.keras.Model(inputs=[image, indices], outputs=y)

    d_image = tf.keras.Input(shape=image_shape)
    d_indices = tf.keras.Input(shape=[None])
    generator_output = tf.keras.Input(shape=image_shape)

    # Build mapping discriminator model (discriminates between good and bad image mappings)
    dy = tf.concat([d_image, generator_output], axis=3)
    dy = QueryableConv2D(8, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(12, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(16, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(24, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(32, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(48, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(64, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(96, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(128, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(192, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = QueryableConv2D(256, [5, 5], padding='SAME')([dy, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = kl.GlobalMaxPooling2D()(dy)
    dy = kl.Dense(128)(dy)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = kl.Dense(64)(dy)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = kl.Dense(32)(dy)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = kl.Dense(16)(dy)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy = kl.Dense(1)(dy)

    mapping_discriminator_model = tf.keras.Model(inputs=[d_image, d_indices, generator_output], outputs=dy)

    generator_output2 = tf.keras.Input(shape=image_shape)

    # Build feasibility discriminator model (discriminates between good and bad output images)
    dy2 = QueryableConv2D(8, [5, 5], padding='SAME')([generator_output2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(12, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(16, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(24, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(32, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(48, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(64, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(96, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(128, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(192, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = QueryableConv2D(256, [5, 5], padding='SAME')([dy2, q])
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = kl.GlobalMaxPooling2D()(dy2)
    dy2 = kl.Dense(128)(dy2)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = kl.Dense(64)(dy2)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = kl.Dense(32)(dy2)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = kl.Dense(16)(dy2)
    dy = kl.LeakyReLU()(dy)
    dy = kl.Dropout(0.3)(dy)
    dy2 = kl.Dense(1)(dy2)

    feasibility_discriminator_model = tf.keras.Model(inputs=generator_output2, outputs=dy2)

    return image_transform_model, mapping_discriminator_model, feasibility_generator_train_model
