"""Test script for queryable convolutional layer.
"""


import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np

from layers import QueryableConv2D


__author__ = "Weston Cook"
__credits__ = ["Weston Cook"]
__license__ = "Not yet licensed"
__email__ = "wwcook@ucsc.edu"
__status__ = "Prototype"
__data__ = "29 Aug. 2019"


BATCH_SIZE = 8
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
QUERY_DIMS = 128


def build_model(img_shape, query_dims):
    # Define input tensors
    x = tf.keras.Input(shape=img_shape)
    Q = tf.keras.Input(shape=[query_dims])
    # Build model
    y = QueryableConv2D([64, 8], [5, 5],
                        activation=['relu', 'relu'])([x, Q])
    y = kl.GlobalMaxPooling2D()(y)
    y = kl.Dense(10, activation='softmax')(y)

    return tf.keras.Model(inputs=[x, Q], outputs=y)


x = np.random.normal(size=(BATCH_SIZE, IMAGE_HEIGHT,
                           IMAGE_WIDTH, IMAGE_DEPTH))
Q = np.random.normal(size=(BATCH_SIZE, QUERY_DIMS))
Y = np.random.randint(0, 10, size=BATCH_SIZE)

model = build_model([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
                    QUERY_DIMS)
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model([x, Q]).shape)

model.fit([x, Q], Y, epochs=10)
