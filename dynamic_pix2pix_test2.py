"""Trains a simple dynamic pix2pix model using RL and live user feedback.
"""

import tensorflow as tf
import cv2 as cv
import numpy as np

from dynamic_pix2pix_models import build_models
from tokenizers import DynamicTextToIndices


__author__ = "Weston Cook"
__license__ = "Not yet licensed"
__email__ = "wwcook@ucsc.edu"
__status__ = "Prototype"
__data__ = "2 Sep. 2019"


IMAGE_SHAPE = [28, 28, 1]
VOCAB_SIZE = 1000
MAX_SAMPLE_COUNT = 1000
TRAINING_FREQUENCY = 5  # Train after every <TRAINING_FREQUENCY> samples

transform_model, discriminator_model, generator_train_model = build_models(
    IMAGE_SHAPE, VOCAB_SIZE, 1, [1], True)
discriminator_model.compile('adam', 'sparse_categorical_crossentropy')
generator_train_model.compile('adam', 'mse')

text_to_indices = DynamicTextToIndices(VOCAB_SIZE)

queries = []
outputs = np.zeros([0] + list(IMAGE_SHAPE))
ratings = np.zeros(0)
sample_count = 0

sample_gather_index = 1
while True:
    print('Gathering sample {}...'.format(sample_gather_index))
    sample_gather_index += 1

    input_img = np.expand_dims(np.zeros(IMAGE_SHAPE, dtype=np.float32), 0)
    try:
        query = text_to_indices(input('Enter command: '), training=True)
    except KeyboardInterrupt:
        print('Closing up.')
        break
    output_img = np.squeeze(transform_model([input_img, np.expand_dims(query, 0)]), axis=0)
    cv.imshow('Output', cv.resize(np.squeeze(output_img), None, fx=10, fy=10,
                                  interpolation=cv.INTER_NEAREST))
    k = cv.waitKey(0)
    cv.destroyWindow('Output')
    if k in [27, ord('q')]:  # Esc, 'q'
        print('Closing up.')
        break
    while True:
        try:
            rating = float(input('Enter rating on range [0, 10]: ')) / 10
            break
        except:
            print('Invalid rating. Try again.')
        if rating < 0 or rating > 1:
            print('Invalid rating. Try again.')
    if sample_count < MAX_SAMPLE_COUNT:
        # Add the samples to the storage
        queries.append(query)
        outputs = np.concatenate([outputs, [output_img]], 0)
        ratings = np.concatenate([ratings, [rating]], 0)
        sample_count += 1
    else:
        # Replace a random sample with the new sample
        index = np.random.randint(0, sample_count)
        queries[index] = query
        outputs[index] = output_img
        ratings[index] = rating
    if sample_count and sample_count % TRAINING_FREQUENCY == 0:
        inputs = np.zeros([sample_count] + list(IMAGE_SHAPE))
        queries_tf = tf.convert_to_tensor(queries, dtype=tf.int32)
        # Train the discriminator/critic
        discriminator_model.trainable = True
        discriminator_model.fit([inputs, queries_tf, outputs], ratings)
        # Train the generator
        discriminator_model.trainable = False
        generator_train_model.fit([inputs, queries_tf], np.ones(sample_count))
cv.destroyAllWindows()
