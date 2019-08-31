"""Tests the dynamic pix2pix models by running a webcam to view the live input and output.
"""

import cv2 as cv
import numpy as np

from dynamic_pix2pix_models import build_models


__author__ = "Weston Cook"
__license__ = "Not yet licensed"
__email__ = "wwcook@ucsc.edu"
__status__ = "Prototype"
__data__ = "31 Aug. 2019"


IMAGE_SHAPE = [64, 64, 1]
VOCAB_SIZE = 1000


transform_model, discriminator_model = build_models(
    IMAGE_SHAPE, VOCAB_SIZE, 64, [64, 96, 96, 128, 128], True)

video_capture = cv.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    if ret:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, tuple(IMAGE_SHAPE[:2]))
        img = np.float32(img) / 255.
        tf_img = np.expand_dims(img, 2)
        reshaped = np.expand_dims(tf_img, 0)
        query = np.random.randint(0, VOCAB_SIZE,
                                  size=(1, np.random.randint(2, 11)))
        out = np.squeeze(transform_model([reshaped, query]))
        display = np.concatenate([img, out], 1)
        display = cv.resize(display, None, fx=10, fy=10,
                            interpolation=cv.INTER_CUBIC)
#        display = np.uint8(display * 255)
        cv.imshow('Display', display)
        k = cv.waitKey(1)
        if k != -1:
            break
    else:
        print('Unable to read video device.')
        break
cv.destroyAllWindows()
video_capture.release()
    
