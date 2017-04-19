"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import math
import logging
import getpass
import numpy as np
import tensorflow as tf
import sympy


def make_logger(level, fmt):
    logger = logging.getLogger(getpass.getuser())
    logger.setLevel(level)
    formatter = logging.Formatter(fmt)
    settings = [
        (logging.INFO, sys.stdout),
        (logging.WARN, sys.stderr),
    ]
    for level, out in settings:
        handler = logging.StreamHandler(out)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def data_augmentation(image, labels, config):
    if config.getboolean('data_augmentation', 'random_brightness'):
        image = tf.image.random_brightness(image, max_delta=63)
    if config.getboolean('data_augmentation', 'random_saturation'):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if config.getboolean('data_augmentation', 'random_hue'):
        image = tf.image.random_hue(image, max_delta=0.032)
    if config.getboolean('data_augmentation', 'random_contrast'):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0, 255)
    image = tf.image.per_image_standardization(image)
    if config.getboolean('data_augmentation', 'noise'):
        image += tf.random_normal(image.get_shape(), stddev=.2)
    return image, labels


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def get_factor2(x):
    factors = sympy.divisors(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    else:
        i = len(factors) // 2
        return factors[i], factors[i]


def calc_pooled_size(size, pooling):
    for ksize in pooling:
        if ksize > 0:
            size = math.ceil(float(size) / ksize)
    return size
