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
import inspect
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
    section = inspect.stack()[0][3]
    _image = image
    with tf.name_scope(section):
        if config.getboolean(section, 'random_brightness'):
            _image = tf.image.random_brightness(_image, max_delta=63)
        if config.getboolean(section, 'random_saturation'):
            _image = tf.image.random_saturation(_image, lower=0.5, upper=1.5)
        if config.getboolean(section, 'random_hue'):
            _image = tf.image.random_hue(_image, max_delta=0.032)
        if config.getboolean(section, 'random_contrast'):
            _image = tf.image.random_contrast(_image, lower=0.5, upper=1.5)
        if config.getboolean(section, 'noise'):
            _image += tf.identity(tf.random_normal(_image.get_shape(), stddev=15) + 127, name='noise')
        grayscale_probability = config.getfloat(section, 'grayscale_probability')
        if grayscale_probability > 0:
            with tf.name_scope('random_grayscale'):
                image = tf.cond(tf.random_uniform([], 0, 1) < grayscale_probability, lambda: tf.tile(tf.image.rgb_to_grayscale(_image), [1] * (len(image.get_shape()) - 1) + [3]), lambda: image)
        _image = tf.clip_by_value(_image, 0, 255)
        with tf.name_scope('random_enable'):
            image = tf.cond(tf.random_uniform([], 0, 1) < config.getfloat(section, 'enable_probability'), lambda: _image, lambda: image)
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
