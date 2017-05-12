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

import inspect
import numpy as np
import tensorflow as tf


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def random_crop(image, objects_coord, width_height, scale=1):
    assert 0 < scale <= 1
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        xy_min = tf.reduce_min(objects_coord[:, :2], 0)
        xy_max = tf.reduce_max(objects_coord[:, 2:], 0)
        margin = width_height - xy_max
        shrink = tf.random_uniform([4], maxval=scale) * tf.concat([xy_min, margin], 0)
        _xy_min = shrink[:2]
        _wh = width_height - shrink[2:] - _xy_min
        objects_coord = objects_coord - tf.tile(_xy_min, [2])
        _xy_min_ = tf.cast(_xy_min, tf.int32)
        _wh_ = tf.cast(_wh, tf.int32)
        image = tf.image.crop_to_bounding_box(image, _xy_min_[1], _xy_min_[0], _wh_[1], _wh_[0])
    return image, objects_coord, _wh


def flip_left_right(image, objects_coord, width_height):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        image = tf.image.flip_left_right(image)
        xmin, ymin, xmax, ymax = objects_coord[:, 0:1], objects_coord[:, 1:2], objects_coord[:, 2:3], objects_coord[:, 3:4]
        width = width_height[0]
        objects_coord = tf.concat([width - xmax, ymin, width - xmin, ymax], 1)
    return image, objects_coord, width_height


def random_flip_left_right(image, objects_coord, width_height, probability=0.5):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: flip_left_right(image, objects_coord, width_height)
        fn2 = lambda: (image, objects_coord, width_height)
        image, objects_coord, width_height = tf.cond(pred, fn1, fn2)
    return image, objects_coord, width_height


def random_grayscale(image, probability=0.5):
    if probability <= 0:
        return image
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1] * (len(image.get_shape()) - 1) + [3])
        fn2 = lambda: image
        image = tf.cond(pred, fn1, fn2)
    return image
