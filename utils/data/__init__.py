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

import os
import re
import importlib
import inspect
import numpy as np
import matplotlib.patches as patches
import tensorflow as tf
import sympy
from .. import preprocess


def decode_image_objects(paths):
    with tf.name_scope(inspect.stack()[0][3]):
        with tf.name_scope('parse_example'):
            reader = tf.TFRecordReader()
            _, serialized = reader.read(tf.train.string_input_producer(paths))
            example = tf.parse_single_example(serialized, features={
                'imagepath': tf.FixedLenFeature([], tf.string),
                'imageshape': tf.FixedLenFeature([3], tf.int64),
                'objects': tf.FixedLenFeature([2], tf.string),
            })
        imagepath = example['imagepath']
        objects = example['objects']
        with tf.name_scope('decode_objects'):
            objects_class = tf.decode_raw(objects[0], tf.int64, name='objects_class')
            objects_coord = tf.decode_raw(objects[1], tf.float32)
            objects_coord = tf.reshape(objects_coord, [-1, 4], name='objects_coord')
        with tf.name_scope('load_image'):
            imagefile = tf.read_file(imagepath)
            image = tf.image.decode_jpeg(imagefile, channels=3)
    return image, example['imageshape'], objects_class, objects_coord


def data_augmentation_full(image, objects_coord, width_height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        random_crop = config.getfloat(section, 'random_crop')
        if random_crop > 0:
            image, objects_coord, width_height = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: preprocess.random_crop(image, objects_coord, width_height, random_crop),
                lambda: (image, objects_coord, width_height)
            )
    return image, objects_coord, width_height


def resize_image_objects(image, objects_coord, width_height, width, height):
    with tf.name_scope(inspect.stack()[0][3]):
        image = tf.image.resize_images(image, [height, width])
        factor = [width, height] / width_height
        objects_coord = objects_coord * tf.tile(factor, [2])
    return image, objects_coord


def data_augmentation_resized(image, objects_coord, width, height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        if config.getboolean(section, 'random_flip_left_right'):
            image, objects_coord = preprocess.random_flip_left_right(image, objects_coord, width)
        if config.getboolean(section, 'random_brightness'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_brightness(image, max_delta=63),
                lambda: image
            )
        if config.getboolean(section, 'random_saturation'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_saturation(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'random_hue'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_hue(image, max_delta=0.032),
                lambda: image
            )
        if config.getboolean(section, 'random_contrast'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'noise'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: image + tf.truncated_normal(image.get_shape()) * tf.random_uniform([], 5, 15),
                lambda: image
            )
        grayscale_probability = config.getfloat(section, 'grayscale_probability')
        if grayscale_probability > 0:
            image = preprocess.random_grayscale(image, grayscale_probability)
    return image, objects_coord


def transform_labels(objects_class, objects_coord, classes, cell_width, cell_height, dtype=np.float32):
    cells = cell_height * cell_width
    mask = np.zeros([cells, 1], dtype=dtype)
    prob = np.zeros([cells, 1, classes], dtype=dtype)
    coords = np.zeros([cells, 1, 4], dtype=dtype)
    offset_xy_min = np.zeros([cells, 1, 2], dtype=dtype)
    offset_xy_max = np.zeros([cells, 1, 2], dtype=dtype)
    assert len(objects_class) == len(objects_coord)
    xmin, ymin, xmax, ymax = objects_coord.T
    x = cell_width * (xmin + xmax) / 2
    y = cell_height * (ymin + ymax) / 2
    ix = np.floor(x)
    iy = np.floor(y)
    offset_x = x - ix
    offset_y = y - iy
    w = xmax - xmin
    h = ymax - ymin
    index = (iy * cell_width + ix).astype(np.int)
    mask[index, :] = 1
    prob[index, :, objects_class] = 1
    coords[index, 0, 0] = offset_x
    coords[index, 0, 1] = offset_y
    coords[index, 0, 2] = np.sqrt(w)
    coords[index, 0, 3] = np.sqrt(h)
    _w = w / 2 * cell_width
    _h = h / 2 * cell_height
    offset_xy_min[index, 0, 0] = offset_x - _w
    offset_xy_min[index, 0, 1] = offset_y - _h
    offset_xy_max[index, 0, 0] = offset_x + _w
    offset_xy_max[index, 0, 1] = offset_y + _h
    wh = offset_xy_max - offset_xy_min
    assert np.all(wh >= 0)
    areas = np.multiply.reduce(wh, -1)
    return mask, prob, coords, offset_xy_min, offset_xy_max, areas


def decode_labels(objects_class, objects_coord, classes, cell_width, cell_height):
    with tf.name_scope(inspect.stack()[0][3]):
        mask, prob, coords, offset_xy_min, offset_xy_max, areas = tf.py_func(transform_labels, [objects_class, objects_coord, classes, cell_width, cell_height], [tf.float32] * 6)
        cells = cell_height * cell_width
        with tf.name_scope('reshape_labels'):
            mask = tf.reshape(mask, [cells, 1], name='mask')
            prob = tf.reshape(prob, [cells, 1, classes], name='prob')
            coords = tf.reshape(coords, [cells, 1, 4], name='coords')
            offset_xy_min = tf.reshape(offset_xy_min, [cells, 1, 2], name='offset_xy_min')
            offset_xy_max = tf.reshape(offset_xy_max, [cells, 1, 2], name='offset_xy_max')
            areas = tf.reshape(areas, [cells, 1], name='areas')
    return mask, prob, coords, offset_xy_min, offset_xy_max, areas


def load_image_labels(paths, classes, width, height, cell_width, cell_height, config):
    with tf.name_scope('batch'):
        image, imageshape, objects_class, objects_coord = decode_image_objects(paths)
        image = tf.cast(image, tf.float32)
        width_height = tf.cast(imageshape[1::-1], tf.float32)
        if config.getboolean('data_augmentation_full', 'enable'):
            image, objects_coord, width_height = data_augmentation_full(image, objects_coord, width_height, config)
        image, objects_coord = resize_image_objects(image, objects_coord, width_height, width, height)
        if config.getboolean('data_augmentation_resized', 'enable'):
            image, objects_coord = data_augmentation_resized(image, objects_coord, width, height, config)
        image = tf.clip_by_value(image, 0, 255)
        objects_coord = objects_coord / [width, height, width, height]
        with tf.device('/cpu:0'):
            labels = decode_labels(objects_class, objects_coord, classes, cell_width, cell_height)
    return image, labels
