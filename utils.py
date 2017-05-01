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
import os
import re
import math
import importlib
import configparser
import logging
import getpass
import inspect
import numpy as np
import matplotlib.patches as patches
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


def get_cachedir(config):
    model = config.get('config', 'model')
    return os.path.join(os.path.expanduser(os.path.expandvars(config.get(model, 'basedir'))), 'cache')


def get_logdir(config):
    model = config.get('config', 'model')
    return os.path.join(os.path.expanduser(os.path.expandvars(config.get(model, 'basedir'))), 'logdir', config.get(model, 'inference'))


def get_inference(config):
    model = config.get('config', 'model')
    return getattr(importlib.import_module('.'.join([model, 'inference'])), config.get(model, 'inference'))


def get_downsampling(config):
    model = config.get('config', 'model')
    return getattr(importlib.import_module('.'.join([model, 'inference'])), config.get(model, 'inference').upper() + '_DOWNSAMPLING')


def decode_image_objects(example, width, height):
    with tf.name_scope(inspect.stack()[0][3]):
        imagepath = example['imagepath']
        objects = example['objects']
        with tf.name_scope('decode_objects'):
            objects_class = tf.decode_raw(objects[0], tf.int64, name='objects_class')
            objects_coord = tf.decode_raw(objects[1], tf.float32)
            objects_coord = tf.reshape(objects_coord, [-1, 4], name='objects_coord')
        with tf.name_scope('load_image'):
            imagefile = tf.read_file(imagepath)
            image_rgb = tf.image.decode_jpeg(imagefile, channels=3)
            image_rgb = tf.image.resize_images(image_rgb, [height, width])
    return image_rgb, objects_class, objects_coord


def transform_labels(objects_class, objects_coord, classes, cell_width, cell_height, dtype=np.float32):
    cells = cell_height * cell_width
    mask = np.zeros([cells, 1], dtype=dtype)
    prob = np.zeros([cells, 1, classes], dtype=dtype)
    coords = np.zeros([cells, 1, 4], dtype=dtype)
    offset_xy_min = np.zeros([cells, 1, 2], dtype=dtype)
    offset_xy_max = np.zeros([cells, 1, 2], dtype=dtype)
    assert len(objects_class) == len(objects_coord)
    for c, (xmin, ymin, xmax, ymax) in zip(objects_class, objects_coord):
        x = cell_width * (xmin + xmax) / 2
        y = cell_height * (ymin + ymax) / 2
        #assert 0 <= x <= cell_width
        #assert 0 <= y <= cell_height
        ix = math.floor(x)
        iy = math.floor(y)
        index = iy * cell_width + ix
        offset_x = x - ix
        offset_y = y - iy
        w = float(xmax - xmin)
        h = float(ymax - ymin)
        mask[index, :] = 1
        prob[index, :, c] = 1
        coords[index, 0, :] = [offset_x, offset_y, math.sqrt(w), math.sqrt(h)]
        offset_xy_min[index, 0, :] = [offset_x - w / 2 * cell_width, offset_y - h / 2 * cell_height]
        offset_xy_max[index, 0, :] = [offset_x + w / 2 * cell_width, offset_y + h / 2 * cell_height]
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


def load_image_labels(paths, classes, width, height, cell_width, cell_height, config):
    with tf.name_scope('batch'):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(tf.train.string_input_producer(paths))
        example = tf.parse_single_example(serialized, features={
            'imagepath': tf.FixedLenFeature([], tf.string),
            'objects': tf.FixedLenFeature([2], tf.string),
        })
        image_rgb, objects_class, objects_coord = decode_image_objects(example, width, height)
        if config.getboolean('data_augmentation', 'enable'):
            image_rgb, objects_coord = data_augmentation(image_rgb, objects_coord, config)
        with tf.device('/cpu:0'):
            labels = decode_labels(objects_class, objects_coord, classes, cell_width, cell_height)
    return image_rgb, labels


def draw_labels(ax, names, width, height, cell_width, cell_height, mask, prob, coords, xy_min, xy_max, areas, color='red', rtol=1e-3):
    plots = []
    for i, (_mask, _prob, _coords, _xy_min, _xy_max, _areas) in enumerate(zip(mask, prob, coords, xy_min, xy_max, areas)):
        _mask = _mask.reshape([])
        _coords = _coords.reshape([-1])
        if np.any(_mask) > 0:
            iy = i // cell_width
            ix = i % cell_width
            plots.append(ax.add_patch(patches.Rectangle((ix * width / cell_width, iy * height / cell_height), width / cell_width, height / cell_height, linewidth=0, facecolor=color, alpha=.2)))
            name = names[np.argmax(_prob)]
            #check coords
            offset_x, offset_y, _w_sqrt, _h_sqrt = _coords
            cell_x, cell_y = ix + offset_x, iy + offset_y
            x, y = cell_x * width / cell_width, cell_y * height / cell_height
            _w, _h = _w_sqrt * _w_sqrt, _h_sqrt * _h_sqrt
            w, h = _w * width, _h * height
            x_min, y_min = x - w / 2, y - h / 2
            plots.append(ax.add_patch(patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=color, facecolor='none')))
            plots.append(ax.annotate(name, (x_min, y_min), color=color))
            #check offset_xy_min and xy_max
            wh = _xy_max - _xy_min
            assert np.all(wh >= 0)
            #np.testing.assert_allclose(wh / [cell_width, cell_height], [_w, _h], rtol=rtol)
            #np.testing.assert_allclose(_xy_min + wh / 2, [offset_x, offset_y], rtol=rtol)
    return plots


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def match_trainable_variables(pattern):
    prog = re.compile(pattern)
    return [v for v in tf.trainable_variables() if prog.match(v.op.name)]


def match_tensor(pattern):
    prog = re.compile(pattern)
    return [op.values()[0] for op in tf.get_default_graph().get_operations() if op.values() and prog.match(op.name)]


def tensorboard_histogram(pattern):
    try:
        for t in match_tensor(pattern):
            tf.summary.histogram(t.op.name, t)
    except configparser.NoOptionError:
        pass


def get_factor2(x):
    factors = sympy.divisors(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    else:
        i = len(factors) // 2
        return factors[i], factors[i]
