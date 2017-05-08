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


def get_cachedir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    name = os.path.basename(config.get('cache', 'names'))
    return os.path.join(basedir, 'cache', name)


def get_logdir(config):
    basedir = os.path.expanduser(os.path.expandvars(config.get('config', 'basedir')))
    model = config.get('config', 'model')
    inference = config.get(model, 'inference')
    name = os.path.basename(config.get('cache', 'names'))
    return os.path.join(basedir, model, inference, name)


def get_inference(config):
    model = config.get('config', 'model')
    return getattr(importlib.import_module('.'.join([model, 'inference'])), config.get(model, 'inference'))


def get_downsampling(config):
    model = config.get('config', 'model')
    return getattr(importlib.import_module('.'.join([model, 'inference'])), config.get(model, 'inference').upper() + '_DOWNSAMPLING')


def calc_cell_width_height(config, width, height):
    downsampling_width, downsampling_height = get_downsampling(config)
    assert width % downsampling_width == 0
    assert height % downsampling_height == 0
    return width // downsampling_width, height // downsampling_height


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


def resize_image_objects(image, imageshape, objects_coord, width, height):
    with tf.name_scope(inspect.stack()[0][3]):
        image = tf.image.resize_images(image, [height, width])
        factor = tf.cast([width, height] / imageshape[1::-1], objects_coord.dtype)
        objects_coord = objects_coord * tf.tile(factor, [2])
    return image, objects_coord


def data_augmentation(image, objects_coord, width, height, config):
    section = inspect.stack()[0][3]
    _image = image
    _objects_coord = tf.cast(objects_coord, tf.int32)
    with tf.name_scope(section):
        if config.getboolean(section, 'random_brightness'):
            _image = tf.image.random_brightness(_image, max_delta=63)
        if config.getboolean(section, 'random_saturation'):
            _image = tf.image.random_saturation(_image, lower=0.5, upper=1.5)
        if config.getboolean(section, 'random_hue'):
            _image = tf.image.random_hue(_image, max_delta=0.032)
        if config.getboolean(section, 'random_contrast'):
            _image = tf.image.random_contrast(_image, lower=0.5, upper=1.5)
        grayscale_probability = config.getfloat(section, 'grayscale_probability')
        if grayscale_probability > 0:
            with tf.name_scope('random_grayscale'):
                _image = tf.cond(tf.random_uniform([], 0, 1) < grayscale_probability, lambda: tf.tile(tf.image.rgb_to_grayscale(_image), [1] * (len(image.get_shape()) - 1) + [3]), lambda: image)
        if config.getboolean(section, 'random_move'):
            with tf.name_scope('random_move'):
                xy_min = tf.reduce_min(objects_coord[:, :2], 0)
                xy_max = tf.reduce_max(objects_coord[:, 2:], 0)
                a = xy_max - [width, height]
                b = xy_min
                xy_move = a + tf.random_uniform([2]) * (b - a)
                _xy_move = tf.cast(xy_move, tf.int32)
                _xy_min = tf.maximum(_xy_move, [0, 0])
                _xy_max = tf.minimum([width, height] + _xy_move, [width, height])
                _wh = _xy_max - _xy_min
                _image = tf.image.crop_to_bounding_box(_image, _xy_min[1], _xy_min[0], _wh[1], _wh[0])
                _xy_move = tf.maximum(-_xy_move, tf.zeros([2], dtype=tf.int32))
                _image = tf.image.pad_to_bounding_box(_image, _xy_move[1], _xy_move[0], height, width)
                objects_coord = objects_coord - tf.tile(xy_move, [2])
        _image = tf.clip_by_value(_image, 0, 255)
        with tf.name_scope('random_enable'):
            image = tf.cond(tf.random_uniform([], 0, 1) < config.getfloat(section, 'enable_probability'), lambda: _image, lambda: image)
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
        image_rgb, imageshape, objects_class, objects_coord = decode_image_objects(paths)
        image_rgb, objects_coord = resize_image_objects(image_rgb, imageshape, objects_coord, width, height)
        if config.getboolean('data_augmentation', 'enable'):
            image_rgb, objects_coord = data_augmentation(image_rgb, objects_coord, width, height, config)
        objects_coord = objects_coord / tf.cast([width, height, width, height], objects_coord.dtype)
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
            np.testing.assert_allclose(wh / [cell_width, cell_height], [[_w, _h]], rtol=rtol)
            np.testing.assert_allclose(_xy_min + wh / 2, [[offset_x, offset_y]], rtol=rtol)
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


def get_factor2(x):
    factors = sympy.divisors(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    else:
        i = len(factors) // 2
        return factors[i], factors[i]
