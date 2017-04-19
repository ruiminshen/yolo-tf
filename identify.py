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
import argparse
import configparser
import importlib
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import utils


def iou(xy_min1, xy_max1, xy_min2, xy_max2):
    assert np.all(xy_min1 <= xy_max1)
    assert np.all(xy_min2 <= xy_max2)
    areas1 = np.multiply.reduce(xy_max1 - xy_min1)
    areas2 = np.multiply.reduce(xy_max2 - xy_min2)
    _xy_min = np.maximum(xy_min1, xy_min2) 
    _xy_max = np.minimum(xy_max1, xy_max2)
    _wh = np.maximum(_xy_max - _xy_min, 0)
    _areas = np.multiply.reduce(_wh)
    assert _areas <= areas1
    assert _areas <= areas2
    return _areas / np.maximum(areas1 + areas2 - _areas, 1e-10)


def non_max_suppress(prob, xy_min, xy_max, threshold=.4):
    _, _, classes = prob.shape
    boxes = [(_prob, _xy_min, _xy_max) for _prob, _xy_min, _xy_max in zip(prob.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]
    for c in range(classes):
        boxes.sort(key=lambda box: box[0][c], reverse=True)
        for i in range(len(boxes) - 1):
            box = boxes[i]
            if box[0][c] == 0:
                continue
            for _box in boxes[i + 1:]:
                if iou(box[1], box[2], _box[1], _box[2]) >= threshold:
                    _box[0][c] = 0
    return boxes


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module('model.' + section)
    basedir = os.path.expanduser(os.path.expandvars(config.get(section, 'basedir')))
    modeldir = os.path.join(basedir, 'model')
    modelpath = os.path.join(modeldir, 'model.ckpt')
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    with tf.Session() as sess:
        image_rgb = scipy.misc.imresize(scipy.misc.imread(args.image), [height, width])
        image_std = utils.per_image_standardization(image_rgb)
        image_std = np.expand_dims(image_std, 0)
        modeler = yolo.Modeler(args, config)
        modeler.param()
        image = tf.placeholder(dtype=tf.float32, shape=image_std.shape, name='image')
        modeler.eval(image)
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, name='global_step')
        tf.global_variables_initializer().run()
        logger.info('load model')
        saver = tf.train.Saver()
        saver.restore(sess, modelpath)
        logger.info('step=%d' % sess.run(global_step))
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image_rgb)
        prob, xy_min, xy_max = sess.run([modeler.model_eval.conf * tf.to_float(modeler.model_eval.conf > args.threshold), modeler.model_eval.xy_min, modeler.model_eval.xy_max], feed_dict={image: image_std})
        boxes = non_max_suppress(prob[0], xy_min[0], xy_max[0])
        for _prob, _xy_min, _xy_max in boxes:
            index = np.argmax(_prob)
            if _prob[index] > args.threshold:
                wh = _xy_max - _xy_min
                _xy_min = _xy_min * [width, height] / [modeler.cell_width, modeler.cell_height]
                _wh = wh * [width, height] / [modeler.cell_width, modeler.cell_height]
                ax.add_patch(patches.Rectangle(_xy_min, _wh[0], _wh[1], linewidth=1, edgecolor='r', facecolor='none'))
                ax.annotate(modeler.names[index], _xy_min, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='input image')
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-t', '--threshold', type=int, default=0.1, help='detection threshold')
    parser.add_argument('--seed', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    assert os.path.exists(args.config)
    config.read(args.config)
    logger = utils.make_logger(importlib.import_module('logging').__dict__[args.level.strip().upper()], config.get('logging', 'format'))
    try:
        main()
    except Exception as e:
        logger.exception('exception')
        raise e
