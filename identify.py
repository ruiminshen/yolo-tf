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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow.contrib.slim as slim
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


def non_max_suppress(conf, xy_min, xy_max, threshold=.4):
    _, _, classes = conf.shape
    boxes = [(_conf, _xy_min, _xy_max) for _conf, _xy_min, _xy_max in zip(conf.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]
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


def read_image(path, width, height, sess):
    imagefile = tf.read_file(path)
    image_rgb = tf.image.decode_jpeg(imagefile, channels=3)
    image_rgb = tf.image.resize_images(image_rgb, [height, width])
    image_std = tf.image.per_image_standardization(image_rgb)
    return sess.run(tf.cast(image_rgb, tf.uint8)), image_std


def main():
    model = config.get('config', 'model')
    yolo = importlib.import_module(model)
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    with tf.Session() as sess:
        image_rgb, image_std = read_image(os.path.expanduser(os.path.expandvars(args.image)), width, height, sess)
        builder = yolo.Builder(args, config)
        builder(tf.expand_dims(image_std, 0))
        global_step = tf.contrib.framework.get_or_create_global_step()
        model_path = tf.train.latest_checkpoint(utils.get_logdir(config))
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        tf.logging.info('global_step=%d' % sess.run(global_step))
        conf, xy_min, xy_max = sess.run([builder.model.conf * tf.to_float(builder.model.conf > args.threshold), builder.model.xy_min, builder.model.xy_max])
        boxes = non_max_suppress(conf[0], xy_min[0], xy_max[0], args.nms_threshold)
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image_rgb)
        cnt = 0
        for _conf, _xy_min, _xy_max in boxes:
            index = np.argmax(_conf)
            if _conf[index] > args.threshold:
                wh = _xy_max - _xy_min
                _xy_min = _xy_min * [width, height] / [builder.model.cell_width, builder.model.cell_height]
                _wh = wh * [width, height] / [builder.model.cell_width, builder.model.cell_height]
                linewidth = min(_conf[index] * 10, 3)
                ax.add_patch(patches.Rectangle(_xy_min, _wh[0], _wh[1], linewidth=linewidth, edgecolor=args.color, facecolor='none'))
                ax.annotate(builder.names[index] + ' (%.1f%%)' % (_conf[index] * 100), _xy_min, color=args.color)
                cnt += 1
        fig.canvas.set_window_title('%d objects detected' % cnt)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='input image')
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-t', '--threshold', type=float, default=0.1, help='detection threshold')
    parser.add_argument('-n', '--nms_threshold', type=float, default=0.4, help='non-max suppress threshold')
    parser.add_argument('--color', default='red', help='bounding box and font color')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    assert os.path.exists(args.config)
    config.read(args.config)
    if args.level:
        tf.logging.set_verbosity(eval('tf.logging.' + args.level.upper()))
    main()
