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
import utils.postprocess


def main():
    model = config.get('config', 'model')
    yolo = importlib.import_module('model.' + model)
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    func = getattr(importlib.import_module('utils.preprocess'), args.preprocess)
    with tf.Session() as sess:
        image_rgb, image_std = func(os.path.expanduser(os.path.expandvars(args.image)), width, height)
        builder = yolo.Builder(args, config)
        builder(tf.expand_dims(image_std, 0))
        global_step = tf.contrib.framework.get_or_create_global_step()
        model_path = tf.train.latest_checkpoint(utils.get_logdir(config))
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        tf.logging.info('global_step=%d' % sess.run(global_step))
        tensors = [builder.model.conf * tf.to_float(builder.model.conf > args.threshold), builder.model.xy_min, builder.model.xy_max]
        conf, xy_min, xy_max = sess.run([tf.check_numerics(t, t.op.name) for t in tensors])
        boxes = utils.postprocess.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.nms_threshold)
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(sess.run(image_rgb))
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
    parser.add_argument('-p', '--preprocess', default='std', help='the preprocess function')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='detection threshold')
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
