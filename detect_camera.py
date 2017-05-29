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

import argparse
import configparser
import importlib
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.postprocess


def main():
    model = config.get('config', 'model')
    yolo = importlib.import_module('model.' + model)
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    preprocess = getattr(importlib.import_module('detect'), args.preprocess)
    with tf.Session() as sess:
        ph_image = tf.placeholder(tf.float32, [1, height, width, 3], name='ph_image')
        builder = yolo.Builder(args, config)
        builder(ph_image)
        global_step = tf.contrib.framework.get_or_create_global_step()
        model_path = tf.train.latest_checkpoint(utils.get_logdir(config))
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        tf.logging.info('global_step=%d' % sess.run(global_step))
        tensors = [builder.model.conf, builder.model.xy_min, builder.model.xy_max]
        tensors = [tf.check_numerics(t, t.op.name) for t in tensors]
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, image_bgr = cap.read()
                assert ret
                image_height, image_width, _ = image_bgr.shape
                scale = [image_width / builder.model.cell_width, image_height / builder.model.cell_height]
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_std = np.expand_dims(preprocess(cv2.resize(image_rgb, (width, height))).astype(np.float32), 0)
                feed_dict = {ph_image: image_std}
                conf, xy_min, xy_max = sess.run(tensors, feed_dict)
                boxes = utils.postprocess.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.threshold, args.threshold_iou)
                for _conf, _xy_min, _xy_max in boxes:
                    index = np.argmax(_conf)
                    if _conf[index] > args.threshold:
                        _xy_min = (_xy_min * scale).astype(np.int)
                        _xy_max = (_xy_max * scale).astype(np.int)
                        cv2.rectangle(image_bgr, tuple(_xy_min), tuple(_xy_max), (255, 0, 255), 3)
                        cv2.putText(image_bgr, builder.names[index] + ' (%.1f%%)' % (_conf[index] * 100), tuple(_xy_min), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('detection', image_bgr)
                cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()
            cap.release()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--preprocess', default='std', help='the preprocess function')
    parser.add_argument('-t', '--threshold', type=float, default=0.3)
    parser.add_argument('--threshold_iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
