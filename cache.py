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
import shutil
from PIL import Image
import tqdm
import numpy as np
import tensorflow as tf
import voc
import utils


def cache_voc(writer, root, names, profile, verify=False):
    path = os.path.join(root, 'ImageSets', 'Main', profile) + '.txt'
    if not os.path.exists(path):
        logger.warn(path + ' not exists')
        return
    with open(path, 'r') as f:
        filenames = [line.strip() for line in f]
    namedict = dict([(name, i) for i, name in enumerate(names)])
    for filename in tqdm.tqdm(filenames):
        imagename, imageshape, objects_class, objects_coord = voc.load_dataset(os.path.join(root, 'Annotations', filename + '.xml'), namedict)
        objects_class = np.array(objects_class, dtype=np.int64)
        objects_coord = np.array(objects_coord, dtype=np.float32)
        if len(objects_class) > 0:
            imagepath = os.path.join(root, 'JPEGImages', imagename)
            if verify:
                with Image.open(imagepath) as img:
                    width, height = img.size
                _height, _width, _ = imageshape
                assert width == _width
                assert height == _height
            example = tf.train.Example(features=tf.train.Features(feature={
                'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
                'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
                'objects': tf.train.Feature(bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
            }))
            writer.write(example.SerializeToString())
        else:
            logger.warn(filename + ' has no object')


def main():
    cachedir = utils.get_cachedir(config)
    os.makedirs(cachedir, exist_ok=True)
    path = os.path.join(cachedir, 'names')
    shutil.copyfile(os.path.expanduser(os.path.expandvars(config.get('cache', 'names'))), path)
    with open(path, 'r') as f:
        names = [line.strip() for line in f]
    for profile in args.profile:
        path = os.path.join(cachedir, profile + '.tfrecord')
        logger.info('write tfrecords file: ' + path)
        with tf.python_io.TFRecordWriter(path) as writer:
            with open(os.path.expanduser(os.path.expandvars(config.get('cache', 'voc'))), 'r') as f:
                roots = [os.path.expanduser(os.path.expandvars(line.strip())) for line in f]
            for root in roots:
                logger.info('loading VOC %s dataset from %s' % (profile, root))
                cache_voc(writer, root, names, profile, args.verify)
    logger.info('%s data are saved into %s' % (str(args.profile), cachedir))
    


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('-v', '--verify', action='store_true')
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
