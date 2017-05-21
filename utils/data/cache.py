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
import inspect
from PIL import Image
import tqdm
import numpy as np
import tensorflow as tf
import utils.data.voc


def verify_imageshape(imagepath, imageshape):
    with Image.open(imagepath) as image:
        return np.all(np.equal(image.size, imageshape[1::-1]))


def verify_image_jpeg(imagepath, imageshape):
    scope = inspect.stack()[0][3]
    try:
        graph = tf.get_default_graph()
        path = graph.get_tensor_by_name(scope + '/path:0')
        decode = graph.get_tensor_by_name(scope + '/decode_jpeg:0')
    except KeyError:
        tf.logging.debug('creating decode_jpeg tensor')
        path = tf.placeholder(tf.string, name=scope + '/path')
        imagefile = tf.read_file(path, name=scope + '/read_file')
        decode = tf.image.decode_jpeg(imagefile, channels=3, name=scope + '/decode_jpeg')
    try:
        image = tf.get_default_session().run(decode, {path: imagepath})
    except:
        return False
    return np.all(np.equal(image.shape[:2], imageshape[:2]))


def check_coords(objects_coord):
    return np.all(objects_coord[:, 0] <= objects_coord[:, 2]) and np.all(objects_coord[:, 1] <= objects_coord[:, 3])


def verify_coords(objects_coord, imageshape):
    assert check_coords(objects_coord)
    return np.all(objects_coord >= 0) and np.all(objects_coord <= np.tile(imageshape[1::-1], [2]))


def fix_coords(objects_coord, imageshape):
    assert check_coords(objects_coord)
    objects_coord = np.maximum(objects_coord, np.zeros([4], dtype=objects_coord.dtype))
    objects_coord = np.minimum(objects_coord, np.tile(np.asanyarray(imageshape[1::-1], objects_coord.dtype), [2]))
    return objects_coord


def voc(writer, name_index, profile, row, verify=False):
    root = os.path.expanduser(os.path.expandvars(row['root']))
    path = os.path.join(root, 'ImageSets', 'Main', profile) + '.txt'
    if not os.path.exists(path):
        tf.logging.warn(path + ' not exists')
        return False
    with open(path, 'r') as f:
        filenames = [line.strip() for line in f]
    annotations = [os.path.join(root, 'Annotations', filename + '.xml') for filename in filenames]
    _annotations = list(filter(os.path.exists, annotations))
    if len(annotations) > len(_annotations):
        tf.logging.warn('%d of %d images not exists' % (len(annotations) - len(_annotations), len(annotations)))
    cnt_noobj = 0
    for path in tqdm.tqdm(_annotations):
        imagename, imageshape, objects_class, objects_coord = utils.data.voc.load_dataset(path, name_index)
        if len(objects_class) <= 0:
            cnt_noobj += 1
            continue
        objects_class = np.array(objects_class, dtype=np.int64)
        objects_coord = np.array(objects_coord, dtype=np.float32)
        imagepath = os.path.join(root, 'JPEGImages', imagename)
        if verify:
            if not verify_coords(objects_coord, imageshape):
                tf.logging.error('failed to verify coordinates of ' + imagepath)
                continue
            if not verify_image_jpeg(imagepath, imageshape):
                tf.logging.error('failed to decode ' + imagepath)
                continue
        assert len(objects_class) == len(objects_coord)
        example = tf.train.Example(features=tf.train.Features(feature={
            'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
            'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
            'objects': tf.train.Feature(bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
        }))
        writer.write(example.SerializeToString())
    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images have no object' % (cnt_noobj, len(filenames)))
    return True


def coco(writer, name_index, profile, row, verify=False):
    root = os.path.expanduser(os.path.expandvars(row['root']))
    year = str(row['year'])
    name = profile + year
    path = os.path.join(root, 'annotations', 'instances_%s.json' % name)
    if not os.path.exists(path):
        tf.logging.warn(path + ' not exists')
        return False
    import pycocotools.coco
    coco = pycocotools.coco.COCO(path)
    catIds = coco.getCatIds(catNms=list(name_index.keys()))
    cats = coco.loadCats(catIds)
    id_index = dict((cat['id'], name_index[cat['name']]) for cat in cats)
    imgIds = coco.getImgIds()
    path = os.path.join(root, name)
    imgs = coco.loadImgs(imgIds)
    _imgs = list(filter(lambda img: os.path.exists(os.path.join(path, img['file_name'])), imgs))
    if len(imgs) > len(_imgs):
        tf.logging.warn('%d of %d images not exists' % (len(imgs) - len(_imgs), len(imgs)))
    cnt_noobj = 0
    for img in tqdm.tqdm(_imgs):
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(anns) <= 0:
            cnt_noobj += 1
            continue
        imagepath = os.path.join(path, img['file_name'])
        width, height = img['width'], img['height']
        imageshape = [height, width, 3]
        objects_class = np.array([id_index[ann['category_id']] for ann in anns], dtype=np.int64)
        objects_coord = [ann['bbox'] for ann in anns]
        objects_coord = [(x, y, x + w, y + h) for x, y, w, h in objects_coord]
        objects_coord = np.array(objects_coord, dtype=np.float32)
        if verify:
            if not verify_coords(objects_coord, imageshape):
                tf.logging.error('failed to verify coordinates of ' + imagepath)
                continue
            if not verify_image_jpeg(imagepath, imageshape):
                tf.logging.error('failed to decode ' + imagepath)
                continue
        assert len(objects_class) == len(objects_coord)
        example = tf.train.Example(features=tf.train.Features(feature={
            'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
            'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
            'objects': tf.train.Feature(bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
        }))
        writer.write(example.SerializeToString())
    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images have no object' % (cnt_noobj, len(_imgs)))
    return True
