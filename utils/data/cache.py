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
from PIL import Image
import tqdm
import numpy as np
import scipy.misc
import tensorflow as tf
import utils.data.voc


def verify_imageshape(imagepath, imageshape):
    with Image.open(imagepath) as img:
        return np.all(np.equal(img.size, imageshape[1::-1]))


def verify_image(imagepath, imageshape):
    img = scipy.misc.imread(imagepath)
    return np.all(np.equal(img.shape[:2], imageshape[:2]))


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
        return
    with open(path, 'r') as f:
        filenames = [line.strip() for line in f]
    cnt_noobj = 0
    for filename in tqdm.tqdm(filenames):
        try:
            imagename, imageshape, objects_class, objects_coord = utils.data.voc.load_dataset(os.path.join(root, 'Annotations', filename + '.xml'), name_index)
            objects_class = np.array(objects_class, dtype=np.int64)
            objects_coord = np.array(objects_coord, dtype=np.float32)
            if verify:
                assert verify_coords(objects_coord, imageshape)
            else:
                objects_coord = fix_coords(objects_coord, imageshape)
            if len(objects_class) <= 0:
                cnt_noobj += 1
                continue
            imagepath = os.path.join(root, 'JPEGImages', imagename)
            if verify:
                assert verify_image(imagepath, imageshape)
            assert len(objects_class) == len(objects_coord)
            example = tf.train.Example(features=tf.train.Features(feature={
                'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
                'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
                'objects': tf.train.Feature(bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
            }))
            writer.write(example.SerializeToString())
        except:
            tf.logging.fatal('error while caching ' + imagepath)
    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images has no objects' % (cnt_noobj, len(filenames)))


def coco(writer, name_index, profile, row, verify=False):
    root = os.path.expanduser(os.path.expandvars(row['root']))
    year = str(row['year'])
    name = profile + year
    path = os.path.join(root, 'annotations', 'instances_%s.json' % name)
    if not os.path.exists(path):
        tf.logging.warn(path + ' not exists')
        return
    from pycocotools.coco import COCO
    coco = COCO(path)
    catIds = coco.getCatIds(catNms=list(name_index.keys()))
    cats = coco.loadCats(catIds)
    id_index = dict((cat['id'], name_index[cat['name']]) for cat in cats)
    imgIds = coco.getImgIds()
    path = os.path.join(root, name)
    cnt_noimg = 0
    cnt_noobj = 0
    for img in tqdm.tqdm(coco.loadImgs(imgIds)):
        try:
            imagepath = os.path.join(path, img['file_name'])
            if not os.path.exists(imagepath):
                cnt_noimg += 1
                continue
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(anns) <= 0:
                cnt_noobj += 1
                continue
            width, height = img['width'], img['height']
            imageshape = [height, width, 3]
            if verify:
                assert verify_image(imagepath, imageshape)
            objects_class = [id_index[ann['category_id']] for ann in anns]
            objects_coord = [ann['bbox'] for ann in anns]
            objects_coord = [(x, y, x + w, y + h) for x, y, w, h in objects_coord]
            objects_class = np.array(objects_class, dtype=np.int64)
            objects_coord = np.array(objects_coord, dtype=np.float32)
            if verify:
                assert verify_coords(objects_coord, imageshape)
            else:
                objects_coord = fix_coords(objects_coord, imageshape)
            assert len(objects_class) == len(objects_coord)
            example = tf.train.Example(features=tf.train.Features(feature={
                'imagepath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imagepath)])),
                'imageshape': tf.train.Feature(int64_list=tf.train.Int64List(value=imageshape)),
                'objects': tf.train.Feature(bytes_list=tf.train.BytesList(value=[objects_class.tostring(), objects_coord.tostring()])),
            }))
            writer.write(example.SerializeToString())
        except:
            tf.logging.fatal('error while caching ' + imagepath)
    if cnt_noimg > 0:
        tf.logging.warn('%d of %d images not exists' % (cnt_noimg, len(imgIds)))
    if cnt_noobj > 0:
        tf.logging.warn('%d of %d images has no objects' % (cnt_noobj, len(imgIds)))
