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

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def std(path, width, height):
    imagefile = tf.read_file(path)
    image_rgb = tf.image.decode_jpeg(imagefile, channels=3)
    image_rgb = tf.image.resize_images(image_rgb, [height, width])
    image_std = tf.image.per_image_standardization(image_rgb)
    return tf.cast(image_rgb, tf.uint8), image_std


def darknet(path, width, height):
    _image = cv2.imread(path)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    _image = cv2.resize(_image, (width, height))
    image = _image / 255.
    return ops.convert_to_tensor(_image), ops.convert_to_tensor(image.astype(np.float32))
