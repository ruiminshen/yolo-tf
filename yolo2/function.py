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

import numpy as np
import tensorflow as tf


def reorg(net, stride=2):
    batch_size, height, width, channels = net.get_shape().as_list()
    _height, _width, _channel = height // stride, width // stride, channels * stride * stride
    net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
    net = tf.transpose(net, [0, 1, 3, 2, 4, 5]) # batch_size, _height, _width, stride, stride, channels
    net = tf.reshape(net, [batch_size, _height, _width, -1], name='merge_channels')
    return net


def main():
    image = [
        (0, 1, 0, 1),
        (2, 3, 2, 3),
        (0, 1, 0, 1),
        (2, 3, 2, 3),
    ]
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, -1)
    with tf.Session() as sess:
        ph_image = tf.placeholder(tf.uint8, image.shape)
        images = sess.run(reorg(ph_image), feed_dict={ph_image: image})
    for i, image in enumerate(np.transpose(images[0], [2, 0, 1])):
        data = np.unique(image)
        assert len(data) == 1
        assert data[0] == i

if __name__ == '__main__':
    main()
