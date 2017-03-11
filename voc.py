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
import sys
import bs4


def load_dataset(path, names):
    _names = dict([(name, i) for i, name in enumerate(names)])
    _path = os.path.join(path, 'Annotations')
    imagenames = []
    imageshapes = []
    labels = []
    for filename in os.listdir(_path):
        with open(os.path.join(_path, filename), 'r') as f:
            anno = bs4.BeautifulSoup(f.read(), 'xml').find('annotation')
        objects = []
        for obj in anno.find_all('object', recursive=False):
            for bndbox, name in zip(obj.find_all('bndbox', recursive=False), obj.find_all('name', recursive=False)):
                if name.text in _names:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    objects.append((xmin, ymin, xmax, ymax, _names[name.text]))
                else:
                    sys.stderr.write(name.text + ' not in names')
        imagenames.append(anno.find('filename').text)
        size = anno.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        imageshapes.append((height, width, depth))
        labels.append(objects)
    return imagenames, imageshapes, labels
