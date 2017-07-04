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

import sys
import bs4


def load_dataset(path, name_index):
    with open(path, 'r') as f:
        anno = bs4.BeautifulSoup(f.read(), 'xml').find('annotation')
    objects_class = []
    objects_coord = []
    for obj in anno.find_all('object', recursive=False):
        for bndbox, name in zip(obj.find_all('bndbox', recursive=False), obj.find_all('name', recursive=False)):
            if name.text in name_index:
                objects_class.append(name_index[name.text])
                xmin = float(bndbox.find('xmin').text) - 1
                ymin = float(bndbox.find('ymin').text) - 1
                xmax = float(bndbox.find('xmax').text) - 1
                ymax = float(bndbox.find('ymax').text) - 1
                objects_coord.append((xmin, ymin, xmax, ymax))
            else:
                sys.stderr.write(name.text + ' not in names\n')
    size = anno.find('size')
    return anno.find('filename').text, (int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)), objects_class, objects_coord
