# TensorFlow implementation of the [YOLO (You Only Look Once)](https://arxiv.org/pdf/1506.02640.pdf) and [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf)

## Dependencies

Python 3, TensorFlow 1.0, TensorLayer, NumPy, SciPy, Pandas, SymPy, Matplotlib, BeautifulSoup4

## Configuration

Configurations are mainly defined in the "config.ini" file. For example, the model name is defined in option "model" in section "config", and the parameters defined in section "queue" is used to maximize GPU usage. The object classes file, the input data cache path, the model base directory (which identifies the parameter "logdir" for TensorBoard and the model data files), the model structure (convolutional layers or fully connected layers) definition files (.tsv) and hyper-parameters are defined in the sections correspoding to the model name.

## Basic Usage

- Download PASCAL VOC2007 data ([training, validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)). Extract these tars into one directory (such as "~/Documents/Database/").

- Run "cache.py" (the PASCAL VOC2007 data directory (such as "~/Documents/Database/VOCdevkit/VOC2007") should be given) to create the cache file for the training program.

- Run "train.py" to start the training process (the model data saved previously will be loaded if it exists). Multiple command line options can be defined to control the training process. Such as the batch size, the learning rate, the model data saving frequency and the maximum number of evaluation. To manually terminate the training program, press Ctrl+C key and the model data will be saved.

- Run "identify.py" to detect objects in an image. Run "export CUDA_VISIBLE_DEVICES=" to avoid out of GPU memory error during the training process.

## License

This project is released as the open source software with the GNU Lesser General Public License version 3 ([LGPL v3](http://www.gnu.org/licenses/lgpl-3.0.html)).
