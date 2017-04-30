# TensorFlow implementation of the [YOLO (You Only Look Once)](https://arxiv.org/pdf/1506.02640.pdf) and [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf)

## Dependencies

Python 3, TensorFlow 1.0, NumPy, SciPy, Pandas, SymPy, Matplotlib, BeautifulSoup4, tqdm

## Configuration

Configurations are mainly defined in the "config.ini" file. For example, the model name is defined in option "model" in section "config", and the parameters defined in section "queue" is used to maximize GPU usage. The object classes file, the model base directory (which identifies the cache path, the parameter "logdir" for TensorBoard and the model data files), the model inference function and hyper-parameters are defined in the sections correspoding to the model name.

## Basic Usage

- Download PASCAL VOC2007 ([training, validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)) and VOC2012 ([training and validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) data. Extract these tars into one directory (such as "~/Documents/Database/").

- Run "cache.py" to create the cache file for the training program.

- Run "train.py" to start the training process (the model data saved previously will be loaded if it exists). Multiple command line options can be defined to control the training process. Such as the batch size, the learning rate, the model data saving frequency and the maximum number of steps. To manually terminate the training program, press Ctrl+C key and the model data will be saved.

- Run "identify.py" to detect objects in an image. Run "export CUDA_VISIBLE_DEVICES=" to avoid out of GPU memory error during the training process.

## License

This project is released as the open source software with the GNU Lesser General Public License version 3 ([LGPL v3](http://www.gnu.org/licenses/lgpl-3.0.html)).
