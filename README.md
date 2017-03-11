# TensorFlow implementation of the [YOLO (Real-Time Object Detection)](https://arxiv.org/pdf/1506.02640.pdf)

## Dependencies

Python 3, TensorFlow 1.0, Numpy, Pandas, Sympy, Matplotlib, BeautyfulSoap

## Basic Usage

- Download PASCAL VOC2007 data ([training, validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)).

- Extract these tars into one directory (~/Documents/Database), which is defined in "voc/path" of config.ini.

- Run yolo.py to create VOC label cache (path is defined in "yolo/cache" of config.ini) for the YOLO training program.

- Run train.py to load the model saved previously (if exists) and start the training process. A base directory (defined in "yolo/dir" of config.ini) identifies the logdir (for TensorBoard) and the model. The model will be saved periodically during the training process, and you can define the maximum evaluate number (-e option in command line) or press Ctrl+C to terminate the training program.

- Run identify.py to detect objects in an image.

## Configuration

The main configuration file is config.ini, which defines parameters such as object classes file (yolo/names), resized image size, model definition file (both convolutional layers and fully connected layers), hyper-parameters (parameter for regularizer and 4 parameters in section "yolo_hparam" for YOLO optimization objectives), and queue parameters (section "queue") to maximize GPU usage.

## Known Issue

Batch normalization still under development.

## License

This project is released as the open source software with the GNU Lesser General Public License version 3 ([LGPL v3](http://www.gnu.org/licenses/lgpl-3.0.html)).
