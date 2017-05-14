# TensorFlow implementation of the [YOLO (You Only Look Once)](https://arxiv.org/pdf/1506.02640.pdf) and [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf)

## Dependencies

Python 3, TensorFlow 1.0, NumPy, SciPy, Pandas, SymPy, Matplotlib, BeautifulSoup4, OpenCV, PIL, tqdm

## Configuration

Configurations are mainly defined in the "config.ini" file. Such as the detection model (config/model), base directory (config/basedir, which identifies the cache files (.tfrecord), the model data files (.ckpt), and summary data for TensorBoard), and the inference function ([model]/inference). Be ware the configurations can be extended using the "-c" command-line argument.

## Basic Usage

- Download the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 2007 ([training, validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)) and 2012 ([training and validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) dataset. Extract these tars into one directory (such as "~/Documents/Database/").

- Download the [COCO](http://mscoco.org/) 2014 ([training](http://msvocds.blob.core.windows.net/coco2014/train2014.zip), [validation](http://msvocds.blob.core.windows.net/coco2014/val2014.zip), and [test](http://msvocds.blob.core.windows.net/coco2014/test2014.zip)) dataset. Extract these zip files into one directory (such as "~/Documents/Database/coco/").

- Run "cache.py" to create the cache file for the training program. A verify command-line argument "-v" is recommended to check the training data and drop the corrupted examples.

- Run "train.py" to start the training process (the model data saved previously will be loaded if it exists). Multiple command-line arguments can be defined to control the training process. Such as the batch size, the learning rate, the optimization algorithm and the maximum number of steps.

- Run "detect.py" to detect objects in an image. Run "export CUDA_VISIBLE_DEVICES=" to avoid out of GPU memory error while the training process is running.

## Examples

### Training a 20 classes Darknet YOLOv2 model from a pretrained 80 classes model

- Cache the 20 classes training data using the customized config file argument. Cache files (.tfrecord) in "~/Documents/Database/yolo-tf/cache/voc" will be created.

```
python3 cache.py -c config.ini config/yolo2/darknet-voc.ini -v
```

- Download a 80 classes Darknet YOLOv2 model (the original file name is "yolo.weights", a [version](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) from Darkflow is recommanded). In this tutorial I put it in "~/Downloads/yolo.weights".

- Parse the 80 classes Darknet YOLOv2 model into Tensorflow format (~/Documents/Database/yolo-tf/yolo2/darknet/coco/model.ckpt). A warning like "xxx bytes remaining" indicates the file "yolo.weights" is not compatiable with the original Darknet YOLOv2 model (defined in the function `model.yolo2.inference.darknet`).

```
python3 parse_darknet_yolo2.py ~/Downloads/yolo.weights -c config.ini config/yolo2/darknet-coco.ini -d
```

- Fine-tuning the 80 classes Darknet YOLOv2 model into a 20 classes model (~/Documents/Database/yolo-tf/yolo2/darknet/voc) except the final convolutional layer and hyper-parameters. Starting the training process with gradient clipping to avoid NaN error. **Be ware the "-d" command-line argument will delete the model files and should be used only once when initializing the model**.

```
python3 train.py -c config.ini config/yolo2/darknet-voc.ini -f ~/Documents/Database/yolo-tf/yolo2/darknet/coco/model.ckpt -e yolo2_darknet/conv loss/hparam -g 0.9 -d
```

- Using the following command in another terminal and opening the address "localhost:6006" in a web browser to monitor the training process.

```
tensorboard --logdir /home/srm/Documents/Database/yolo-tf/yolo2/darknet/voc
```

- If you think your model is stabilized, press Ctrl+C to cancel and restart the training process without gradient clipping.

```
python3 train.py -c config.ini config/yolo2/darknet-voc.ini
```

- Training about 60,000 steps and detect objects with a camra.

```
python3 detect_camra.py -c config.ini config/yolo2/darknet-voc.ini
```

## License

This project is released as the open source software with the GNU Lesser General Public License version 3 ([LGPL v3](http://www.gnu.org/licenses/lgpl-3.0.html)).

# Acknowledgements

This project is mainly inspired by the following projects:

* [YOLO (Darknet)](https://pjreddie.com/darknet/yolo/).
* [Darkflow](https://github.com/thtrieu/darkflow).
