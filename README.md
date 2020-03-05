# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [FaceBoxes.PyTorch](https://github.com/sfzhang15/FaceBoxes.PyTorch) and [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).


### Contents
- [Installation](#installation)
- [Training](#training)
- [Detection](#detection)
- [Export CPP source code](#export-cpp-source-code)
- [Design your own model](#design-your-own-model)

## Installation
1. Install [PyTorch](https://pytorch.org/) >= v1.0.0 following official instruction.

2. Clone this repository. We will call the cloned directory as `$TRAIN_ROOT`.
```Shell
git clone https://github.com/ShiqiYu/libfacedetection.train
```

_Note: Codes are based on Python 3+._

## Training
1. Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html) dataset, place the images under this directory:
  ```Shell
  $TRAIN_ROOT/data/WIDER_FACE_rect/images
  ```
  and create a symbol link to this directory from  
  ```Shell
  $TRAIN_ROOT/data/WIDER_FACE_landmark/images
  ```
2. Train the model using WIDER FACE:
  ```Shell
  cd $TRAIN_ROOT/tasks/task1/
  python3 train.py
  ```

## Detection
```Shell
cd $TRAIN_ROOT/tasks/task1/
./detect.py -m weights/yunet_final.pth --image_file=filename.jpg
```

## Export CPP source code
The following bash code can export a CPP file for project [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
```Shell
cd $TRAIN_ROOT/tasks/task1/
./exportcpp.py -m weights/yunet_final.pth -o output.cpp
```
## Design your own model
You can copy $TRAIN_ROOT/tasks/task1/ to $TRAIN_ROOT/tasks/task2/ or other similar directory, and then modify the model defined in file: tasks/task2/yufacedetectnet.py .

