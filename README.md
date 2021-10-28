# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [FaceBoxes.PyTorch](https://github.com/sfzhang15/FaceBoxes.PyTorch) and [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

Visualization of our network architecture: [[netron]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/tasks/task1/onnx/YuFaceDetectNet.onnx).


### Contents
- [Installation](#installation)
- [Preparation](#Preparation)
- [Training](#training)
- [Detection](#detection)
- [Evaluation on WIDER Face](#evaluation-on-wider-face)
- [Export CPP source code](#export-cpp-source-code)
- [Export to ONNX model](#export-to-onnx-model)
- [Design your own model](#design-your-own-model)
- [Citation](#citation)

## Installation
1. Install [PyTorch](https://pytorch.org/) >= v1.7.0 following official instruction.

2. Clone this repository. We will call the cloned directory as `$TRAIN_ROOT`.
    ```Shell
    git clone https://github.com/ShiqiYu/libfacedetection.train
    ```

3. Install dependencies.
    ```shell
    pip install -r requirements.txt
    ```

_Note: Codes are based on Python 3+._

## Preparation

1. Download the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) dataset and its [evaluation tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip).
2. Extract zip files under `$TRAIN_ROOT/data/widerface` as follows:
    ```shell
    $ tree data/widerface
    data/widerface
    ├── eval_tools
    ├── wider_face_split
    ├── WIDER_test
    ├── WIDER_train
    ├── WIDER_val
    └── trainset.json           
    ```
_NOTE: \
We relabled the WIDER Face train set using [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace). New labels are in 
`$TRAIN_ROOT/data/widerface/trainset.json`, which is the COCO_format annotations file used in DALI dataloader._

## Training
```Shell
cd $TRAIN_ROOT/tasks/task1/
python train.py
```

## Detection
```Shell
cd $TRAIN_ROOT/tasks/task1/
python detect.py -m weights/yunet_final.pth --image_file=filename.jpg
```

## Evaluation on WIDER Face
1. Build NMS module.
    ```shell
    cd $TRAIN_ROOT/src/widerface_eval
    python setup.py build_ext --inplace
    ```

2. Perform evaluation. To reproduce the following performance, run on the default settings. Run `python test.py --help` for more options.
    ```shell
    cd $TRAIN_ROOT/tasks/task1/
    python test.py -m weights/yunet_final.pth
    ```

_NOTE: We now use the Python version of `eval_tools` from [here](https://github.com/wondervictor/WiderFace-Evaluation)._

Performance on WIDER Face (Val): scales=[1.], confidence_threshold=0.3:
```
AP_easy=0.856, AP_medium=0.842, AP_hard=0.727
```

## Export CPP source code
The following bash code can export a CPP file for project [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
```Shell
cd $TRAIN_ROOT/tasks/task1/
python exportcpp.py -m weights/yunet_final.pth -o output.cpp
```

## Export to onnx model
Export to onnx model for [libfacedetection/example/opencv_dnn](https://github.com/ShiqiYu/libfacedetection/tree/master/example/opencv_dnn).
```shell
cd $TRAIN_ROOT/tasks/task1/
python exportonnx.py -m weights/yunet_final.pth
```

## Design your own model
You can copy `$TRAIN_ROOT/tasks/task1/` to `$TRAIN_ROOT/tasks/task2/` or other similar directory, and then modify the model defined in file: tasks/task2/yufacedetectnet.py .


## Citation
The loss used in training is EIoU, a novel extended IoU. More details can be found in:

	@article{eiou,
	 author={Peng, Hanyang and Yu, Shiqi},
  	 journal={IEEE Transactions on Image Processing}, 
  	 title={A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization}, 
  	 year={2021},
  	 volume={30},
  	 pages={5032-5044},
	 doi={10.1109/TIP.2021.3077144}
	 }
The paper can be open accessed at https://ieeexplore.ieee.org/document/9429909.

We also published a paper on face detection to evaluate different methods.

	@article{facedetect-yu,
	 author={Yuantao Feng and Shiqi Yu and Hanyang Peng and Yan-ran Li and Jianguo Zhang}
	 title={Detect Faces Efficiently: A Survey and Evaluations},
	 journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
	 year={2021}
	 }
	 
The paper can be open accessed at https://ieeexplore.ieee.org/document/9580485

