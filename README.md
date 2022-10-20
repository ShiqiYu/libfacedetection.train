# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Some data processing functions from [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet) modifications.

Visualization of our network architecture: [\[netron\]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/tasks/task1/onnx/YuFaceDetectNet.onnx).

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

1. Install [PyTorch](https://pytorch.org/) >= v1.7.0 following official instruction. e.g.\
   On GPU platforms (cu102):\\
   ```shell
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
   ```
2. Install [MMCV](https://github.com/open-mmlab/mmcv) >= v1.3.17 following official instruction. e.g.\\
   ```shell
   pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
   ```
3. Clone this repository. We will call the cloned directory as `$TRAIN_ROOT`.
   ```Shell
   git clone https://github.com/Wwupup/wwfacedet
   cd wwfacedet
   python setup.py develop
   ```
4. Install dependencies.
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
   ├── wider_face_split
   ├── WIDER_test
   ├── WIDER_train
   ├── WIDER_val
   └── labelv2
         ├── train
         │   └── labelv2.txt
         └── val
             ├── gt
             └── labelv2.txt
   ```

<!-- _NOTE: \
We relabled the WIDER Face train set using [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace). New labels are in
`$TRAIN_ROOT/data/widerface/trainset.json`, which is the COCO_format annotations file used in DALI dataloader._ -->

## Training

Following MMdetection training processing.

```Shell
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh ./config/yunet/x.py 2
```

## Detection

```Shell
python tools/detect-image.py ./config/x.py ./work_dirs/x/latest.pth ./image.jpg
```

## Evaluation on WIDER Face

```shell
python tools/test_widerface.py ./config/x.py ./work_dirs/x/latest.pth --mode 2
```

Performance on WIDER Face (Val): confidence_threshold=0.02, nms_threshold=0.45, in origin size:

```
AP_easy=0.899, AP_medium=0.883, AP_hard=0.792
```

## Export CPP source code

The following bash code can export a CPP file for project [libfacedetection](https://github.com/ShiqiYu/libfacedetection)

```Shell
python tools/export2cpp.py ./config/x.py ./work_dirs/x/latest.pth
```

## Export to onnx model

Export to onnx model for [libfacedetection/example/opencv_dnn](https://github.com/ShiqiYu/libfacedetection/tree/master/example/opencv_dnn).

```shell
python tools/wwdet2onnx.py ./config/x.py ./work_dirs/x/latest.pth
```

## Compare ONNX model with other works

Inference on exported ONNX models using ONNXRuntime:

```shell
python tools/compare_inference.py ./onnx/wwdet.onnx --mode AUTO --eval --score_thresh 0.02 --nms_thresh 0.45
```

Some similar approaches(e.g. SCRFD, Yolo5face, retinaface) to inference are also supported.

With Intel i7-12700K and `input_size = origin size, score_thresh = 0.3, nms_thresh = 0.45`, some results are list as follow:

| Model                   | AP_easy | AP_medium | AP_hard | #Params | Params Ratio | MFlops | Forward (ms) |
| ----------------------- | ------- | --------- | ------- | ------- | ------------ | ------ | ------------ |
| SCRFD0.5(ICLR2022)      | 0.879   | 0.863     | 0.759   | 631410  | 7.43x        | 184    | 22.3         |
| Retinaface0.5(CVPR2020) | 0.899   | 0.866     | 0.660   | 426608  | 5.02X        | 245    | 13.9         |
| YuNet(Ours)             | 0.885   | 0.877     | 0.762   | 85006   | 1.0x         | 136    | 10.6         |

The compared ONNX model is available in https://share.weiyun.com/nEsVgJ2v Password：gydjjs

## Citation

The loss used in training is EIoU, a novel extended IoU. More details can be found in:

```
@article{eiou,
 author={Peng, Hanyang and Yu, Shiqi},
 journal={IEEE Transactions on Image Processing},
 title={A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization},
 year={2021},
 volume={30},
 pages={5032-5044},
 doi={10.1109/TIP.2021.3077144}
 }
```

The paper can be open accessed at https://ieeexplore.ieee.org/document/9429909.

We also published a paper on face detection to evaluate different methods.

```
@article{facedetect-yu,
 author={Yuantao Feng and Shiqi Yu and Hanyang Peng and Yan-ran Li and Jianguo Zhang}
 title={Detect Faces Efficiently: A Survey and Evaluations},
 journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
 year={2021}
 }
```

The paper can be open accessed at https://ieeexplore.ieee.org/document/9580485
