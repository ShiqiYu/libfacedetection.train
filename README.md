# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [FaceBoxes.PyTorch](https://github.com/sfzhang15/FaceBoxes.PyTorch) and [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

Visualization of our network architecture: [[netron]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_yunet_final_dynamic_simplify.onnx).


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
3. Install NVIDIA DALI following official instruction: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html

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
    └── trainset.json           
    ```
_NOTE: \
We relabled the WIDER Face train set using [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace). New labels are in 
`$TRAIN_ROOT/data/widerface/trainset.json`, which is the COCO_format annotations file used in DALI dataloader._

## Training
```Shell
python train.py -c ./config/yufacedet.yaml -t demo 
```

## Detection
```Shell
python detect.py -c ./config/yufacedet.yaml -m weights/yunet_final.pth --target filename.jpg 
```

## Evaluation on WIDER Face
1. Build NMS module.
    ```shell
    cd tools/widerface_eval
    python setup.py build_ext --inplace
    ```

2. Perform evaluation. To reproduce the following performance, run on the default settings. Run `python test.py --help` for more options.
    ```shell
    python test.py -m weights/yunet_final.pth -c ./config/yufacedet.yaml
    ```

_NOTE: We now use the Python version of `eval_tools` from [here](https://github.com/wondervictor/WiderFace-Evaluation)._

Performance on WIDER Face (Val): confidence_threshold=0.3, nms_threshold=0.45, in origin size:
```
AP_easy=0.882, AP_medium=0.871, AP_hard=0.767
```

## Export CPP source code
The following bash code can export a CPP file for project [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
```Shell
python exportcpp.py -c ./config/yufacedet.yaml -m weights/yunet_final.pth
```

## Export to onnx model
Export to onnx model for [libfacedetection/example/opencv_dnn](https://github.com/ShiqiYu/libfacedetection/tree/master/example/opencv_dnn).
```shell
python exportonnx.py -m weights/yunet_final.pth
```
## Compare ONNX model with other works
Inference on exported ONNX models using ONNXRuntime:
```shell
python tools/compare_inference.py ./onnx/yunet_final_dynamic_simplify.onnx --mode AUTO --eval --score_thresh 0.3 --nms_thresh 0.45
```
Some similar approaches(e.g. SCRFD, Yolo5face, retinaface) to inference are also supported.

With Intel i7-12700K and `input_size = origin size, score_thresh = 0.3, nms_thresh = 0.45`, some results are list as follow:
 | Model | AP_easy | AP_medium | AP_hard | #Params | Params Ratio | MFlops | Froward (ms) | 
 | ----- | ------- | --------- | ------- | ------- | ------------ | ------ | ------- | 
 | SCRFD0.5(ICLR2022) | 0.879 | 0.863 | 0.759 | 631410 | 7.43x | 184 | 22.3 | 12.9
 | Retinaface0.5(CVPR2020) | 0.899 | 0.866 | 0.660 | 426608 | 5.02X | 245 | 13.9 | 
 | YuNet(Ours) | 0.885 | 0.877 | 0.762 | 85006 | 1.0x | 136 | 10.6 |

The compared ONNX model is avaliable in https://share.weiyun.com/nEsVgJ2v Password：gydjjs

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
