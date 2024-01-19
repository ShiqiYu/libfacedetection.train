from mmdet.core.export import build_model_from_cfg
import cv2
import numpy as np
import torch
import os

try:
    from tinynn.converter import TFLiteConverter
except ImportError:
    s = """
    Use tinynn to convert the model to TFLite format.
    Please refer to: https://github.com/alibaba/TinyNeuralNetwork
    ```
    git clone https://github.com/alibaba/TinyNeuralNetwork.git
    cd TinyNeuralNetwork
    python setup.py install   
    ``` 
    """
    raise ImportError()


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config

def resize_img(img, mode):
    if mode == 'ORIGIN':
        det_img, det_scale = img, 1.
    elif mode == 'AUTO':
        assign_h = ((img.shape[0] - 1) & (-32)) + 32
        assign_w = ((img.shape[1] - 1) & (-32)) + 32
        det_img = np.zeros((assign_h, assign_w, 3), dtype=np.uint8)
        det_img[:img.shape[0], :img.shape[1], :] = img
        det_scale = 1.
    else:
        if mode == 'VGA':
            input_size = (640, 480)
        else:
            input_size = list(map(int, mode.split(',')))
        assert len(input_size) == 2
        x, y = max(input_size), min(input_size)
        if img.shape[1] > img.shape[0]:
            input_size = (x, y)
        else:
            input_size = (y, x)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale

def load_model():
    config = 'configs/yunet_n.py'
    checkpoint = 'weights/yunet_n.pth'

    # build the model and load checkpoint
    model = build_model_from_cfg(config, checkpoint, None)
    model.forward = model.feature_test
    return model

def main_worker(output_path):
    model = load_model()
    model.cpu()
    model.eval()

    dummy_input = torch.rand((1, 3, 256, 320))

    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = 'qnnpack'

    # The code section below is used to convert the model to the TFLite format
    # If you want perform dynamic quantization on the float models,
    # you may refer to `dynamic.py`, which is in the same folder.
    # As for static quantization (e.g. quantization-aware training and post-training quantization),
    # please refer to the code examples in the `examples/quantization` folder.
    converter = TFLiteConverter(model, dummy_input, output_path)
    converter.convert()
    print('ok')


if __name__ == '__main__':
    # output_path = os.path.join(CURRENT_PATH, 'yunet_n_with_postprocess_320x320.tflite')
    output_folder = os.path.join(CURRENT_PATH, '../tflite/')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_path = os.path.join(output_folder, 'yunet_n_320x256.tflite')
    main_worker(output_path)

