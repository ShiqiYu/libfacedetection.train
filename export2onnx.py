import argparse
import os
import yaml
import sys
import torch
import cv2
import time
import numpy as np
import onnx
import onnxruntime
from onnxsim import simplify
from model import YuDetectNet

parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('--config', '-c', default="./config/yufacedet.yaml", type=str, help='config to test')
parser.add_argument('--model', '-m', default="./weights/yunet_final.pth", type=str, help='model weights path')
parser.add_argument('--dynamic', action='store_true', help='use dynamic axes export')
parser.add_argument('--simplify', action='store_true', help='use onnx-simplifier to simplify onnx')
parser.add_argument('--size', type=int, nargs='+', default=640, help='input size to export')
parser.add_argument('--test', action='store_true', help='test onnx and pth')

def arg_initial(args):
    with open(args.config, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    return cfg

def main():
    args = parser.parse_args()
    cfg = arg_initial(args)

    torch.set_grad_enabled(False)
    net = YuDetectNet(cfg)
    net.load_state_dict(torch.load(args.model), strict=True)
    net.eval()

    if isinstance(args.size, int):
        input_size = (args.size, args.size)
    elif len(args.size) == 1:
        input_size = (args.size[0], args.size[0])
    elif len(args.size) == 2:
        input_size = args.size
    else:
        raise ValueError(f'Input size should be 1 or 2 but get {len(args.size)} ({args.size})')
    input_shape = (1, 3, *input_size)
    img = torch.randn(input_shape, requires_grad=False)
    input_names = ['input']
    output_names = ['loc', 'conf', 'iou']
    # dynamic_axes = None
    dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
    dynamic_axes[input_names[0]] = {
        0: '?',
        2: '?',
        3: '?'
    }


    if args.dynamic:
        output_path = os.path.abspath(os.path.join('./onnx', f'yunet_{os.path.basename(args.model[:-4])}_dynamic.onnx')) 
    else:
        output_path = os.path.abspath(os.path.join('./onnx', f'yunet_{os.path.basename(args.model[:-4])}_{input_size[0]}_{input_size[1]}.onnx')) 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    print(f'Export:\n{args.model}\nTo onnx:\n{output_path}')
    torch.onnx.export(
        model=net,
        args=img,
        f=output_path, 
        input_names=input_names,
        output_names=output_names,        
        export_params=True,
        verbose=False,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes if args.dynamic else None,
        opset_version=11)

    if args.simplify:
        net_onnx = onnx.load(output_path)
        os.remove(output_path)
        #print(model.graph.input[0])
        if args.dynamic:
            input_shapes = {net_onnx.graph.input[0].name : list(input_shape)}
            net_onnx, check = simplify(net_onnx, input_shapes=input_shapes, dynamic_input_shape=True)
        else:
            net_onnx, check = simplify(net_onnx)
        assert check, "Simplified ONNX model could not be validated"
        output_path_simplify = output_path.replace('.onnx', '_simplify.onnx')
        onnx.save(net_onnx, output_path_simplify)
        output_path = output_path_simplify

    if args.test:
        print('Test model:')   
        net_onnx = onnx.load(output_path)
        onnx.checker.check_model(net_onnx)
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}

        epoch = 50

        ort_t_1 = time.time()
        for i in range(epoch):
            ort_outs = ort_session.run(None, ort_inputs)
        ort_t_2 = time.time()
        
        torch_t_1 = time.time()
        for i in range(epoch):
            torch_outs = net(img)
        torch_t_2 = time.time()

        if args.simplify:
            ort_session_sim = onnxruntime.InferenceSession(output_path)
            ort_sim_t_1 = time.time()
            for i in range(epoch):
                ort_outs_sim = ort_session_sim.run(None, ort_inputs)
            ort_sim_t_2 = time.time()

        print(f"Loop {epoch}")
        print(f"torch time:{torch_t_2 - torch_t_1}")
        print(f"ort time: {ort_t_2 - ort_t_1}")
        if args.simplify:
            print(f"ort_sim time: {ort_sim_t_2 - ort_sim_t_1}")

        for torch_out, ort_out in zip(torch_outs, ort_outs):
            np.testing.assert_allclose(to_numpy(torch_out), ort_out, rtol=1e-03, atol=1e-05)
        print('Successful!')
    
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    main()
    