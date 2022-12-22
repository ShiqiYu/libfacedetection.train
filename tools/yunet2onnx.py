# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from functools import partial

import numpy as np
import onnx
import torch
from mmcv import Config, DictAction

from mmdet.core.export import build_model_from_cfg, preprocess_example_input

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')

    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')

    args = parser.parse_args()
    return args


def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 normalize_cfg,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None,
                 skip_postprocess=False):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    img_list, img_meta_list = [one_img], [[one_meta]]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)

    output_names = []
    for t in ['cls', 'obj', 'bbox']:
        output_names.extend([f'{t}_{s}' for s in [8, 16, 32]])

    if model.bbox_head.use_kps:
        output_names.extend([f'kps_{s}' for s in [8, 16, 32]])

    input_name = 'input'
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {out: {0: 'batch', 1: 'dim'} for out in output_names}
        dynamic_axes[input_name] = {0: 'batch', 2: 'height', 3: 'width'}

    torch.onnx.export(
        model,
        img_list,
        output_file,
        input_names=[input_name],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    model.forward = origin_forward

    # get the custom op path
    # ort_custom_op_path = ''
    # try:
    #     from mmcv.ops import get_onnxruntime_op_path
    #     ort_custom_op_path = get_onnxruntime_op_path()
    # except (ImportError, ModuleNotFoundError):
    #     warnings.warn('If input model has custom op from mmcv, '
    #         'you may have to build mmcv with ONNXRuntime from source.')

    if do_simplify:
        import onnxsim
        from mmdet import digit_version

        min_required_version = '0.3.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        input_dic = {'input': img_list[0].detach().cpu().numpy()}
        model_opt, check_ok = onnxsim.simplify(
            output_file,
            input_data=input_dic,
        )
        if check_ok:
            output_file_sim = output_file.replace('.onnx', '_sim.onnx')
            onnx.save(model_opt, output_file_sim)
            os.remove(output_file)            
            os.rename(output_file_sim, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')

    net = onnx.load(output_file)
    onnx.checker.check_model(net)

    if net.ir_version < 4:
        print('Model with ir_version below 4 requires'
              'to include initilizer in graph input')
        return

    inputs = net.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in net.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(net, output_file)
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        import onnxruntime
        # wrap onnx model
        if test_img is None:
            input_config['input_path'] = input_img

        # prepare input once again
        one_img, one_meta = preprocess_example_input(input_config)

        # get pytorch output, specific for yunet
        with torch.no_grad():
            cls_preds, bbox_preds, obj_preds, kps_preds = model.feature_test(one_img)
            flatten_cls_preds = [
                cls_pred.permute(0, 2, 3, 1).view(1, -1, 1).sigmoid().numpy()
                for cls_pred in cls_preds
            ]
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).view(1, -1, 4).numpy()
                for bbox_pred in bbox_preds
            ]
            flatten_kps_preds = [
                kps_pred.permute(0, 2, 3, 1).view(1, -1, 10).numpy()
                for kps_pred in kps_preds
            ]
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).view(1, -1, 1).sigmoid().numpy()
                for objectness in obj_preds
            ]
        pytorch_results = flatten_cls_preds + flatten_objectness + flatten_bbox_preds + flatten_kps_preds

        # get onnx output
        session = onnxruntime.InferenceSession(output_file, None)
        onnx_results = session.run(None, {session.get_inputs()[0].name: one_img.detach().numpy()})

        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for o_res, p_res in zip(onnx_results, pytorch_results):
            np.testing.assert_allclose(
                o_res, p_res, rtol=1e-02, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')
    print('Over!')


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


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.data.test.pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # build the model and load checkpoint
    model = build_model_from_cfg(args.config, args.checkpoint,
                                 args.cfg_options)

    if not args.input_img:
        args.input_img = 'demo/demo.jpg'
    normalize_cfg = parse_normalize_cfg(cfg.data.test.pipeline)

    tag = 'dynamic' if args.dynamic_export \
        else f'{input_shape[-2]}_{input_shape[-1]}'

    if args.output_file is None:
        output_path = ('./onnx/'
                    f'{os.path.basename(args.config).rstrip(".py")}'
                    f'_{tag}.onnx')
        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))
        args.output_file = output_path

    # convert model to onnx file
    pytorch2onnx(
        model,
        args.input_img,
        input_shape,
        normalize_cfg,
        opset_version=args.opset_version,
        show=False,
        output_file=args.output_file,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export)
