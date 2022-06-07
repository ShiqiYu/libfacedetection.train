import torch
import argparse
from model import YuDetectNet
import yaml


parser = argparse.ArgumentParser(description='Face and Landmark Detection')
parser.add_argument('-c', '--config', default='./config/yufacedet.yaml',
                    type=str, help='config path to open')
parser.add_argument('-m', '--trained_model', default='./weights/yunet_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-o', '--output', default='./weights_cpp/facedetectcnn-data.cpp',
                    type=str, help='The output cpp file, trained parameters inside')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



if __name__ == '__main__':

    torch.set_grad_enabled(False)
    with open(args.config, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # net and model
    net = YuDetectNet(cfg)    # initialize detector
    net = load_model(net, args.trained_model, True)
    net.eval()

    print('Finished loading model!')
    
    net.export_cpp(args.output)
    print(f'Finish export cpp-data to {args.output}')