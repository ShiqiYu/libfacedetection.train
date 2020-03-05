# config.py

cfg = {
    'name': 'YuFaceDetectNet',
    #'min_sizes': [[32, 64, 128], [256], [512]],
    #'steps': [32, 64, 128],
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True
}
