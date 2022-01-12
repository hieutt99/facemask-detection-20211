import os, sys 
import pandas as pd 
from utils.arguments import RunningArguments
import pandas as pd
from torch.utils.data import DataLoader 
# from utils.dataset import ImageLabelDataset
# from utils.augmentations import TRANSFORM_DICT
import logging


from utils.faster_rcnn.trainer import FasterRCNNTrainer
from utils.yolo.trainer import YoloTrainer
TRAINER_DICT={
    'fasterrcnn_resnet50_fpn': FasterRCNNTrainer,
    'fasterrcnn_mobilenet_v3_large_fpn':FasterRCNNTrainer,
    'yolov5s': YoloTrainer, 
    'yolov5m': YoloTrainer,
}

def get_trainer(model_type):
    return TRAINER_DICT[model_type]

def init_running_args(args, ):
    if os.path.exists(args.config):
        running_args = RunningArguments.load_from_file(args.config)
    else:
        running_args = RunningArguments(**vars(args))
    print(running_args)
    return running_args

# def build_dataloader(running_args, meta=None, mode='train', include_labels=True):
#     '''
#     ImageLabelDataset only
#     '''
#     _folder = getattr(running_args, f'{mode}_folder')
#     if not isinstance(meta, pd.DataFrame):
#         _meta = getattr(running_args, f'{mode}_meta')
#         meta = load_meta(_meta, running_args)
#     _policy = getattr(running_args,f'{mode}_policy')
#     folder = os.path.join(_folder, 'images') if 'images' not in _folder else _folder
#     # meta = load_meta(os.path.join(_folder, _meta), running_args)
#     dataset = ImageLabelDataset(folder, meta, include_labels=include_labels, transform=TRANSFORM_DICT[_policy])
#     data_loader = DataLoader(
#                             dataset, batch_size=running_args.batch_size, 
#                             shuffle=True if mode=='train' else False, 
#                             )
#     return data_loader

def load_meta(path, running_args=None):
    try:
        meta = pd.read_csv(path, header=0,)
        if running_args:
            labels = load_label(running_args)
            meta = meta.filter(labels)
        return meta
    except :
        raise FileNotFoundError(f"File meta {path} not exist")

def load_label(running_args):
    l = ['fname']
    if running_args.labels != None:
        for item in running_args.labels:
            l.append(item)
    else:
        l = ['fname', 'mask', 'distancing', '5k']
    return l

def create_logger(name, path=None):
    logger = logging.getLogger(name)
    logFormatter = logging.Formatter('[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s')
    if path == None:
        consoleHandler = logging.StreamHandler()
    else:
        consoleHandler = logging.StreamHandler(path)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)
    return logger