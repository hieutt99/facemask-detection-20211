import os, sys
from re import L
from torch.optim import lr_scheduler

from torch.utils import data
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from utils.functions import *
from utils.arguments import *
from yolov5_utils.datasets import create_dataloader
import ruamel.yaml as yaml

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ===================================================================================
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=os.path.join(os.getcwd(), 'configs', 'train_config.yaml'), type=str)
parser.add_argument('--do_train', nargs='?', type=str2bool, const=True, default=False)
parser.add_argument('--do_predict', nargs='?', type=str2bool, const=True, default=False)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--train_folder', default='data/train', type=str)
parser.add_argument('--test_folder', default='data/public_test', type=str)
parser.add_argument('--train_policy', default='train', type=str)
parser.add_argument('--test_policy', default='test', type=str)
parser.add_argument('--labels', nargs='+', default=None)

args = parser.parse_args()
# ===================================================================================
running_args = init_running_args(args)

data_args = DataArguments.load_from_file(running_args.data_config)


hyp_path = os.path.join(os.getcwd(), 'configs', 'hyps', 'hyp_finetune.yaml')
with open(hyp_path, 'r') as fp:
    hyp = yaml.safe_load(fp)
print(hyp)

# ===================================================================================
if running_args.do_train:
    train_loader, dataset = create_dataloader(data_args.train, imgsz=640, batch_size=running_args.batch_size,
                                augment=True, hyp=hyp, stride=32, image_weights=True, prefix="Train: ")

val_loader, dataset = create_dataloader(data_args.val, imgsz=640, batch_size=running_args.batch_size,
                                augment=False, stride=32, image_weights=True, prefix="Eval: ")

test_loader, dataset = create_dataloader(data_args.test, imgsz=640, batch_size=running_args.batch_size,
                                augment=False, stride=32, image_weights=True, prefix="Eval: ")



import torch
from models.utils import create_model

model = create_model(running_args.model_type, num_classes=data_args.num_classes)


trainer = get_trainer(running_args.model_type)(model, running_args.train_args, data_args=data_args)
# trainer.train(train_loader)

# trainer.train(train_loader)
trainer.eval(test_loader)

# ===================================================================================


if __name__ == '__main__':

    print("ok")

