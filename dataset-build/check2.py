import os, sys 
from shutil import copy
from typing_extensions import final 
from PIL import Image
from tqdm.auto import tqdm 


CUR_DIR = os.getcwd()
BASE_DIR = CUR_DIR
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

final_data_dir = os.path.join(BASE_DIR, 'final_dataset')

train_img_dir = os.path.join(DATA_DIR, 'images', 'train')
train_lb_dir = os.path.join(DATA_DIR, 'labels', 'train')


final_train_img_dir = os.path.join(final_data_dir, 'images', 'train')
final_train_lb_dir = os.path.join(final_data_dir, 'labels', 'train')

files = os.listdir(final_train_lb_dir)

print(len(files))


for f in files:
    name, ext = os.path.splitext(f)

    t1 = os.path.join(final_train_img_dir, f'{name}.png')
    t2 = os.path.join(final_train_img_dir, (f'{name}.jpg'))
    if not (os.path.exists(t1) or os.path.exists(t2)):
        print(f)