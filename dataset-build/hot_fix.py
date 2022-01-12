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

files = os.listdir(final_train_img_dir)

print(len(files))

cnt = 0

for f in tqdm(files):
    name, ext = os.path.splitext(f)
    if ext == '.png':
        cnt+=1

        image = Image.open(os.path.join(final_train_img_dir, f))
        image = image.convert('RGB')
        image.save(os.path.join(final_train_img_dir, f'{name}.jpg'))
        os.remove(os.path.join(final_train_img_dir, f))
print(cnt)