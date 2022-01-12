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

def read_label(path):
    def handle_number(label):

        try:
            l, x, y, w, h = label[0], label[1], label[2], label[3], label[4]
        except:
            print(label)
            return None
        return [int(l), float(x), float(y), float(w), float(h)]
    with open(path, 'r') as fp:
        content = fp.read().strip('\n')
    if len(content) == 0:
        return None
    ls = content.split('\n')
    for item in ls:
        check = handle_number(item.split(' '))
        if check == None:
            print(ls)
            print(path)
    labels = [handle_number(item.split(' ')) for item in ls]
    return labels

def count_label(label):
    count = [0,0,0]
    for l in label:
        count[l[0]]+=1
    return count 

def add(total, label):
    for i, item in enumerate(label):
        total[i] += item
    return total

count = [0,0,0]
for item in files:
    name, ext = os.path.splitext(item)
    fname = f'{name}.txt'
    path = os.path.join(final_train_lb_dir, fname)
    label = read_label(path)
    c = count_label(label)
    count = add(count, c)
print(count)