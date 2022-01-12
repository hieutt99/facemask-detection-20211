import os, sys 
from shutil import copy


CUR_DIR = os.getcwd()
BASE_DIR = CUR_DIR
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
# MODE = 'train'
MODE = 'val'
IMAGE_DIR = os.path.join(DATA_DIR, 'images', MODE)
LABEL_DIR = os.path.join(DATA_DIR, 'labels', MODE)

TGT_DIR = os.path.join(BASE_DIR, 'invalid_dataset')
TGT_IMAGE_DIR = os.path.join(TGT_DIR, 'images', MODE)
TGT_LABEL_DIR = os.path.join(TGT_DIR, 'labels', MODE)


# def read_label(path):
#     '''check empty'''
#     with open(path, 'r') as fp:
#         content = fp.read().strip('\n')
#     if len(content) == 0:
#         # print(path)
#         return False
#     else:
#         return True

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
    ls = content.split('\n')
    for item in ls:
        check = handle_number(item.split(' '))
        if check == None:
            print(ls)
            print(path)
    labels = [handle_number(item.split(' ')) for item in ls]
    return labels

def check_label(label, val=2):
    for l in label:
        if l[0] == val:
            return True
    return False


files = os.listdir(IMAGE_DIR)
print(len(files))


# '''remove empty'''
# cnt = 0
# for f in files:
#     name, _ = os.path.splitext(f)
#     lb_file = f'{name}.txt'
#     lb_path = os.path.join(LABEL_DIR, lb_file)
#     labels = read_label(lb_path)
#     # if check_label(labels):
#     #     cnt+1
#     if not labels:
#         os.remove(os.path.join(IMAGE_DIR, f))
#         os.remove(lb_path)
#         print(f'deleted {lb_path} and {os.path.join(IMAGE_DIR, f)}')

# print(cnt)

# sys.exit()

# =======================================

cnt = 0

temp = []

for f in files:
    name, _ = os.path.splitext(f)
    lb_file = f'{name}.txt'
    lb_path = os.path.join(LABEL_DIR, lb_file)
    labels = read_label(lb_path)
    if check_label(labels):
        cnt+=1
        copy(lb_path, os.path.join(TGT_LABEL_DIR, lb_file))
        copy(os.path.join(IMAGE_DIR, f), os.path.join(TGT_IMAGE_DIR, f))

print(cnt)

temp = list(set(temp))
print(temp)

cnt = 0
for f in files:
    name, _ = os.path.splitext(f)
    lb_file = f'{name}.txt'
    lb_path = os.path.join(LABEL_DIR, lb_file)
    labels = read_label(lb_path)
    if check_label(labels, 0):
        cnt+=1
        copy(lb_path, os.path.join(TGT_LABEL_DIR, lb_file))
        copy(os.path.join(IMAGE_DIR, f), os.path.join(TGT_IMAGE_DIR, f))

    if cnt == 20:
        break