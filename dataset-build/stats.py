import os, sys 
from shutil import copy 


CUR_DIR = os.getcwd()
BASE_DIR = CUR_DIR
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

train_img_dir = os.path.join(DATA_DIR, 'images', 'train')
train_lb_dir = os.path.join(DATA_DIR, 'labels', 'train')

files = os.listdir(train_img_dir)

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

def check_label(label, value=2):
    for l in label:
        if l[0] ==2:
            return True 

    return False

data = []
for f in files:
    name, ext = os.path.splitext(f)
    lb_file = f'{name}.txt'

    img_path = os.path.join(train_img_dir, f)
    lb_path = os.path.join(train_lb_dir, lb_file)

    label = read_label(lb_path)

    c = count_label(label)

    data.append((c[2], img_path, lb_path))

def sort_by_value(l):
    l.sort(key=lambda x:x[0], reverse=True)
    return l

data = sort_by_value(data)

# for i in range(10):
#     print(data[i])


# data = data[:500]

count = [0,0,0]
for item in data:
    label = read_label(item[-1])
    c = count_label(label)
    count = add(count, c)
print(count)

