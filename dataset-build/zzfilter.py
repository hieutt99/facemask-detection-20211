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

files = os.listdir(train_img_dir)

print(len(files))

def sort_by_value(l, reverse=True):
    l.sort(key=lambda x:x[0], reverse=reverse)
    return l

temp = [(int(os.path.splitext(item)[0]), item) for item in files]
temp = sort_by_value(temp, reverse=False)

# files = [f[1] for f in temp[:800]]
files = [f[1] for f in temp[800:]]

# for f in tqdm(files):
#     name, ext = os.path.splitext(f)
#     lb_file = f'{name}.txt'

#     img_path = os.path.join(train_img_dir, f)
#     lb_path = os.path.join(train_lb_dir, lb_file)

#     tgt_img_path = os.path.join(final_train_img_dir, f)
#     tgt_lb_path = os.path.join(final_train_lb_dir, lb_file)

#     copy(img_path, tgt_img_path)
#     copy(lb_path, tgt_lb_path)
# sys.exit()


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


# data = []

# for f in tqdm(files):
#     name, ext = os.path.splitext(f)
#     lb_file = f'{name}.txt'

#     img_path = os.path.join(train_img_dir, f)
#     lb_path = os.path.join(train_lb_dir, lb_file)

#     image = Image.open(img_path)
#     labels = read_label(lb_path)
#     if check_label(labels):
#         data.append(f)

# print(len(data))

# for f in tqdm(data):
#     name, ext = os.path.splitext(f)
#     lb_file = f'{name}.txt'

#     img_path = os.path.join(train_img_dir, f)
#     lb_path = os.path.join(train_lb_dir, lb_file)

#     tgt_img_path = os.path.join(final_train_img_dir, f)
#     tgt_lb_path = os.path.join(final_train_lb_dir, lb_file)

#     copy(img_path, tgt_img_path)
#     copy(lb_path, tgt_lb_path)

# sys.exit()

data = []

for f in tqdm(files):
    name, ext = os.path.splitext(f)
    lb_file = f'{name}.txt'

    img_path = os.path.join(train_img_dir, f)
    lb_path = os.path.join(train_lb_dir, lb_file)

    image = Image.open(img_path)
    labels = read_label(lb_path)
    if not check_label(labels):
        data.append(f)

files = data
data = []

for f in tqdm(files):
    name, ext = os.path.splitext(f)
    lb_file = f'{name}.txt'

    img_path = os.path.join(train_img_dir, f)
    lb_path = os.path.join(train_lb_dir, lb_file)

    image = Image.open(img_path)
    labels = read_label(lb_path)
    ls = [str(item[0]) for item in labels]
    boxes = [item[1:] for item in labels]
    w,h = image.size

    m = max(w, h)

    # new_boxes = []
    # for box in boxes:
    #     new_boxes.append(convert_bbox(box, w, h))
    
    # new_boxes, new_labels = filter_boxes(new_boxes, labels)

    # lb_str = form_label_string(new_labels)
    # print(lb_str)

    # with open(lb_path, 'w') as fp:
    #     fp.write(lb_str)

    c = count_label(labels)

    x = 0
    # if not m < 600 and m<1000 and c[x]<10 and c[0]<10:
    if not m < 600 and m<1000:
    #     data.append((c[x], img_path, lb_path))

        data.append((c[x], img_path, lb_path))

def sort_by_value(l, val=0):
    l.sort(key=lambda x:x[val], reverse=True)
    return l

data = sort_by_value(data)
print(len(data))

for i in range(10):
    print(data[i])

count = [0,0,0]
for item in data:
    label = read_label(item[-1])
    c = count_label(label)
    count = add(count, c)
print(count)



for item in tqdm(data):
    img_file = os.path.basename(item[1])
    label_file = os.path.basename(item[-1])
    tgt_image_path = os.path.join(final_train_img_dir, img_file)
    tgt_label_path = os.path.join(final_train_lb_dir, label_file)
    # print(tgt_image_path, tgt_label_path)

    # copy(item[1], tgt_image_path)
    # copy(item[-1], tgt_label_path)

    # os.remove(tgt_image_path)
    # os.remove(tgt_label_path)

    name, _ = os.path.splitext(img_file)
    temp = os.path.join(final_train_img_dir, f'{name}.jpg')
    if not os.path.exists(temp):
        copy(item[1], tgt_image_path)
        copy(item[-1], tgt_label_path)

        # os.remove(tgt_image_path)
        # os.remove(tgt_label_path)