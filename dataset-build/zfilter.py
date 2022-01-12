import os, sys 
from shutil import copy 
from PIL import Image
from tqdm.auto import tqdm 


CUR_DIR = os.getcwd()
BASE_DIR = CUR_DIR
DATA_DIR = os.path.join(BASE_DIR, 'aug_dataset')

final_data_dir = os.path.join(BASE_DIR, 'final_dataset')

train_img_dir = os.path.join(DATA_DIR, 'images', 'train')
train_lb_dir = os.path.join(DATA_DIR, 'labels', 'train')

final_train_img_dir = os.path.join(final_data_dir, 'images', 'train')
final_train_lb_dir = os.path.join(final_data_dir, 'labels', 'train')


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

cnt = 0
count = [0, 0, 0]
for f in files:
    name, ext = os.path.splitext(f)
    lb_file = f'{name}.txt'

    img_path = os.path.join(train_img_dir, f)
    lb_path = os.path.join(train_lb_dir, lb_file)

    label = read_label(lb_path)
    if label == None:
        os.remove(img_path)
        os.remove(lb_path)
        print(f"Remove empty {img_path}-{lb_path}")
    else:
        c = count_label(label)
        count = add(count, c)

    check = check_label(label)
    if not check:
        os.remove(img_path)
        os.remove(lb_path)
        print(f"Remove not incorrectly wearing {img_path}-{lb_path}")
    else:
        cnt += 1

def convert_bbox(box, img_width, img_height):
    x_center = box[0]*img_width
    y_center = box[1]*img_height
    box_width = box[2]*img_width
    box_height = box[3]*img_height
    x_min = int(x_center - box_width/2)
    y_min = int(y_center - box_height/2)
    x_max = int(x_min + box_width)
    y_max = int(y_min + box_height)
    box = [x_min, y_min, x_max, y_max]
    return box
def filter_boxes(boxes, labels=None):
    new_boxes = []
    new_labels = []
    for index, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        w = xmax - xmin
        h = ymax - ymin
        if (not (w<15 or h<15) and not (w<h/3 or h<w/3)):
            new_boxes.append(box)
            if labels:
                new_labels.append(labels[index])
    return new_boxes, new_labels
data = []

def form_label_string(labels):
    s = []
    for l in labels:
        l = [str(item) for item in l]
        s.append(' '.join(l))
    return '\n'.join(s)+'\n'
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

    # new_boxes = []
    # for box in boxes:
    #     new_boxes.append(convert_bbox(box, w, h))
    
    # new_boxes, new_labels = filter_boxes(new_boxes, labels)

    # lb_str = form_label_string(new_labels)
    # # print(lb_str)

    # with open(lb_path, 'w') as fp:
    #     fp.write(lb_str)

    c = count_label(labels)

    # if c[0] < 5 and c[0] > 1 and c[1]<5:
    #     data.append((c[2], img_path, lb_path))

    data.append((c[2], img_path, lb_path))

def sort_by_value(l):
    l.sort(key=lambda x:x[0], reverse=True)
    return l

data = sort_by_value(data)

for i in range(10):
    print(data[i])

print(len(data))
data = data[:479]

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

    name, _ = os.path.splitext(img_file)
    temp = os.path.join(final_train_img_dir, f'{name}.jpg')
    if not os.path.exists(temp):
        copy(item[1], tgt_image_path)
        copy(item[-1], tgt_label_path)

        # os.remove(tgt_image_path)
        # os.remove(tgt_label_path)