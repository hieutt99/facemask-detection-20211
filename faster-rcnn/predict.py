from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch 
from PIL import Image 
import numpy as np 

model_path = ''
num_classes = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()


image_path = ""
image = Image.open(image_path)

# làm nào mà nó thành ảnh tensor scale 0-1 thì làm nhé <3 


images = [image.to(device)]
with torch.no_grad():
    predictions = model(images)

# predictions có dạng [{'boxes': torch.Tensor(), 'labels': torch.Tensor(), 'scores': torch.Tensor()}]

print(predictions)