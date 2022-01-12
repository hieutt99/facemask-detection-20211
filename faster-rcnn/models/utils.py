import torch.nn.functional as F
from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def interpolate(image, size=224):
    return F.interpolate(
                        image, 
                        size=size, 
                        mode='bilinear', 
                        align_corners=True
                        )

def crop_images(images, bboxes, interpolate=True):
    pass 

MODEL_DICT = {
    'fasterrcnn_resnet50_fpn':fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn':fasterrcnn_mobilenet_v3_large_fpn,
}

def create_model(model_name, num_classes=3):
    if model_name in MODEL_DICT.keys():
        model =  MODEL_DICT[model_name](pretrained = True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    return None