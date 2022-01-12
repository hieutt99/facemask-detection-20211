# import os, sys
# from torch.utils.data import Dataset
# from PIL import Image
# import torch 
# import numpy as np
# from utils.augmentations import DEFAULT_TEST_TRANSFORM, VAL_VAE_TRANSFORM
# torch.manual_seed(32)

# class ImageLabelDataset(Dataset):
#     def __init__(self, images_folder, meta, transform=DEFAULT_TEST_TRANSFORM, include_labels=True):
#         self.images_folder = images_folder
#         self.meta = meta
#         self.transform = transform
#         self.include_labels = include_labels

#     def __len__(self, ):
#         return len(self.meta)

#     def __getitem__(self, index):
#         image = np.array(Image.open(os.path.join(self.images_folder, 
#                                     self.meta.iloc[index].filter(['fname']).item())))
#         output = self.transform(image=image)
#         output['image'] = output['image']/255
#         if self.include_labels:
#             output['label'] = torch.tensor(self.meta.iloc[index,1:].to_numpy().astype(np.uint8)).float()
#         return output

# class VAEDataset(Dataset):
#     def __init__(self, images_folder, transform=VAL_VAE_TRANSFORM):
#         self.images_folder = images_folder
#         self.transform = transform
#         self.fnames = os.listdir(self.images_folder)

#     def __len__(self):
#         return len(self.fnames)

#     def __getitem__(self, index):
#         image = np.array(Image.open(os.path.join(self.images_folder, self.fnames[index])))
#         output = self.transform(image=image)
#         output['image'] = output['image']/255
#         return output