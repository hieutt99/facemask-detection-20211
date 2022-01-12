# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import torch 
# torch.manual_seed(32)

# DEFAULT_TRAIN_TRANSFORM = A.Compose([
#         A.SmallestMaxSize(max_size=320),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
#         A.RandomCrop(height=320, width=320),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         ToTensorV2(),
#     ])
# DEFAULT_TEST_TRANSFORM = A.Compose([
#         A.SmallestMaxSize(max_size=320),
#         A.Resize(320, 320),
#         ToTensorV2(),
#     ])

# TRAIN_VAE_TRANSFORM = A.Compose([
#     A.Resize(64, 64),
#     # A.Resize(28, 28),
#     A.augmentations.transforms.GaussNoise(),
#     A.HorizontalFlip(),
#     A.RandomBrightnessContrast(),
#     A.RandomContrast(),
#     A.Sharpen(),
#     ToTensorV2(),
# ])

# VAL_VAE_TRANSFORM = A.Compose([
#     A.Resize(64, 64),
#     ToTensorV2(),
# ])


# TRANSFORM_DICT = {
#     'train':DEFAULT_TRAIN_TRANSFORM,
#     'test':DEFAULT_TEST_TRANSFORM,
#     'train_vae': TRAIN_VAE_TRANSFORM,
#     'val_vae': VAL_VAE_TRANSFORM,
# }