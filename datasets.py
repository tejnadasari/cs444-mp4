import torchvision as tv
import numpy as np
import pickle
from custom_dataset import CustomDataset

def get_transforms(split, size):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 224:
        resize_dim = 256
        crop_dim = 224
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToPILImage(),
                tv.transforms.Resize(resize_dim, antialias=True),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToPILImage(),
                tv.transforms.Resize(resize_dim, antialias=True),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform


def get_flower102(flower_dir,split):


    with open(f'{flower_dir}/{split}_data.pkl', 'rb') as f:
        X, y = pickle.load(f)

    if split == 'test':
        return CustomDataset(X,y,get_transforms('test',224))
    return CustomDataset(X,y,get_transforms('train',224))
