import os
import json
from PIL import Image

import torch
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
}

def get_classes_dict(root='./CUB-200'):
    classes_dict = {}
    with open(os.path.join(root, 'lists', 'classes.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            classes_dict[line] = i
    return classes_dict


class CUB200Dataset(torch.utils.data.Dataset):
    def __init__(self, root='./CUB-200', train=True):
        self.train = train
        self.classes_dict = get_classes_dict(root=root)
        self.img_paths = []
        self.labels = []
        with open(os.path.join(root, 'lists', 'train.txt' if train else 'test.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                self.img_paths.append(os.path.join(root, 'images', line))
                self.labels.append(self.classes_dict[line.split('/')[0]])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = data_transforms['train' if self.train else 'test'](img)
        return img, self.labels[index]
        

if __name__ == "__main__":
    data = CUB200Dataset(train=False)
    print(len(data))