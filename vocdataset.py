import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

        self.image_paths, self.labels = self.parse_annotations()

    def parse_annotations(self):
        image_paths = []
        labels = []

        with open(f'{self.root_dir}/ImageSets/Main/{self.image_set}.txt', 'r') as file:
            lines = file.readlines()

        for line in lines:
            image_id = line.strip()
            image_path = f'{self.root_dir}/JPEGImages/{image_id}.jpg'
            annotation_path = f'{self.root_dir}/Annotations/{image_id}.xml'

            label = [0] * len(self.classes)

            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                label[self.classes.index(class_name)] = 1

            image_paths.append(image_path)
            labels.append(label)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.FloatTensor(self.labels[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label




