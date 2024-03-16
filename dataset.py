import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Tuple
import torch

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    with open("data/contestants/ModelStealingPub.pt", 'rb') as f:
        dataset = torch.load(f)
    print(dataset.ids, dataset.imgs, dataset.labels)

    for i in range(10):
        plt.imshow(dataset.imgs[i])
        plt.savefig(f"temp{i}")