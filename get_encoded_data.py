import torch
from api_requests import model_stealing
import pickle


from torch.utils.data import Dataset
from typing import Tuple

from PIL import Image
import os
import numpy as np

NUM_OF_SING_QUE=10

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
    dataset = torch.load("data/contestants/ModelStealingPub.pt")
    print(dataset.ids, dataset.imgs, dataset.labels)

    generated = {}
    generated['idx']=[]
    generated['img_idx']=[]
    generated['representations']=[]

    # Create a folder to save the image if it doesn't exist
    folder_path = "images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for idx, (img_idx, img) in enumerate(zip(dataset.ids, dataset.imgs)):
        if (idx % 100 == 0):
            print(f'{idx} / {len(dataset.ids)}')

        # Save the image in the folder
        image_name = str(img_idx) + ".png"
        image_path = os.path.join(folder_path, image_name)
        img.save(image_path)

        vectors = []
        for _ in range(NUM_OF_SING_QUE):
            vector = model_stealing(image_path)
            vectors.append(vector)

        vectors_array = np.array(vectors)
        mean_vector = np.mean(vectors_array, axis=0)
            
        generated['idx'].append(idx)
        generated['img_idx'].append(img_idx)
        generated['representations'].append(mean_vector.tolist())

    with open('generated_data', 'wb') as f:
        pickle.dump(generated, f)
