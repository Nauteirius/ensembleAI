import torch
from api_requests import model_stealing
import pickle


from torch.utils.data import Dataset
from typing import Tuple

from PIL import Image
import os

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
    generated['obrazki']=[]
    generated['reprezentacje']=[]
    generated['id']=[]

    for idx, (img_idx, img) in enumerate(zip(dataset.ids, dataset.imgs)):

        #-------------------------------------------------------------------------------------------------

        if (idx % 100 == 0):
            print(f'{idx} / {len(dataset.ids)}')


        # Load your image
        #image = Image.open(img)  # Replace "example.jpg" with the path to your image

        # Create a folder to save the image if it doesn't exist
        folder_path = "images"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        
        # Save the image in the folder
        image_name = str(img_idx) + ".png"  # Specify the name for the saved image
        image_path = os.path.join(folder_path, image_name)
        img.save(image_path)

        # Now, you have the path to the saved image
        #print("Image saved to:", image_path)







        #------------------------------------------------------------------------------------------------


        encoding = model_stealing(image_path)
        #generated.append((idx, img_idx, encoding))
        
        generated['obrazki'].append(img_idx)
        generated['reprezentacje'].append(encoding)
        generated['id'].append(idx)

    with open('generated_data', 'wb') as f:
        pickle.dump(generated, f)
