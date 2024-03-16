import torch
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = torch.load("modelstealing/data/ExampleModelStealingPub.pt")
    print(dataset.ids, dataset.imgs, dataset.labels)

    for i in range(10):
        plt.show(dataset.imgs[i])        
