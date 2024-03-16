import torch
from api_requests import model_stealing
import pickle

if __name__ == '__main__':
    dataset = torch.load("modelstealing/data/ExampleModelStealingPub.pt")
    print(dataset.ids, dataset.imgs, dataset.labels)

    generated = []
    for idx, (img_idx, img) in enumerate(zip(dataset.ids, dataset.imgs)):
        encoding = model_stealing(img)
        generated.append((idx, img_idx, encoding))

    with open('generated_data') as f:
        pickle.dump(generated, f)
