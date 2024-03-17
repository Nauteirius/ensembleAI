import os
import json
import numpy as np
from typing import List
import requests
import torch
import pickle
from torchvision.transforms.functional import pil_to_tensor
from dataset import TaskDataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


SERVER_URL = "http://127.0.0.1:5000"
TEAM_TOKEN = "[paste your team token here]"
TIMEOUT=10000
IDS_NUM = 13000

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, output, label):
        mse = F.mse_loss(output, label)
        batch_size = label.size(0)
        return mse / batch_size

def prepared_data():
    with open('generated_data', 'rb') as f:
        data = pickle.load(f)
    
    # returns {['img_idx'], ['idx'], ['representation']} - check get_encoded_data.py
    return data

INPUT_DIM = (3, 32, 32)
OUTPUT_DIM = 512

if __name__ == "__main__":
    with open('data/contestants/ModelStealingPub.pt', 'rb') as f:
        dataset = torch.load(f)

    ids = dataset.ids[:IDS_NUM]
    Xs = [pil_to_tensor(img.convert("RGB")).type(torch.float) for img in dataset.imgs[:IDS_NUM]]
    # ys = torch.tensor([float(label) for label in dataset.labels], dtype=torch.float)
    ys = torch.tensor(prepared_data()['representations'], dtype=torch.float)

    loader = torch.utils.data.DataLoader(list(zip(Xs, ys)), batch_size=64, shuffle=True)
    
    # Define the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Adjust the first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpooling to maintain spatial size

    # Adjust the final fully connected layer to output 512 features
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, OUTPUT_DIM)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = CustomMSELoss()

    # for x, y in zip(Xs, ys):
    for x, y in loader:
        opt.zero_grad()
        outputs = model(x)
        print(outputs.shape)
        print(y.shape)
        loss = loss_fn(outputs, y)
        loss.backward()
        opt.step()
        
    
    np.savez(
        "example_submission.npz",
        ids=np.random.permutation(20000),
        representations=np.random.randn(20000, 192),
    )
    
    # Check model
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = F.mse_loss(outputs, y, reduction='sum')
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader.dataset)
    print(avg_loss)
    