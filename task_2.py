import os
import json
import numpy as np
from typing import List
import requests
import torch
from torchvision.transforms.functional import pil_to_tensor
from dataset import TaskDataset


SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "[paste your team token here]"
TIMEOUT=10000
IDS_NUM = 2000

def sybil_attack(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise Exception("Invalid endpoint")
    
    ENDPOINT = f"/sybil/{binary_or_affine}/{home_or_defense}"
    URL = SERVER_URL + ENDPOINT

    ids = ids = ",".join(map(str, ids))

    response = requests.get(
        URL, params={"ids": ids}, headers={"token": TEAM_TOKEN}, timeout=TIMEOUT
    )

    if response.status_code == 200:
        return json.loads(response.content.decode())["representation"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


def sybil_attack_reset():
    ENDPOINT = "/sybil/reset"
    URL = SERVER_URL + ENDPOINT

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}, timeout=TIMEOUT
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


def prepare_data():
    ids = [i for i in range(IDS_NUM)]
    representations = sybil_attack(ids, "home", "affine")
    return representations

INPUT_DIM = (3, 32, 32)
OUTPUT_DIM = 192

if __name__ == "__main__":
    with open('data/contestants/SybilAttack.pt', 'rb') as f:
        dataset = torch.load(f)

    ids = dataset.ids[:IDS_NUM]
    Xs = [pil_to_tensor(img.convert("RGB")).type(torch.float) for img in dataset.imgs[:IDS_NUM]]
    ys = torch.normal(0, 1, size=(IDS_NUM, OUTPUT_DIM), dtype=torch.float)
    ys = prepare_data(ids)

    loader = torch.utils.data.DataLoader(list(zip(Xs, ys)), batch_size=64, shuffle=True)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 10, 3),
        # torch.nn.BatchNorm2d(16),
        torch.nn.Conv2d(10, 1, 3),
        # torch.nn.BatchNorm2d(16),
        torch.nn.Flatten(),
        torch.nn.ReLU(),
        torch.nn.Linear(784, OUTPUT_DIM)
    )

    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

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
    