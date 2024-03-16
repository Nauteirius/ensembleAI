import os
import json
import numpy as np
from typing import List
import requests
import torch
from torchvision.transforms.functional import pil_to_tensor
from dataset import TaskDataset
import pickle


SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "[TOKEN]"
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
        return json.loads(response.content.decode())["representations"]
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



# Be careful. This can be done only 4 times an hour.
# Make sure your file has proper content.
def sybil_submit(path_to_npz_file: str, binary_or_affine: str):
    if binary_or_affine not in ["binary", "affine"]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/submit"
    url = SERVER_URL + endpoint

    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN}, timeout=TIMEOUT)

    if response.status_code == 200:
        print("OK")
        print(response.json())
    else:
        print(
            f"Request submit failed. Status code: {response.status_code}, content: {response.json()}"
        )


def prepare_data(ids):
    # ids = [i for i in range(IDS_NUM)]
    representations = sybil_attack(ids, "home", BINARY_OR_AFFINE)
    return representations

BINARY_OR_AFFINE = "affine"
INPUT_DIM = (3, 32, 32)
OUTPUT_DIM = 192

if __name__ == "__main__":
    
    reset = False
    if reset:
        sybil_attack_reset()
    
    
    with open('data/contestants/SybilAttack.pt', 'rb') as f:
        dataset = torch.load(f)

    ids = dataset.ids[:IDS_NUM]
    Xs = [pil_to_tensor(img.convert("RGB")).type(torch.float) for img in dataset.imgs[:IDS_NUM]]
    # ys = torch.normal(0, 1, size=(IDS_NUM, OUTPUT_DIM), dtype=torch.float)
    
    print("requesting representations")
    ys = prepare_data(ids)
    # print(ys)
    # with open("ys.pickle", 'wb') as f:
    #     pickle.dump(ys, f)

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

    print("start training")

    # for x, y in zip(Xs, ys):
    for x, y in loader:
        opt.zero_grad()
        outputs = model(x)
        # print(outputs.shape)
        # print(y.shape)
        loss = loss_fn(outputs, y)
        loss.backward()
        opt.step()
        
    SOLUTION_FILE = "example_submission.npz"
    np.savez(
        SOLUTION_FILE,
        ids=np.random.permutation(20000),
        representations=np.random.randn(20000, 192),
    )
    print("subbmit solution")
    sybil_submit(SOLUTION_FILE, BINARY_OR_AFFINE)
