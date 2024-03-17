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
TEAM_TOKEN = "zZ9HuhBABqiNLD7i"
TIMEOUT=1000000
# IDS_NUM = 2
IDS_OFFSET = 0
IDS_NUM = 1000
SOLUTION_IDS_NUM = 20000


def sybil(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in [
        "binary",
        "affine",
    ]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/{home_or_defense}"
    url = SERVER_URL + endpoint
    ids = ",".join(map(str, ids))
    response = requests.get(url, params={"ids": ids}, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        representations = response.json()["representations"]
        ids = response.json()["ids"]
        return representations
    else:
        raise Exception(
            f"Sybil failed. Code: {response.status_code}, content: {response.json()}"
        )


# Be careful. This can be done only 4 times an hour.
# Make sure your file has proper content.
def sybil_submit(binary_or_affine: str, path_to_npz_file: str):
    if binary_or_affine not in ["binary", "affine"]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/submit"
    url = SERVER_URL + endpoint

    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})

    if response.status_code == 200:
        print("OK")
        print(response.json())
    else:
        print(
            f"Request submit failed. Status code: {response.status_code}, content: {response.json()}"
        )


def sybil_reset(binary_or_affine: str, home_or_defense: str):
    if binary_or_affine not in ["binary", "affine"]:
        raise Exception("Invalid endpoint")
    
    if home_or_defense not in ["home", "defense"]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/reset/{home_or_defense}"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print("Request ok")
        print(response.json())
    else:
        raise Exception(
            f"Sybil reset failed. Code: {response.status_code}, content: {response.json()}"
        )


def prepare_data(ids):
    # ids = [i for i in range(IDS_NUM)]
    representations = sybil(ids, "home", BINARY_OR_AFFINE)
    return representations

# def reverse_affine(points):


BINARY_OR_AFFINE = "affine"
INPUT_DIM = (3, 32, 32)
OUTPUT_DIM = 384

if __name__ == "__main__":

    reset = False
    if reset:
        sybil_reset(BINARY_OR_AFFINE, "home")
        exit(0)

    with open('data/contestants/SybilAttack.pt', 'rb') as f:
        dataset = torch.load(f)
    ids = dataset.ids[IDS_OFFSET:IDS_NUM + IDS_OFFSET]
    
    download_data = False
    if download_data:
        print("requesting representations")
        ys = prepare_data(ids)
        with open('ys_data.pickle', 'wb') as f:
            pickle.dump(ys, f)
        print(ys)
        print(type(ys))
        exit(0)
    else:
        with open('ys_data.pickle', 'rb') as f:
            ys = pickle.load(f)
    ys = [torch.tensor(y) for y in ys]
    # print([y.shape for y in ys])
    print(len(ys))

    Xs = [pil_to_tensor(img.convert("RGB")).type(torch.float) for img in dataset.imgs[IDS_OFFSET:IDS_NUM + IDS_OFFSET]]
    # ys = torch.normal(0, 1, size=(IDS_NUM, OUTPUT_DIM), dtype=torch.float)

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
        
    # TODO if affine then
    # linalg.solve(B)
        
    # generate output
    SOLUTION_FILE = "submission.npz"
    
    ids = dataset.ids[:SOLUTION_IDS_NUM]
    Xs = [pil_to_tensor(img.convert("RGB")).type(torch.float) for img in dataset.imgs[:SOLUTION_IDS_NUM]]
    outputs = [model(x).detach().numpy() for x in Xs]
        
    np.savez(
        SOLUTION_FILE,
        ids=np.array(ids),
        representations=np.array(outputs).squeeze(1)
    )
    print("subbmit solution")
    sybil_submit(BINARY_OR_AFFINE, SOLUTION_FILE)
