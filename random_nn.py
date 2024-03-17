import requests
import torch

OUT_DIM = 512
SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "zZ9HuhBABqiNLD7i"
OUTPUT_FILE = 'model.onnx'

def model_stealing_submission(path_to_onnx_file: str):
    ENDPOINT = "/modelstealing/submit"
    URL = SERVER_URL + ENDPOINT

    with open(path_to_onnx_file, "rb") as onnx_file:
        response = requests.post(
            URL, files={"file": onnx_file}, headers={"token": TEAM_TOKEN}, timeout=10000
        )

        if response.status_code == 200:
            return response.content["score"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")

if __name__ == '__main__':
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 1, 3),
        torch.nn.Flatten(),
        torch.nn.Linear(900, OUT_DIM)
    )
    x = torch.normal(0, 1, size=(1, 3, 32, 32), dtype=torch.float)
    y = model(x)
    print(y.shape)
    
    torch.onnx.export(model, torch.Tensor(1,3,32,32), OUTPUT_FILE)    
    model_stealing_submission(OUTPUT_FILE)

