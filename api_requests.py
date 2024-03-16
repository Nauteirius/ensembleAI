import os
import json
import requests

TIMEOUT=10000
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000/")
TEAM_TOKEN = os.environ.get("TEAM_TOKEN", "[TOKEN]")

def model_stealing(path_to_png_file: str):
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    with open(path_to_png_file, "rb") as img_file:
        response = requests.get(
            URL, files={"file": img_file}, headers={"token": TEAM_TOKEN}, timeout=TIMEOUT
        )

        if response.status_code == 200:
            return json.loads(response.content.decode())["representation"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


def model_stealing_reset():
    ENDPOINT = "/modelstealing/reset"
    URL = SERVER_URL + ENDPOINT

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}, timeout=TIMEOUT
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


if __name__ == '__main__':
    img = ""
    output = model_stealing(img)
    print(output)
