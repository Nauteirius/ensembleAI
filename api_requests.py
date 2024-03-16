import requests

SERVER_URL = "[paste server url here]"
TEAM_TOKEN = "[paste your team token here]"

def model_stealing(path_to_png_file: str):
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    with open(path_to_png_file, "rb") as img_file:
        response = requests.get(
            URL, files={"file": img_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return response.content["representation"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")

def model_stealing_reset():
    ENDPOINT = "/modelstealing/reset"
    URL = SERVER_URL + ENDPOINT

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
