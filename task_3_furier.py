import os
import numpy as np 
import requests
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from scipy.fft import fft

SERVER_URL = os.getenv("SERVER_URL", 'http://34.71.138.79:9090')
TEAM_TOKEN = os.getenv("TEAM_TOKEN", '[TOKEN]')

INPUT_FILE = './data/contestants/DefenseTransformationSubmit.npz'
OUTPUT_FILE = './data/contestants/DefenseTransformationSubmitSolution.npz'

# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.
def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN}, timeout=10000)
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )

#imgs_3 = np.load('C:/Users/micha/OneDrive/Pulpit/hackathon/ensembleAI/data/contestants/DefenseTransformationEvaluate.npz')
imgs_4 = np.load(INPUT_FILE)
print(imgs_4['representations'].shape)

res = fft(imgs_4['representations'])
res = np.abs(res)
print(res.shape)

np.savez(OUTPUT_FILE, representations=res)

print(res)

submit = False
if submit:
    defense_submit(OUTPUT_FILE)
