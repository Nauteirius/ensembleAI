import os
import numpy as np 
import requests
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from scipy.spatial.distance import cosine as cos_dist

SERVER_URL = os.getenv("SERVER_URL")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")

INPUT_FILE = './data/contestants/DefenseTransformationSubmit.npz'
OUTPUT_FILE = './data/contestants/DefenseTransformationSubmitSolution.npz'

# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.
def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN}, timeout=1000)
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )

#imgs_3 = np.load('C:/Users/micha/OneDrive/Pulpit/hackathon/ensembleAI/data/contestants/DefenseTransformationEvaluate.npz')
imgs_4 = np.load(INPUT_FILE)

scaler = RobustScaler()
res = scaler.fit_transform(imgs_4['representations'])
# scaler = QuantileTransformer(n_quantiles=60)
# res = scaler.fit_transform(imgs_4['representations'])
res = 1 - res
np.savez(OUTPUT_FILE, representations=res)

print(res)
print(np.mean([cos_dist(x, y) for x, y in zip(res, imgs_4['representations'])]))
print(res.dtype)

submit = True
if submit:
    defense_submit(OUTPUT_FILE)
