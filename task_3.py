import numpy as np 
from sklearn.preprocessing import MinMaxScaler
#imgs_3 = np.load('C:/Users/micha/OneDrive/Pulpit/hackathon/ensembleAI/data/contestants/DefenseTransformationEvaluate.npz')
imgs_4 = np.load('C:/Users/micha/OneDrive/Pulpit/hackathon/ensembleAI/data/contestants/DefenseTransformationSubmit.npz')

scaler = MinMaxScaler()
res = scaler.fit_transform(imgs_4['representations'])
res = 1 - res
np.savez('C:/Users/micha/OneDrive/Pulpit/hackathon/ensembleAI/data/contestants/DefenseTransformationSubmitSolution.npz', representations=res)

