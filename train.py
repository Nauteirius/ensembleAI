import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from get_encoded_data import TaskDataset


# Załadowanie danych z pliku pickle
with open('generated_data', 'rb') as f:
    data = pickle.load(f)

class OurData(Dataset):
    def __init__(self, dane, lista_id, transform=None):
        
        self.imgid = list([dane['obrazki'][x] for x in lista_id])#, dtype=torch.float32) #to sa id obrazka
        self.dataset = torch.load("data/contestants/ModelStealingPub.pt")
        #self.obrazki=self.dataset.imgs[*self.imgid]
        #self.obrazki=self.dataset.imgs[[item for sublist in self.imgid for item in sublist]]
        #flattened_list = [item for sublist in nested_list for item in sublist]
        self.images=[self.dataset.imgs[rel_idx] for rel_idx, idx in enumerate(self.imgid)]

        self.reprezentations = torch.tensor(dane['reprezentacje'], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.imgid)

    def __getitem__(self, idx):
        image = self.images[idx]
        representation = self.reprezentations[idx]

        if self.transform:
            image = self.transform(image)

        return image, representation
    

    



#class 
# Definicja modelu
    
#import api
# class Model(nn.Module):
#     model = api.ModelApi.setup_model()
from torchvision.models import resnet50, ResNet50_Weights


#own loss
import torch.nn.functional as F

class CustomLossFunction(torch.nn.Module):
    def __init__(self):
        super(CustomLossFunction, self).__init__()

    def forward(self, output, labels):
        batch_size = output.size(0)
        #print(f'{output.shape} -output_shape, {labels.shape} - labels.shape')
        mse = F.mse_loss(output, labels)
        return mse / batch_size
    

class UnsqueezeModule(torch.nn.Module):
    def __init__(self, dim):
        super(UnsqueezeModule, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)
# Inicjalizacja modelu, funkcji straty i optymalizatora
#model = Model()
    
class Model(nn.Module):
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            UnsqueezeModule(1),              
            torch.nn.Identity())
            #nn.ReLU(inplace=True),
            #nn.Linear(512, output_size)
        #)

    def forward(self, x):
        return self.resnet(x)

#model = resnet50(pretrained=True)#(weights=ResNet50_Weights.DEFAULT)
model = Model(output_size=512)
# num_features = model.fc.in_features
# model.fc = torch.nn.Sequential(torch.nn.Linear(num_features,512),
#     UnsqueezeModule(1),              
#     torch.nn.Identity())



criterion = nn.MSELoss()#CustomLossFunction()#nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.000001)#optim.Adam(model.parameters(), lr=0.00001)



x = 5 #co ile bedzie nasz przyklad walidacyjny x=5 to znaczy 1/5 to przyklady walidacyjne

# Tworzenie indeksów dla danych treningowych i walidacyjnych
index_training = []
index_validation = []
for i in range(len(data)):
    if i % x == 0:
        index_validation.append(i)
    else:
        index_training.append(i)



# Definicja transformacji danych treningowych (augmentacja)
training_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Tworzenie obiektu MojeDane dla danych treningowych z augmentacją
# Tworzenie obiektów MojeDane dla danych treningowych i walidacyjnych
print(data['obrazki'][1])
data_training = OurData(data, index_training, transform=training_transforms)
data_validation = OurData(data, index_validation ,transform=training_transforms)

# Ustawienie hiperparametrów
epochs = 1000
batch_size = 256



input_shape=0

loader_training = DataLoader(data_training, batch_size=batch_size, shuffle=True)
loader_validation = DataLoader(data_validation, batch_size=batch_size)
print("uczy sie")
# Pętla uczenia
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader_training):
        optimizer.zero_grad()
        input_shape=inputs.shape

        outputs = model(inputs)
        #outputs.unsqueeze(1).unsqueeze(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_training)//batch_size}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    # Walidacja modelu po każdej epoce
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader_validation:
            outputs = model(inputs)
            #outputs.unsqueeze(1).unsqueeze(-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Validation loss: {total_loss/len(data_validation):.4f}')

# torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss/len(loader_validation)
#     }, f'model_checkpoint_epoch_{epoch}.pt')

onnx_program = torch.onnx.export(model, torch.Tensor(2,3,32,32), 'resnet.onnx')
print('Uczenie zakończone!')
