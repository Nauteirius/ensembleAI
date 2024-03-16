import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Załadowanie danych z pliku pickle
with open('data.pickle', 'rb') as f:
    dane = pickle.load(f)

class MojeDane(Dataset):
    def __init__(self, dane, transform=None):
        self.obrazki = torch.tensor(dane['obrazki'], dtype=torch.float32)
        self.reprezentacje = torch.tensor(dane['reprezentacje'], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.obrazki)

    def __getitem__(self, idx):
        obrazek = self.obrazki[idx]
        reprezentacja = self.reprezentacje[idx]

        if self.transform:
            obrazek = self.transform(obrazek)

        return obrazek, reprezentacja
    

    



#class 
# Definicja modelu
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(rozmiar_wej, rozmiar_wyj)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Inicjalizacja modelu, funkcji straty i optymalizatora
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



x = 5 #co ile bedzie nasz przyklad walidacyjny x=5 to znaczy 1/5 to przyklady walidacyjne

# Tworzenie indeksów dla danych treningowych i walidacyjnych
indeksy_treningowe = []
indeksy_walidacyjne = []
for i in range(len(dane)):
    if i % x == 0:
        indeksy_walidacyjne.append(i)
    else:
        indeksy_treningowe.append(i)



# Definicja transformacji danych treningowych (augmentacja)
transformacje_treningowe = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Tworzenie obiektu MojeDane dla danych treningowych z augmentacją
# Tworzenie obiektów MojeDane dla danych treningowych i walidacyjnych
dane_treningowe = MojeDane(dane[indeksy_treningowe],transform=transformacje_treningowe)
dane_walidacyjne = MojeDane(dane[indeksy_walidacyjne],transform=transformacje_treningowe)

# Ustawienie hiperparametrów
liczba_epok = 10
rozmiar_batch = 32



# Tworzenie obiektu DataLoader dla danych treningowych
loader_treningowy = DataLoader(dane_treningowe, batch_size=rozmiar_batch, shuffle=True)

# Pętla uczenia
for epoka in range(liczba_epok):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(DataLoader(dane_treningowe, batch_size=rozmiar_batch, shuffle=True)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoka [{epoka+1}/{liczba_epok}], Krok [{i+1}/{len(dane_treningowe)//rozmiar_batch}], Strata: {running_loss/100:.4f}')
            running_loss = 0.0
    
    # Walidacja modelu po każdej epoce
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in DataLoader(dane_walidacyjne, batch_size=rozmiar_batch):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f'Epoka [{epoka+1}/{liczba_epok}], Strata walidacji: {total_loss/len(dane_walidacyjne):.4f}')

print('Uczenie zakończone!')