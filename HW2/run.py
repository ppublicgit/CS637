import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from read_cifar import load, normalize

Xtr_, Ytr_, Xte, Yte = load(r'data/cifar-10-batches-py')

Xtr_ = Xtr_.swapaxes(1,3).swapaxes(2,3)
Xte = Xte.swapaxes(1,3).swapaxes(2,3)
Xtr_, Xte = normalize(Xtr_), normalize(Xte)

class CNN(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


size = Xtr_.shape[0]
num_val = int((1-0.25) * size)
Xtr = Xtr_[:num_val]
Xva = Xtr_[num_val:]
Ytr = Ytr_[:num_val]
Yva = Ytr_[num_val:]

Xtr = torch.Tensor(Xtr)
Xva = torch.Tensor(Xva)
Xte = torch.Tensor(Xte)
Ytr = torch.Tensor(Ytr).type(torch.LongTensor)
Yva = torch.Tensor(Yva).type(torch.LongTensor)
Yte = torch.Tensor(Yte).type(torch.LongTensor)
#class MyDataset(Dataset):
#    def __init__(self, data, targets, transform=None):
#        self.data = data
#        self.targets = torch.LongTensor(targets)
#        self.transform = transform
#
#    def __getitem__(self, index):
#        breakpoint()
#        x = self.data[index]
#        y = self.targets[index]
#
#        if self.transform:
#            breakpoint()
#            x = self.transform(x)
#
#        return x, y
#
#    def __len__(self):
#        return len(self.data)
#
#transform = transforms.Compose([transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                ])
#norm = transform(Xtr)
#breakpoint()
train_dataset = TensorDataset(Xtr, Ytr)
val_dataset = TensorDataset(Xva, Yva)
test_dataset = TensorDataset(Xte, Yte)

# Create data loaders.
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

def save_model_state(model, epoch, model_name):
    """Saves current model state to temp folder to be loaded later during model selection"""
    # save model checkpoint
    torch.save({
                'model_state_dict': model.state_dict(),
                }, f"./temp/model_{model_name}_{epoch}.pth")

def load_state_epoch(model, epoch, model_name):
    """Loads a specified model state from temp folder"""
    # load the model checkpoint
    checkpoint = torch.load(f"./temp/model_{model_name}_{epoch}.pth")
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Previously trained model weights state_dict loaded...')
    return model

def train(dataloader, model, loss_fn, optimizer):
    """Train the model using the loss funciton and optimizer provided.

    Loop over each batch and train the model."""
    size = len(dataloader.dataset)
    tr_loss, tr_correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss_fn(pred, y).item()
        tr_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    tr_loss /= size
    tr_correct /= size
    return tr_loss, tr_correct

def test(dataloader, model, loss_fn):
    """"Get the performance for the model's current state."""
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

device = "cpu"
model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

epochs = 100
tr_acc, tr_loss = [], []
val_acc, val_loss = [], []
for t in range(1, epochs+1):
    print(f"Epoch {t}\n-------------------------------")
    tl, ta = train(train_dataloader, model, loss_fn, optimizer)
    print("Train Error")
    tl, ta = test(train_dataloader, model, loss_fn)
    print("Val Error:")
    vl, va = test(val_dataloader, model, loss_fn)
    tr_acc.append(ta)
    tr_loss.append(tl)
    val_acc.append(va)
    val_loss.append(vl)
    if t % 1 == 0:
        save_model_state(model, t, "simple")
print("Done!")
