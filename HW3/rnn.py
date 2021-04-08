#importing the libraries
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn

print("imports worked")

#creating the dataset
x = np.arange(1,721,1)
y = np.sin(x*np.pi/180)  + np.random.randn(720)*0.05
plt.plot(y)

# structuring the data
X = []
Y = []
for i in range(0,710):
     list1 = []
     for j in range(i,i+10):
         list1.append(y[j])
     X.append(list1)
     Y.append(y[j+1])


#train test split
X = np.array(X)
Y = np.array(Y)
x_train = X[:360]
x_test = X[360:]
y_train = Y[:360]
y_test = Y[360:]


class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len

dataset = timeseries(x_train,y_train)
train_loader = DataLoader(dataset,shuffle=True,batch_size=256)

class neural_network(nn.Module):
    def __init__(self):
        super(neural_network,self).__init__()
        self.lstm = nn.LSTM(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(in_features=5,out_features=1)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = output[:,-1,:]
        output = self.fc1(torch.relu(output))
        return output

model = neural_network()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
epochs = 1500

for i in range(epochs):
    for j,data in enumerate(train_loader):
        y_pred = model(data[:][0].view(-1,10,1)).reshape(-1)
        loss = criterion(y_pred,data[:][1])
        loss.backward()
        optimizer.step()
    if i%50 == 0:
        print(i,"th iteration : ",loss)


test_set = timeseries(x_test,y_test)
test_pred = model(test_set[:][0].view(-1,10,1)).view(-1)
plt.plot(test_pred.detach().numpy(),label='predicted')
plt.plot(test_set[:][1].view(-1),label='original')
plt.legend()
