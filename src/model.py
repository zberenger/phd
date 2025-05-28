import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

# Dataset class
class Data(Dataset):
    def __init__(self, X_t, Y_t):
        self.X = X_t.reshape((X_t.shape[0], -1))
        self.Y = Y_t
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):      
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


# Neural network architecture
class Net(nn.Module):
    def __init__(self, input, output, latent_space_size=5, H1=100, H2=80, H3=50, H5=50, H6=80, H7=100):
        super(Net,self).__init__()
        # encoder
        self.linear1 = nn.Linear(input,H1,bias=False)
        self.linear2 = nn.Linear(H1,H2,bias=False)
        self.linear3 = nn.Linear(H2,H3,bias=False)
        self.linear4 = nn.Linear(H3,latent_space_size,bias=False)

        # decoder
        self.linear5 = nn.Linear(latent_space_size,H5,bias=False)
        self.linear6 = nn.Linear(H5,H6,bias=False)
        self.linear7 = nn.Linear(H6,H7,bias=False)
        self.linear8 = nn.Linear(H7,output,bias=False)
    
    def encoder(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        return x

    def decoder(self, x):
        x = F.leaky_relu(self.linear5(x))
        x = F.leaky_relu(self.linear6(x))
        x = F.leaky_relu(self.linear7(x))
        x = F.leaky_relu(self.linear8(x))
        # x = F.relu(self.linear8(x))
        # x = F.softmax(self.linear8(x), dim=-1)
        ## AFTER
        x = x / (x.sum(dim=-1, keepdim=True)+1e-5)
        ## BEFORE
        # x = self.linear8(x)
        # x = F.softmax((x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True), dim=-1)
        return x
        
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z
    
    
class Net_comp(nn.Module):
    def __init__(self, input, output, H1=200, H2=100, H3=200):
        super(Net_comp,self).__init__()
        # encoder
        self.linear1 = nn.Linear(input,H1,bias=False)
        self.linear2 = nn.Linear(H1,H2,bias=False)
        self.linear25 = nn.Linear(H2,H2,bias=False)
        self.linear3 = nn.Linear(H2,H3,bias=False)
        self.linear4 = nn.Linear(H3,output,bias=False)


    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear25(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.softmax(self.linear4(x), dim=-1)
        return x
    

def preprocessing_data_for_nn(input, ground_truth, test_size=0.25, batch_size=32):
    torch.manual_seed(42)
    indices = np.arange(input.shape[0])
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(input, ground_truth, indices, test_size=test_size, random_state=42)

    # pytorch variables
    X_train = torch.from_numpy(X_train.astype(np.float32))
    Y_train = torch.from_numpy(np.asarray(Y_train).astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    Y_test = torch.from_numpy(np.asarray(Y_test).astype(np.float32))

    # create dataset
    trainloader = DataLoader(dataset=Data(X_train, Y_train), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=Data(X_test, Y_test), batch_size=batch_size, shuffle=True)

    return trainloader, testloader, X_train, Y_train, X_test, Y_test, indices_train, indices_test

def training(trainloader, X_test, Y_test, net, nb_epochs=100, criterion=nn.MSELoss(), optimizer=None, display=True):
  if optimizer == None:
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001)

  losses = []
  test_losses = []
  epoch = 0
  
  while epoch < nb_epochs:
    # print(epoch)
    running_loss = 0.0
    nb_loss = 0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
    #   inputs, labels = inputs.to(device), labels.to(device)

      # zero the gradient buffers
      optimizer.zero_grad()   

      # optimisation
      output, _ = net(inputs)
      loss = criterion(output, labels)
      # loss = criterion(output, labels, torch.ones(inputs.shape[0]))
      loss.backward()
      optimizer.step()    # Does the update

      # statistics
      running_loss += loss.item()
      nb_loss += 1
        
    if epoch%(nb_epochs//min(20, nb_epochs//2)) == 0:
      # test loss
      with torch.no_grad():
        outputs_test, _ = net(X_test.reshape((X_test.shape[0], -1)))
        # test_loss = criterion(outputs_test, Y_test, torch.ones(Y_test.shape[0]))
        test_loss = criterion(outputs_test, Y_test)
        test_losses.append(test_loss.item())
      
      # train loss
      if epoch%(nb_epochs//min(20, nb_epochs//2)) == 0:
        print(f'[{epoch + 1}, {i + 1:5d}] Train loss: {running_loss/nb_loss:.7f}; Test loss: {test_loss:.7f}')
      losses.append(running_loss/nb_loss)
      running_loss = 0.0

    if epoch%(nb_epochs//min(10, nb_epochs//2)) == 0 and display:
      # plot results
      fig = plt.figure(figsize=(20,10), constrained_layout=True)
      axs = fig.subplots(3, 4, sharey=True)
      for i in range(12):
        idx = np.random.randint(len(inputs))
        # plot prediction
        axs[i%3,i%4].plot(labels[idx].detach().numpy(), color='tab:orange', label='ground truth')
        axs[i%3,i%4].plot(output[idx].detach().numpy(), color='tab:blue', label='predicted')

      plt.legend()
      fig.suptitle('Validation sample at epoch {}'.format(epoch+1))
      plt.show(block = False)
    
    epoch += 1

  return net, losses, test_losses, epoch