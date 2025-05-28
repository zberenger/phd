### IMPORTS ###
import numpy as np
random_seed=0
np.random.seed(random_seed)
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.rcParams.update({'font.size': 30})

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)

### VARIABLES ###
Nz = 512 # number of discrete heights
z = np.linspace(-10, 30, Nz) # heights

### FUNCTIONS ###
def create_p_sample2(x):
  # sampling from Gaussians
  ground_canopy_reg = np.random.uniform(0, 1) # ratio ground vs canopy
  mean0 = np.random.uniform(-5,5)
  std0 = np.random.uniform(0.1,2)
  ground_peak = scipy.stats.norm.pdf(x, mean0, std0)
  pdf = ground_canopy_reg*ground_peak/max(ground_peak) # ground reflection

  std1 = np.random.uniform(1,4)
  mean1 = np.random.uniform(max(mean0+std0+2*std1, -2),20)
  crown_peak = scipy.stats.norm.pdf(x, mean1, std1)
  pdf += (1-ground_canopy_reg)*crown_peak/max(crown_peak)

  param = [mean0, std0, mean1, std1, ground_canopy_reg]

  return pdf, param

def create_p_compressed_from_param(x, param, compression):
  # sampling from Gaussians
  ground_canopy_reg = param[-1] # ratio ground vs canopy
  mean0 = param[0]
  std0 = param[1]/compression
  mean1 = param[2]
  std1 = param[3]/compression

  ground_peak = scipy.stats.norm.pdf(x, mean0, std0)
  crown_peak = scipy.stats.norm.pdf(x, mean1, std1)
  pdf = ground_canopy_reg*ground_peak/max(ground_peak) + (1-ground_canopy_reg)*crown_peak/max(crown_peak)
  compressed_param = [mean0, std0, mean1, std1, ground_canopy_reg]

  return pdf, compressed_param


### CREATE DATA ###
Nsamples = 10000
original_psamples = [create_p_sample2(z) for i in range(Nsamples)]
original_profiles, original_params = zip(*original_psamples)
original_profiles, original_params = np.asarray(original_profiles), np.asarray(original_params)
np.random.seed(random_seed)
compression_values = np.random.uniform(1.0, 5.0, Nsamples)
compressed_psamples = [create_p_compressed_from_param(z, original_params[i], compression_values[i]) for i in range(Nsamples)]
compressed_profiles, compressed_params = zip(*compressed_psamples)
compressed_profiles, compressed_params = np.asarray(compressed_profiles), np.asarray(compressed_params)

### NETWORK VARIABLES ###
test_size = 0.25
batch_size = 64
input = np.asarray([original_profiles[k]/np.sum(original_profiles[k]) for k in range(Nsamples)])
indices = np.arange(input.shape[0])
reference = np.asarray([compressed_profiles[k]/np.sum(compressed_profiles[k]) for k in range(Nsamples)])
X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(np.concatenate((input, compression_values.reshape(-1, 1)), axis=1), reference, indices, test_size=test_size, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# pytorch
X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(np.asarray(Y_train).astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(np.asarray(Y_test).astype(np.float32))

# create dataset
class Data(Dataset):
    def __init__(self, X_t, Y_t, indices_t):
        self.X = X_t.reshape((X_t.shape[0], -1))
        self.Y = Y_t
        self.indices = indices_t
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.indices[index]

    def __len__(self):
        return self.len

data = Data(X_train, Y_train, indices_train)
trainloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=Data(X_test, Y_test, indices_test), batch_size=batch_size, shuffle=True)
print(f"Shape of data: {data.X.shape}")


### ARCHITECTURE ### 
class Net(nn.Module):
    def __init__(self, input, H1, H2, H3, H4, output):
        super(Net,self).__init__()
        # encoder
        self.linear1 = nn.Linear(input,H1,bias=False)
        self.linear2 = nn.Linear(H1,H2,bias=False)
        self.linear25 = nn.Linear(H2,H4,bias=False)
        self.linear5 = nn.Linear(H4,H2,bias=False)
        self.linear3 = nn.Linear(H2,H3,bias=False)
        self.linear4 = nn.Linear(H3,output,bias=False)


    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear25(x))
        x = F.leaky_relu(self.linear5(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.softmax(self.linear4(x), dim=-1)
        return x

# parameters
input_dim = X_train[0].reshape(-1).shape[0]    # number of variables + compression value
hidden_dim1 = 200 # hidden layers
hidden_dim2 = 100 # hidden layers
hidden_dim4 = 50 # hidden layers
hidden_dim3 = 200 # hidden layers
output_dim = Y_train[0].reshape(-1).shape[0]    # "number of classes"

net = Net(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim).to(device)
print(net.parameters)

# Hyperparameters
learning_rate = 0.001
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


### TRAINING ###
torch.manual_seed(0)
nb_epochs = 200
losses = []
test_losses = []
epoch = 0
while epoch < nb_epochs:#
  # print(epoch)
  running_loss = 0.0
  nb_loss = 0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels, indices = data

    # zero the gradient buffers
    optimizer.zero_grad()

    # optimisation
    output = net(inputs)
    loss = criterion(output, labels) #torch.zeros(labels.shape))
    loss.backward()
    optimizer.step()    # Does the update

    # statistics
    running_loss += loss.item()
    nb_loss += 1

  if epoch%(nb_epochs//100) == 0:
    # test loss
    with torch.no_grad():
      # calculate outputs by running matrices through the network
      outputs_test = net(X_test.reshape((X_test.shape[0], -1)))
      test_loss = criterion(outputs_test, Y_test)
      test_losses.append(test_loss.item())

    # train loss
    if epoch%(nb_epochs//50) == 0:
      print(f'[{epoch + 1}, {i + 1:5d}] Train loss: {running_loss/nb_loss:.7f}; Test loss: {test_loss:.7f}')
    losses.append(running_loss/nb_loss)
    running_loss = 0.0

  if epoch%(nb_epochs//10) == 0:
    # plot results
    fig, axs = plt.subplots(3,4,figsize=(15,10),sharey=True)
    for i in range(12):
      idx = np.random.randint(len(inputs))
      # plot prediction
      axs[i%3,i%4].plot(inputs[idx][:-1].detach().numpy(), color='k', label='input')
      axs[i%3,i%4].plot(labels[idx].detach().numpy(), color='tab:orange', label='comp '+str(inputs[idx][-1].cpu().numpy()))
      axs[i%3,i%4].plot(output[idx].detach().numpy(), color='tab:blue', label='predicted')

    fig.tight_layout()
    plt.legend()
    plt.show()

  epoch += 1
  # scheduler.step()

print('Finished Training at epoch number {}'.format(epoch))


### INFERENCE ###
fig, axs = plt.subplots(2, 4, figsize=(15,8), sharey=True)
for i in range(8):
  idx = np.random.randint(len(X_test))
  with torch.no_grad():
    # calculate predicted profile
    predicted = net(X_test[idx].reshape(-1))

  # plot prediction
  axs[i//4,i%4].plot(z,Y_test[idx].detach().numpy()/max(Y_test[idx].detach().numpy()), '-r', label='compressed')
  axs[i//4,i%4].plot(z,predicted/max(predicted), label='predicted')
  axs[i//4,i%4].plot(z,X_test[idx][:-1].detach().numpy()/max(X_test[idx][:-1].detach().numpy()), '-g', label='input')

fig.tight_layout()
plt.legend()
plt.show()



### SAVE MODEL ###
torch.save({
    'epoch': epoch,
    'net' : net,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': losses,
    'val_loss': test_losses,
}, './comp_6layers.pth')

np.savetxt('X_test.txt', X_test.cpu().detach().numpy())