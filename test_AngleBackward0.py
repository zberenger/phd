import argparse
import json
import itertools
#import gc

### IMPORTS ###
import numpy as np
import math
random_seed=0
import random
random.seed(random_seed)
import scipy.io
import scipy.signal
import scipy.spatial
import scipy.stats
import scipy.interpolate
import scipy.linalg
import matplotlib.pyplot as plt
from time import time
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.rcParams.update({'font.size': 30})

from interpolation.cubic_interpolator_irr import interpol_cubic_irr

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(0)

torch.autograd.set_detect_anomaly(True)

Nz = 512
Wrg = 8
Waz = 17

# Neural network architecture
class Net(nn.Module):
    def __init__(self, input, H1, H2, H3, H4, H5, H6, H7, output, input_conv, Wrg, Waz):
        super(Net,self).__init__()
        # encoder
        self.linear1 = nn.Linear(input,H1,bias=False)
        # self.linear2 = nn.Linear(H1,H2,bias=False)
        # self.linear3 = nn.Linear(H2,H3,bias=False)
        self.linear3 = nn.Linear(H1,H3,bias=False)
        self.linear4 = nn.Linear(H3,H4,bias=False)

        # decoder
        self.linear5 = nn.Linear(H4,H5,bias=False)
        # self.linear6 = nn.Linear(H5,H6,bias=False)
        # self.linear7 = nn.Linear(H6,H7,bias=False)
        self.linear7 = nn.Linear(H5,H7,bias=False)
        self.linear8 = nn.Linear(H7,output,bias=False)

        self.conv2d = nn.Conv2d(input, input, kernel_size=(Wrg, Waz), bias=False)

    def encoder(self, x):
        x = F.leaky_relu(self.linear1(x))
        # x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        return x

    def decoder(self, x):
        x = F.leaky_relu(self.linear5(x))
        # x = F.leaky_relu(self.linear6(x))
        x = F.leaky_relu(self.linear7(x))
        x = F.softmax(self.linear8(x), dim=-1)
        return x

    def forward(self, x):
        c = F.leaky_relu(self.conv2d(x))
        z = self.encoder(torch.squeeze(c))
        x = self.decoder(z)
        return x, z, c

# parameters
input_conv = (Nz+1, Wrg, Waz)  # size of the window
input_dim = Nz+1    # number of variables
hidden_dim1 = 100 # hidden layers
hidden_dim2 = 80 # hidden layers
hidden_dim3 = 50 # hidden layers
hidden_dim4 = 5 # hidden layers - size of the latent space
hidden_dim5 = 50 # hidden layers
hidden_dim6 = 80 # hidden layers
hidden_dim7 = 100 # hidden layers
output_dim = Nz    # "number of classes"

net = Net(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, hidden_dim7, output_dim, input_conv, Wrg, Waz).to(device)
print(net.parameters)

torch.manual_seed(0)
stime = time()
learning_rate = 0.001
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
stretch = T.TimeStretch(n_freq=1).to(device)
pos_0 = 128
nb_epochs = 20

epoch = 0
while epoch < nb_epochs:
    # zero the gradient buffers
    optimizer.zero_grad()

    # get the output
    inputs = torch.randn(10, Nz+1, Wrg, Waz, dtype=torch.float, device=device)
    output, _, _ = net(inputs)
    x2_tmp = torch.zeros(len(output), 512, dtype=torch.cfloat, device=device)
    scale = np.random.uniform(0.1, 1.3, len(output))
    for k in range(len(output)):
        if scale[k]<1.0:
            tmp = torch.ones(output[k].shape, device=device)
            tmp[0] = 1e-38
            tmp[1] = 1e-38
            print(tmp*output[k])
            x2_tmp[k][pos_0-int(pos_0*(int(np.ceil(Nz*scale[k])))/Nz):pos_0+(int(np.ceil(Nz*scale[k])))-int(pos_0*(int(np.ceil(Nz*scale[k])))/Nz)] = stretch((tmp*output[k]+ 1e-15).type(torch.cfloat), 1./scale[k])
        else:
            x2_tmp[k] = stretch(output[k].type(torch.cfloat), 1/scale[k])[int(pos_0*int(np.ceil(Nz*scale[k]))/Nz)-pos_0:int(pos_0*int(np.ceil(Nz*scale[k]))/Nz)+Nz-pos_0]
    x2 = torch.stack([x2_tmp[k]/torch.sum(x2_tmp[k]+1e-5) for k in range(len(output))])
    print(torch.isnan(x2).any())
    x3 = torch.zeros(x2.shape, dtype=float, device = device)
    print(torch.isnan(x3).any())
    loss = torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3+1e-2) - 10*torch.log10(x2+1e-2), ord=2, dim=1)))

    loss.backward()
    optimizer.step()

    print(f'[{epoch + 1}] Train loss: {loss:.7f}; Time: {time()-stime:.7f}')
    epoch += 1
