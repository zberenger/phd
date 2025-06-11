import neptune
from neptune.utils import stringify_unsupported
import argparse
import json
import itertools
#import gc

### IMPORTS ###
import numpy as np
random_seed=0
import random
random.seed(random_seed)
import scipy.io
import scipy.signal
import scipy.stats
import scipy.interpolate
import scipy.linalg
import matplotlib.pyplot as plt
from time import time
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.rcParams.update({'font.size': 30})

from interpolation.cubic_interpolator_irr import interpol_cubic_irr
from src.model import Net_comp

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(0)


parser = argparse.ArgumentParser(description="Process algorithm")
parser.add_argument("--exp-index", type=str, help="id of experiment", default=0)

args = parser.parse_args()
exp_num = int(args.exp_index)
print(f"Exp number: {exp_num}")

f = open("hyperparameters.json")
exp_hyperparams = json.load(f)

def make_all_possibilities(l):
    shapes = np.array([x.shape[0] for x in l])
    total_number = np.prod(shapes)

    result = np.zeros((total_number, len(l)))
    for i, x in enumerate(itertools.product(*l)):
        result[i] = np.array(x)

    return result


list_possibilities = []
keys = list(exp_hyperparams.keys())
for key in keys:
    list_possibilities.append(np.arange(len(exp_hyperparams[key])))

all_possibilities = make_all_possibilities(list_possibilities)
experiment = all_possibilities[exp_num]
print(f"Experiment: {experiment}")


run = neptune.init_run(
    project="zoeb/dilation",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNzQ4OWYzOC1jYzIzLTQ1NjEtOTliNC1jZmRmYTlmZjE3M2QifQ==",
    mode="offline",
)  # your credentials



### PARAMETERS
Nz = 512 # number of discrete heights
z = np.linspace(-10, 30, Nz) # heights
epsilon = 1e-1 # thermal noise

# simulated data
Nlook = 60 # number of looks in the simulated data
Nsamples = 10000 # number of simulated samples
Nim = 6 # number of simulated images in the stack
kz = np.linspace(0.2, (Nim-1)*0.2 + 0.2, num=Nim)/2
A = np.exp(-1j*kz.reshape(-1,1)*z.reshape(-1,1).transpose())
A_simu = None
az_out_sel = 115

yue = scipy.io.loadmat('./biosar_profile_az116.mat', squeeze_me=True)#, simplify_cells=True)
print(f"Ref profile data keys: {yue.keys()}")

Tomo_SP_CP = yue['Tomo_SP_CP']
az_out_sel = yue['az_out_sel']-1 # matlab to python conversion
kz_sel = yue['kz_sel']
rg_out_camp = yue['rg_out_camp']-1
spec_wv = yue['spec_wv']

mat = scipy.io.loadmat('./Biosar2_L_band_demo_data.mat', squeeze_me=True)#, simplify_cells=True) ## for newer scipy versions
print(f"BioSAR2 data keys: {mat.keys()}")

kz = mat['kz']
k = mat['k']
I = k[:,:,:,0]
rg_ax = np.linspace(0, I.shape[0], I.shape[0], endpoint=False)
az_ax = np.linspace(0, I.shape[1], I.shape[1], endpoint=False)


Nr, Na, N = I.shape
I_def = I #.item(pol_def)

interp_goal = exp_hyperparams["interp_goal"][int(experiment[keys.index("interp_goal")])]
def interpolate_cubic(kz, I, interp_goal):
    Nr, Na, N = kz.shape
    kz_interp = np.zeros((Nr, Na, interp_goal), dtype =kz.dtype)
    I_interp = np.zeros((Nr, Na, interp_goal), dtype=I.dtype)
    for i in range(Nr):
        if i%100==0:
                print(i)
        for j in range(Na):
            kz_interp[i, j, :] = np.linspace(np.min(kz[i, j, :]), np.max(kz[i, j, :]), interp_goal)
            I_interp[i, j, :] = interpol_cubic_irr(kz[i, j, :], I[i, j, :], kz_interp[i, j, :])
    
    return kz_interp, I_interp

if interp_goal != 'None':
    kz_interp, I_interp = interpolate_cubic(kz, I_def, interp_goal)
    Nim = int(interp_goal)
else:
    kz_interp, I_interp = kz, I_def
    Nim = 6


def conv2(x, y, mode='same'):
    return np.rot90(scipy.signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def generate_covariance_matrix(F, x_ax, y_ax, Wx, Wy):
    Ny, Nx, N = F.shape
    # pixel sampling for meter spaced ax
    dx = x_ax[1] - x_ax[0]
    dy = y_ax[1] - y_ax[0]

    Lx = np.round(Wx/(2*dx))
    x = np.arange(x_ax.shape[0]) # No subsampling here
    Ly = np.round(Wy/(2*dy))
    y = np.arange(y_ax.shape[0])

    mean_filter_mask = np.ones((int(2*Ly+1), int(2*Lx+1)))/((2*Lx+1)*(2*Ly+1)) # 2*+1 for odd box

    Cov = np.ones((Ny, Nx, N, N), dtype=complex)
    Corr = np.ones((Ny, Nx, N, N), dtype=complex)

    for n in range(N):
        In = F[:,:,n]
        Cnn = scipy.signal.convolve2d(In * np.conjugate(In), mean_filter_mask, mode='same')

        for m in range(n, N):
            Im = F[:,:,m]
            Cmm = scipy.signal.convolve2d(Im * np.conjugate(Im), mean_filter_mask, mode='same')
            Cnm = scipy.signal.convolve2d(In * np.conjugate(Im), mean_filter_mask, mode='same')
            
            # coherence
            coe = Cnm / np.sqrt(Cnn*Cmm + 10e-15)
            Cov[:, :, n, m] = Cnm
            Cov[:, :, m, n] = np.conj(Cnm)
            Corr[:, :, n, m] = coe
            Corr[:, :, m, n] = np.conj(coe)
                
    return Cov, Corr

def display_COV(COV):
    ni, nj, n, _ = COV.shape
    COV_display = np.zeros((ni*n, nj*n), dtype='c16')
    for line in range(0,n):
        for col in range(0, n):
            COV_display[line*ni:line*ni+ni, col*nj:col*nj+nj] = COV[:,:,line,col]
    plt.figure(figsize=(15,15))
    plt.imshow(np.abs(COV_display))
    
    
Wrg = exp_hyperparams["Wrg"][int(experiment[keys.index("Wrg")])]
Waz = exp_hyperparams["Waz"][int(experiment[keys.index("Waz")])]
t = time()
Cov_def, _ = generate_covariance_matrix(I_interp, az_ax, rg_ax, Waz, Wrg)
print(f"Covariance computed in: {time() - t}")
  

# LiDAR filtering
dem = scipy.io.loadmat('./L_band_SW_DEM_CHM_sub.mat', squeeze_me=True)#, simplify_cells=True) ## for newer scipy versions
CHM = dem['CHM']
DTM = dem['DTM']
az_ax_dem = dem['az_ax']
rg_ax_dem = dem['rg_ax']

Lrg = np.round(Wrg/2)
Laz = np.round(Waz/2)
mean_filter_mask = np.ones((int(2*Laz+1), int(2*Lrg+1)))/((2*Lrg+1)*(2*Laz+1))

SR_CHM_filtered = scipy.signal.convolve2d(CHM, mean_filter_mask, mode='same')
SR_DTM_filtered = scipy.signal.convolve2d(DTM, mean_filter_mask, mode='same')


def bf(COV, kz, az_selected):
    a_sub = np.arange(Na)
    r_sub = np.arange(Nr)

    az_ax_sub = az_ax[a_sub]
    a = np.abs(az_ax_sub-az_ax[az_selected])
    t = np.unravel_index(np.argmin(a, axis=None), a.shape)
    a0 = t[0]
    kz_sub = kz[r_sub, :, :][:, a_sub, :]

    Sp_estimator = np.zeros((z.shape[0], r_sub.shape[0]))

    for r in range(r_sub.shape[0]):
        kz_r = kz_sub[r, a0,:].reshape(1, -1)
        A = np.exp(-1j*z.reshape(-1,1).dot(kz_r))

        cov = COV[r,a0,:,:]
        Sp_estimator[:,r] = np.real(np.diag(A @ cov @ np.conjugate(A).T))

    return Sp_estimator

bf_def = bf(Cov_def, kz_interp, az_out_sel)



# only 2 gaussians
def create_p_sample2(x):
  # sampling from Gaussians
  ground_canopy_reg = np.random.uniform(0, 1) # ratio ground vs canopy
  mean0 = np.random.uniform(-5,5)
  std0 = np.random.uniform(0.1,2)
  ground_peak = scipy.stats.norm.pdf(x, mean0, std0)
  gauss_list = ground_canopy_reg*ground_peak/max(ground_peak) # ground reflection

  std1 = np.random.uniform(1,4)
  mean1 = np.random.uniform(max(mean0+std0+2*std1, -2),20)
  crown_peak = scipy.stats.norm.pdf(x, mean1, std1)
  gauss_list += (1-ground_canopy_reg)*crown_peak/max(crown_peak)

  return gauss_list, [mean0, std0, mean1, std1, ground_canopy_reg]

### simulate reflectivity vectors
p = []
parameters = []
np.random.seed(random_seed)
t0 = time()
for i in range(Nsamples):
  if i%1000 == 0:
    print(i)
  pdf, param = create_p_sample2(z)
  p.append(pdf)
  parameters.append(param)
print(f"Time taken for profile simulation: {time() - t0}")

# normalisation - to obtain APA normalised
p_norm = np.transpose(np.asarray(p)/np.sum(p, axis=1).reshape(-1, 1))


# compute real A matrices randomly for each sample
def geometry_simulation(z, kz, Nsamples, random_seed):
  Nr, Na, _ = kz.shape
  A = []
  np.random.seed(random_seed)
  r_a_coord = np.transpose(np.random.randint([[Nr], [Na]], size=(2, Nsamples)))
  for i in range(Nsamples):
    A.append(np.transpose(np.exp(-1j*z.reshape(-1,1).dot(kz[r_a_coord[i][0], r_a_coord[i][1], :].reshape(1,-1)))))

  return np.asarray(A), r_a_coord

A_simu, r_a_coord = geometry_simulation(z, kz_interp, Nsamples, random_seed)

def create_z_samples(p, A, Nlook, epsilon):
  # create independent samples from theoretic p profiles
  zsim = []
  Nim = np.asarray(A).shape[1]

  for i in range(p.shape[1]):
    C = A[i] @ np.diag(p[:, i]) @ np.conj(np.transpose(A[i])) + epsilon * np.eye(Nim)
    L = np.linalg.cholesky(C)
    noise = np.random.randn(Nim, Nlook)/np.sqrt(2) + 1j*np.random.randn(Nim, Nlook)/np.sqrt(2)
    zsim.append(L @ noise)

  return zsim

# create independent samples from theoretic p profiles
t0 = time()
zsim = create_z_samples(p_norm, A_simu, Nlook, epsilon)
print(f"Time taken to simulate measures: {time() - t0}")

# compute correlation
def simulated_correlation(zsim, W):
  corr = []
  for ns in range(len(zsim)):
    cov = zsim[ns] @ np.conj(np.transpose(zsim[ns])) / W
    D = np.diag(1./np.sqrt(np.diag(cov)))
    corr.append(D @ cov @ D)
  return corr

corr = simulated_correlation(zsim, Nlook)

# beamforming
pbf = [np.real(np.sum(np.conjugate(A_simu[i]) * (corr[i] @ A_simu[i]), axis=0))/Nim**2 for i in range(len(zsim))]




### FUNCTIONS
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

# Process data
def preprocessing_data_for_nn(input, ground_truth=np.transpose(np.asarray(p_norm)), test_size=0.25, batch_size=32):
  X_train, X_test, Y_train, Y_test = train_test_split(input, ground_truth, test_size=test_size, random_state=42)
  print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

  # pytorch variables
  X_train = torch.from_numpy(X_train.astype(np.float32))
  Y_train = torch.from_numpy(np.asarray(Y_train).astype(np.float32))
  X_test = torch.from_numpy(X_test.astype(np.float32))
  Y_test = torch.from_numpy(np.asarray(Y_test).astype(np.float32))

  # create dataset
  data = Data(X_train, Y_train)
  trainloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
  testloader = DataLoader(dataset=Data(X_test, Y_test), batch_size=batch_size, shuffle=True)

  return trainloader, testloader, X_train, Y_train, X_test, Y_test

# Neural network architecture
class Net(nn.Module):
    def __init__(self, input, H1, H2, H3, H4, H5, H6, H7, output):
        super(Net,self).__init__()
        # encoder
        self.linear1 = nn.Linear(input,H1,bias=False)
        self.linear2 = nn.Linear(H1,H2,bias=False)
        self.linear3 = nn.Linear(H2,H3,bias=False)
        self.linear4 = nn.Linear(H3,H4,bias=False)

        # decoder
        self.linear5 = nn.Linear(H4,H5,bias=False)
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
        return x

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

# Create network with a certain latent space size
def create_autoencoder(X_train, Y_train, latent_space_size=5):
  # parameters
  input_dim = X_train[0].reshape(-1).shape[0]    # number of variables
  hidden_dim1 = 100 # hidden layers
  hidden_dim2 = 80 # hidden layers
  hidden_dim3 = 50 # hidden layers
  hidden_dim4 = latent_space_size # hidden layers - size of the latent space
  hidden_dim5 = 50 # hidden layers
  hidden_dim6 = 80 # hidden layers
  hidden_dim7 = 100 # hidden layers
  output_dim = Y_train[0].reshape(-1).shape[0]    # "number of classes"

  net = Net(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, hidden_dim7, output_dim)

  # Hyperparameters
  learning_rate = 0.001
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  return net, learning_rate, criterion, optimizer

### Plot loss graphs
def plot_losses(losses, test_losses, nb_epochs=100):
  plt.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses), label='Train loss')
  plt.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses), label='Test loss')
  plt.legend()
  plt.show()

  plt.plot(np.linspace(nb_epochs//4, nb_epochs, len(losses)-len(losses)//4), np.array(losses[len(losses)//4:]), label='Train loss')
  plt.plot(np.linspace(nb_epochs//4, nb_epochs, len(losses)-len(losses)//4), np.array(test_losses[len(losses)//4:]), label='Test loss')
  plt.legend()
  plt.show()

  plt.plot(np.linspace(nb_epochs//2, nb_epochs, len(losses)-len(losses)//2), np.array(losses[len(losses)//2:]), label='Train loss')
  plt.plot(np.linspace(nb_epochs//2, nb_epochs, len(losses)-len(losses)//2), np.array(test_losses[len(losses)//2:]), label='Test loss')
  plt.legend()
  plt.show()


### predict random test profile and visualize their latent codes
def predict_random(X_test, nb_rows=2, nb_columns=4):
  total_rows = nb_rows*2
  fig, axs = plt.subplots(total_rows, nb_columns, figsize=(15,12), sharey=True)
  for i in range(nb_rows * nb_columns):
    idx = np.random.randint(len(X_test))
    with torch.no_grad():
      # calculate predicted profile
      predicted, latent_codes = net(X_test[idx].reshape(-1))

    # plot prediction
    axs[2*(i//total_rows),i%nb_columns].plot(z,Y_test[idx].detach().numpy(), '-r', label='ground truth')
    axs[2*(i//total_rows),i%nb_columns].plot(z,predicted, label='predicted')
    axs[2*(i//total_rows)+1,i%nb_columns].plot(latent_codes, 'x', markersize=10, label='latent codes')

  fig.tight_layout()
  handles, labels = [(a + b) for a, b in zip(axs[2*(i//total_rows),i%nb_columns].get_legend_handles_labels(), axs[2*(i//total_rows)+1,i%nb_columns].get_legend_handles_labels())]
  fig.legend(handles, labels, loc='upper right')
  plt.show()



# variables
test_size = 0.25
batch_size = 64

input = np.asarray([pbf[k]/np.sum(pbf[k]) for k in range(Nsamples)])
indices = np.arange(input.shape[0])
reference = np.asarray([np.transpose(p_norm)[k]/np.sum(np.transpose(p_norm)[k]) for k in range(Nsamples)])
X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(input, reference, indices, test_size=test_size, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# pytorch variables
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


# Neural network architecture
class Net(nn.Module):
    def __init__(self, input, H1, H2, H3, H4, H5, H6, H7, output):
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
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

# parameters
input_dim = X_train[0].reshape(-1).shape[0]    # number of variables
hidden_dim1 = 100 # hidden layers
hidden_dim2 = 80 # hidden layers
hidden_dim3 = 50 # hidden layers
hidden_dim4 = 5 # hidden layers - size of the latent space
hidden_dim5 = 50 # hidden layers
hidden_dim6 = 80 # hidden layers
hidden_dim7 = 100 # hidden layers
output_dim = Y_train[0].reshape(-1).shape[0]    # "number of classes"

net = Net(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, hidden_dim7, output_dim).to(device)
print(net.parameters)

# Hyperparameters
learning_rate = exp_hyperparams["learning_rate"][int(experiment[keys.index("learning_rate")])]
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Dilations
net_comp = Net_comp(input_dim+1, input_dim)
checkpoint = torch.load("comp_4layers_TV2_001_300ep.pth")
net_comp.load_state_dict(checkpoint['model_state_dict'])
net_comp.eval()

# Training  - FINAL NORMALIZED translations Nz/16 normal
torch.manual_seed(0)
stime = time()
nb_epochs = exp_hyperparams["nb_epochs"][int(experiment[keys.index("nb_epochs")])]
# nb_epochs = 100
alpha = exp_hyperparams["alpha"][int(experiment[keys.index("alpha")])]
epsilon = 1e-1 # thermal noise
stretch = T.TimeStretch(n_freq=1).to(device)
pos_0 = 128

losses = []
losses1 = []
losses2 = []
test_losses = []
test_losses1 = []
test_losses2 = []
epoch = 0
while epoch < nb_epochs:
  running_loss = 0.0
  running_loss1 = 0.0
  running_loss2 = 0.0
  nb_loss = 0

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels, indices_inputs = data
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the gradient buffers
    optimizer.zero_grad()

    # get the output
    output, _ = net(inputs)

    # compute the transformations
    A_tensor = torch.tensor(A_simu[indices_inputs], dtype=torch.cfloat, device=device)

    # net dilation 
    # scale = 1.0/np.random.uniform(0.1, 1.3, len(output)) # for net
    # X_comp = torch.cat((output, torch.from_numpy(scale.reshape(-1,1).astype(np.float32))), dim=1)
    # X_dilated = net_comp(X_comp)
    scale = np.random.uniform(0.1, 1.3, len(output)) # for stretch
    trans_bool = np.random.randint(0, 3, len(output))
    x2_tmp = torch.zeros(len(output), Nz, dtype=torch.cfloat, device=device)
    for k in range(len(output)):
      if trans_bool[k]:
        # dilation with stretch
        if scale[k]<1.0:
          x2_tmp[k][pos_0-int(pos_0*(int(np.ceil(Nz*scale[k])))/Nz):pos_0+(int(np.ceil(Nz*scale[k])))-int(pos_0*(int(np.ceil(Nz*scale[k])))/Nz)] = stretch((output[k]+1e-15).type(torch.cfloat), 1./scale[k])
        else:
          x2_tmp[k] = stretch((output[k]+1e-15).type(torch.cfloat), 1/scale[k])[int(pos_0*int(np.ceil(Nz*scale[k]))/Nz)-pos_0:int(pos_0*int(np.ceil(Nz*scale[k]))/Nz)+Nz-pos_0]
        # # dilation with net
        # x2_tmp[k] = X_dilated[k]

        if trans_bool[k]==2:
          x2_tmp[k] = torch.roll(x2_tmp[k], int(np.random.normal(0, 0.25*Nz/16)))
      else:
        # translation
        # tmp = np.random.randint(int(Nz/16))-10
        # x2[k] = output[k] @ (torch.diag(torch.ones(Nz-abs(tmp), device=device), diagonal=tmp) + torch.diag(torch.ones(abs(tmp), device=device), diagonal=np.sign(tmp)*(abs(tmp)-Nz))) # reversible
        scale[k] = int(np.random.normal(0, 0.25*Nz/16))
        x2_tmp[k] = torch.roll(output[k], int(scale[k]))

    x2 = x2_tmp/(torch.sum(x2_tmp + 1e-5, axis=1))[..., None]
    # x2 = torch.stack([x2_tmp[k]/torch.sum(x2_tmp[k]+1e-5) for k in range(len(output))])
    # zsim_tmp = torch.stack([A_tensor[k] @ torch.diag(torch.sqrt(x2[k]+1e-15)).cfloat() @
    #                        (torch.randn(Nz, Nlook, device=device) +
    #                         1j*torch.randn(Nz, Nlook, device=device))/np.sqrt(2) +
    #                         np.sqrt(epsilon)*(torch.randn(Nim, Nlook, device=device) +
    #                                           1j*torch.randn(Nim, Nlook, device=device))/np.sqrt(2)
    #                         for k in range(len(output))])
    zsim_tmp = torch.matmul(torch.matmul(A_tensor, torch.diag_embed(torch.sqrt(x2 + 1e-5))),
                           (torch.randn(A_tensor.shape[0], Nz, Nsimu, device=device) +
                            1j*torch.randn(A_tensor.shape[0], Nz, Nsimu, device=device)) /
                            np.sqrt(2)) + np.sqrt(epsilon)*(torch.randn(A_tensor.shape[0], Nim, Nsimu, device=device) +
                                              1j*torch.randn(A_tensor.shape[0], Nim, Nsimu, device=device))/np.sqrt(2)
    cov_tmp = torch.stack([zsim_tmp[k] @ torch.transpose(torch.conj(zsim_tmp[k]), 0, 1) / Nlook ## TODO
                        for k in range(len(output))])
    diag_tmp = torch.stack([torch.diag(1./torch.sqrt(torch.diag(cov_tmp[k])))
                            for k in range(len(output))])
    corr_tmp = torch.stack([diag_tmp[k] @ cov_tmp[k] @ diag_tmp[k]
                            for k in range(len(output))])
    pbf_tmp = torch.stack([torch.real(torch.sum(torch.conj(A_tensor[k]) * (corr_tmp[k] @ A_tensor[k])/Nim**2, axis=0))/
                           torch.sum(torch.real(torch.sum(torch.conj(A_tensor[k]) * (corr_tmp[k] @ A_tensor[k])/Nim**2, axis=0))+1e-5)
                           for k in range(len(output))])

    # predict from transformed beamforming profile
    x3, _ = net(pbf_tmp)

    # optimisation - loss: data attachment term Adiag(p)A'-R + regularization
    R_tensor = torch.tensor(np.asarray(corr)[indices_inputs], dtype=torch.cfloat, device=device)
    # loss1 = (torch.sum(torch.stack([torch.linalg.norm(A_tensor[k] @
    #                                                   torch.diag(output[k]).cfloat() @
    #                                                   torch.transpose(torch.conj(A_tensor[k]), 0, 1) - R_tensor[k], ord='fro')
    #                                 for k in range(len(inputs))])))
    # losses on network outputs
    # cov_loss = [A_tensor[k] @ torch.diag(output[k]).cfloat() @ torch.transpose(torch.conj(A_tensor[k]), 0, 1) + epsilon * torch.eye(Nim, device=device)
    #                                     for k in range(len(inputs))]
    # corr_loss = [torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-15)) @ cov_loss[k] @ torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-15))
    #                                     for k in range(len(inputs))]
    # loss1 = (torch.sum(torch.stack([torch.square(torch.linalg.norm(corr_loss[k] - R_tensor[k], ord='fro'))
    #                                     for k in range(len(inputs))])))
    cov_loss = torch.matmul(torch.matmul(A_tensor, torch.diag_embed(output).cfloat()),
                            torch.transpose(torch.conj(A_tensor), 1, 2)) + epsilon * torch.eye(Nim, dtype=torch.cfloat, device=device).reshape((1, Nim, Nim)).repeat(output.shape[0], 1, 1)
    corr_loss = torch.matmul(torch.matmul(torch.diag_embed(1./torch.sqrt(torch.diagonal(cov_loss, dim1=-2, dim2=-1)+1e-5)), cov_loss),
                             torch.diag_embed(1./torch.sqrt(torch.diagonal(cov_loss, dim1=-2, dim2=-1)+1e-5)))
    loss1 = (torch.sum(torch.square(torch.linalg.norm(corr_loss - R_tensor, ord='fro', dim=(-1, -2)))))
    loss2 = alpha * torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3+1e-2) - 10*torch.log10(x2+1e-2), ord=2, dim=1)))
    loss = loss1 + loss2

    loss.backward()
    optimizer.step()    # Does the update
    # print(loss.item())
    # statistics
    running_loss += loss.item()
    running_loss1 += loss1.item()
    running_loss2 += loss2.item()
    nb_loss += 1
  
  print(f'[{epoch + 1}] Train loss: {running_loss/nb_loss:.7f}; Data attachment: {running_loss1/nb_loss:.7f}; Regularization: {running_loss2/nb_loss:.7f}; Time: {time()-stime:.7f}')
  
  if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
    # test loss
    with torch.no_grad():
      for j, (X_test_b, _, indices_test_b) in enumerate(testloader, 0):
        # calculate outputs by running matrices through the network ## LAAAAAA
        outputs_test, _ = net(X_test_b.reshape((X_test_b.shape[0], -1)).to(device))

        # compute the transformations
        A_tensor_test = torch.tensor(A_simu[indices_test_b], dtype=torch.cfloat, device=device)

        # transformation
        scale_test = 1.0/np.random.uniform(0.1, 1.5, len(outputs_test))
        X_comp_test = torch.cat((outputs_test, torch.from_numpy(scale_test.reshape(-1,1).astype(np.float32))), dim=1)
        X_dilated_test = net_comp(X_comp_test)
        # scale_test = np.random.uniform(0.1, 1.5, len(outputs_test))
        trans_bool_test = np.random.randint(0, 3, len(outputs_test))

        x2_test = torch.zeros(len(outputs_test), Nz, dtype=torch.cfloat, device=device)
      for k in range(len(outputs_test)):
        if trans_bool_test[k]:
          # # dilation with stretch
          # if scale_test[k]<1.0:
          #   x2_test[k][128-int(128*(int(np.ceil(Nz*scale_test[k])))/Nz):128+(int(np.ceil(Nz*scale_test[k])))-int(128*(int(np.ceil(Nz*scale_test[k])))/Nz)] = stretch(outputs_test[k], 1./scale_test[k])
          # else:
          #   x2_test[k] = stretch(outputs_test[k], 1./scale_test[k])[int(128*int(np.ceil(Nz*scale_test[k]))/Nz)-128:int(128*int(np.ceil(Nz*scale_test[k]))/Nz)+Nz-128]
          
          # dilation with net
          x2_test[k] = X_dilated_test[k]

          if trans_bool_test[k]==2:
            x2_test[k] = torch.roll(x2_test[k], int(np.random.normal(0, 0.25*Nz/16)))
          x2_test[k] = x2_test[k]/torch.sum(x2_test[k]+1e-5)
        else:
          # translation
        #   tmp = np.random.randint(int(Nz/16))-10
        #   x2_test[k] = outputs_test[k] @ torch.diag(torch.ones(Nz-abs(tmp), device=device), diagonal=tmp) + torch.diag(torch.ones(abs(tmp), device=device), diagonal=np.sign(tmp)*(abs(tmp)-Nz))) # reversible
          x2_test[k] = torch.roll(outputs_test[k], int(np.random.normal(0, 0.25*Nz/16)))
          x2_test[k] = x2_test[k]/torch.sum(x2_test[k]+1e-5)


      zsim_test = torch.stack([A_tensor_test[k] @ torch.diag(torch.sqrt(x2_test[k]+1e-15)).cfloat() @
                               (torch.randn(Nz, Nlook, device=device) +
                                1j*torch.randn(Nz, Nlook, device=device))/np.sqrt(2) +
                               np.sqrt(epsilon)*(torch.randn(Nim, Nlook, device=device) +
                                                 1j*torch.randn(Nim, Nlook, device=device))/np.sqrt(2)
                               for k in range(len(outputs_test))])
      cov_test = torch.stack([zsim_test[k] @ torch.transpose(torch.conj(zsim_test[k]), 0, 1) / Nlook
                              for k in range(len(outputs_test))])
      diag_test = torch.stack([torch.diag(1./torch.sqrt(torch.diag(cov_test[k])))
                              for k in range(len(outputs_test))])
      corr_test = torch.stack([diag_test[k] @ cov_test[k] @ diag_test[k]
                              for k in range(len(outputs_test))])
      pbf_test = torch.stack([torch.real(torch.sum(torch.conj(A_tensor_test[k]) * (corr_test[k] @ A_tensor_test[k])/Nim**2, axis=0))/
                              torch.sum(torch.real(torch.sum(torch.conj(A_tensor_test[k]) * (corr_test[k] @ A_tensor_test[k])/Nim**2, axis=0))+1e-5) for k in range(len(outputs_test))])

      # predict from translated beamforming profile
      x3_test, _ = net(pbf_test)

      R_tensor_test = torch.tensor(np.asarray(corr)[indices_test], dtype=torch.cfloat, device=device)
      # losses on network outputs
      cov_loss = [A_tensor_test[k] @ torch.diag(outputs_test[k]).cfloat() @ torch.transpose(torch.conj(A_tensor_test[k]), 0, 1) + epsilon * torch.eye(Nim)
                                          for k in range(len(X_test))]
      corr_loss = [torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-15)) @ cov_loss[k] @ torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-15))
                                          for k in range(len(X_test))]
      test_loss1 = (torch.sum(torch.stack([torch.square(torch.linalg.norm(corr_loss[k] - R_tensor_test[k], ord='fro'))
                                          for k in range(len(X_test))])))
      test_loss2 = alpha * torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3_test+1e-2) - 10*torch.log10(x2_test+1e-2), ord=2, dim=1)))
                  #  alpha * torch.linalg.norm(x3_test - x2_test, ord=2))
      test_loss = test_loss1 + test_loss2
      test_losses.append(test_loss.item())
      test_losses1.append(test_loss1.item())
      test_losses2.append(test_loss2.item())

    # train loss
    if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
      print(f'[{epoch + 1}, {i + 1:5d}] Train loss: {running_loss/nb_loss:.7f}; Data attachment: {running_loss1/nb_loss:.7f}; Regularization: {running_loss2/nb_loss:.7f}; Test loss: {test_loss.item():.7f}; Time: {time()-stime:.7f}')
    losses.append(running_loss/nb_loss)
    losses1.append(running_loss1/nb_loss)
    losses2.append(running_loss2/nb_loss)
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0

  if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
    # plot results
    fig, axs = plt.subplots(3,4,figsize=(15,10),sharey=True)
    for i in range(12):
      idx = np.random.randint(len(inputs))
      # plot prediction
      axs[i%3,i%4].plot(z, labels[idx].cpu().detach().numpy(), color='tab:orange', label='ground truth')
      axs[i%3,i%4].plot(z, output[idx].cpu().detach().numpy(), color='tab:blue', label='predicted')

    fig.tight_layout()
    plt.legend()
    # plt.show()

    run[f"training_val/{epoch + 1}"].upload(fig)
    plt.close()

  epoch += 1

total_time = time() - stime
print(f'Finished Training at epoch number {epoch} in {total_time} seconds')




# print loss statistics - with translation 2 and 0.1 1.3 dilation 200 epochs alpha 0.1 true losses
fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses), 'r', label='Train loss')
ax2.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses), 'b', label='Test loss')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
#plt.show()
    
run[f"final_losses/general"].upload(fig)
plt.close()

fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses), 'r', label='Train loss')
ax.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses1), '--r', label='Train loss 1')
ax.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses2), '-xr', label='Train loss 2')
ax2.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses), 'b', label='Test loss')
ax2.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses1), '--b', label='Test loss 1')
ax2.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses2), '-xb', label='Test loss 2')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
#plt.show()
    
run[f"final_losses/all"].upload(fig)
plt.close()

fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(nb_epochs//4, nb_epochs, len(losses)-len(losses)//4), np.array(losses[len(losses)//4:]), 'r', label='Train loss')
ax2.plot(np.linspace(nb_epochs//4, nb_epochs, len(losses)-len(losses)//4), np.array(test_losses[len(losses)//4:]), 'b', label='Test loss')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
#plt.show()
    
run[f"final_losses/three_quarters"].upload(fig)
plt.close()

fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(nb_epochs//2, nb_epochs, len(losses)-len(losses)//2), np.array(losses[len(losses)//2:]), 'r', label='Train loss')
ax2.plot(np.linspace(nb_epochs//2, nb_epochs, len(losses)-len(losses)//2), np.array(test_losses[len(losses)//2:]), 'b', label='Test loss')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
#plt.show()
    
run[f"final_losses/half"].upload(fig)
plt.close()


## 100 epochs
fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[0].cpu().detach().numpy(), label='bf')
plt.plot(z, pbf_tmp[0].cpu().detach().numpy(), label='bf(x2)')
plt.plot(z, output[0].cpu().detach().numpy(), label='output')
plt.plot(z, x2[0].cpu().detach().numpy(), label='x2')
plt.plot(z, x3[0].cpu().detach().numpy(), label='x3')
plt.plot(z, labels[0].cpu().detach().numpy(), label='refs', c='k')
plt.legend()
#plt.show()
    
run[f"results/simulated_0"].upload(fig)
plt.close()

fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[10].cpu().detach().numpy(), label='bf')
plt.plot(z, pbf_tmp[10].cpu().detach().numpy(), label='bf(x2)')
plt.plot(z, output[10].cpu().detach().numpy(), label='output')
plt.plot(z, x2[10].cpu().detach().numpy(), label='x2')
plt.plot(z, x3[10].cpu().detach().numpy(), label='x3')
plt.plot(z, labels[10].cpu().detach().numpy(), label='refs', c='k')
plt.legend()
#plt.show()
    
run[f"results/simulated_10"].upload(fig)
plt.close()


bf_def_predicted = bf_def.copy()
bf_def_torch = torch.from_numpy(bf_def.astype(np.float32))
for i in range(Nr):
  with torch.no_grad():
    # calculate predicted profile
    bf_def_predicted[:,i], _ = net((bf_def_torch[:,i]/torch.sum(bf_def_torch[:,i]+1e-5)).reshape(-1))


# final reconstructed slice -- log alpha 0.1 100 epochs dilation 0.1 1.3 translation 2 fully simplified corrected varying
fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def[:,rg_out_camp], extent=[0,bf_def[:,rg_out_camp].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(3)
plt.title('HH after beamforming')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def_predicted[:,rg_out_camp], extent=[0,bf_def[:,rg_out_camp].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(3)
plt.title('HH after NN')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_predicted"].upload(fig)
plt.close()

# final reconstructed slice (dB)
fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def[:,rg_out_camp]))
plt.imshow(tmp, extent=[0,bf_def[:,rg_out_camp].shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(3)
plt.title('HH after beamforming (dB)')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_log"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def_predicted[:,rg_out_camp])+10e-15)
plt.imshow(tmp, extent=[0,bf_def[:,rg_out_camp].shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(3)
plt.title('HH after NN (dB)')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_predicted_log"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10((np.abs(bf_def_predicted[:,rg_out_camp])+10e-15)/np.max(np.abs(bf_def_predicted[:,rg_out_camp])+10e-15))
im=plt.imshow(tmp, extent=[0,bf_def[:,rg_out_camp].shape[1],min(z),max(z)], origin='lower')
plt.plot(np.arange(len(rg_out_camp)), SR_DTM_filtered[rg_out_camp,az_out_sel]-SR_DTM_filtered[rg_out_camp,az_out_sel], 'w', linewidth=3.0)
plt.plot(np.arange(len(rg_out_camp)), SR_CHM_filtered[rg_out_camp, az_out_sel], 'w', linewidth=3.0)
plt.clim(-20, 0)
axes=plt.gca()
axes.set_aspect(4)
plt.locator_params(nbins=5)
plt.xlabel('range [bin]')
plt.ylabel('height z [m]')
plt.colorbar(im,fraction=0.046, pad=0.01)
#plt.show()
    
run[f"results/bf_def_predicted_log_dem"].upload(fig)
plt.close()

# # save model bf EI 100
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'train_loss': losses,
#     'train_loss_att': losses1,
#     'train_loss_reg': losses2,
#     'val_loss': test_losses,
#     'val_loss_att': test_losses1,
#     'val_loss_reg': test_losses2,
# }, './ei_bf_100.pth')

# np.savetxt('X_test.txt', X_test.cpu().detach().numpy())

params = {
    "az_out_sel": az_out_sel,
    "interp_goal": interp_goal,
    "test_size": test_size,
    "batch_size": batch_size,
    "alpha": alpha,
    "learning_rate": learning_rate,
    "nb_epochs": nb_epochs,
    "epsilon": epsilon,
}

run["parameters"] = params
run["parameters/rg_out_camp"] = stringify_unsupported(rg_out_camp)
run["parameters/net_parameters"] = stringify_unsupported(net.parameters)
run["parameters/az_ax"] = stringify_unsupported(az_ax)
run["parameters/rg_ax"] = stringify_unsupported(rg_ax)

results = {
    "total_time_training": total_time
}

run["results/arrays"] = results
run["results/losses"] = stringify_unsupported(losses)
run["results/losses_att"] = stringify_unsupported(losses1)
run["results/losses_reg"] = stringify_unsupported(losses2)
run["results/test_losses"] = stringify_unsupported(test_losses)
run["results/test_losses_att"] = stringify_unsupported(test_losses1)
run["results/test_losses_reg"] = stringify_unsupported(test_losses2)
run["results/bf_def_predicted"] = stringify_unsupported(bf_def_predicted)


run.stop()


