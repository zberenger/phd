import neptune
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
    project="zoeb/slc",
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

def gaussian_kernel(row, column, sigma=1.):
    x, y = np.meshgrid(np.linspace(-1, 1, column), np.linspace(-1, 1, row))
    dst = np.sqrt(x**2+y**2)
    # normal = 1./(2.0 * np.pi * sigma**2)
    kernel = np.exp(-0.5 * np.square(dst) / np.square(sigma)) # * normal
    return kernel / np.sum(kernel)

def generate_covariance_matrix(F, x_ax, y_ax, Wx, Wy, gaussian=False):
    Ny, Nx, N = F.shape
    # pixel sampling for meter spaced ax
    dx = x_ax[1] - x_ax[0]
    dy = y_ax[1] - y_ax[0]
    
    Lx = np.round(Wx/(2*dx))
    Ly = np.round(Wy/(2*dy))
    mean_filter_mask = np.ones((int(2*Ly+1), int(2*Lx+1)))/((2*Lx+1)*(2*Ly+1)) # 2*+1 for odd box 
    if gaussian==True:
        mean_filter_mask = gaussian_kernel(int(2*Ly+1), int(2*Lx+1), sigma=1.)

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
Cov_def = generate_covariance_matrix(I_interp, az_ax, rg_ax, Waz, Wrg)
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


##### REAL ########

np.random.seed(random_seed)
training_coord = np.load('training_coord_slc.npy')
print(f"Number of training coordinates: {len(training_coord)}")

bf_training = np.load('bf_training_slc.npy')
neighbors_training = np.load('neighbors_training_slc.npy')
A_training = np.conj(np.swapaxes(np.load('A_training_slc.npy'), 1, 2))
kz_neighbors = np.load('kz_neighbors_slc.npy')
R_training = np.load('R_training_slc.npy')
print(f"Shape of bf_training: {bf_training.shape}")
print(f"Shape of neighbors_training: {neighbors_training.shape}")
print(f"Shape of A_training: {A_training.shape}")
print(f"Shape of neighbors_kz: {kz_neighbors.shape}")
print(f"Shape of R_training: {R_training.shape}")

# variables
test_size = 0.25
batch_size = exp_hyperparams["batch_size"][int(experiment[keys.index("batch_size")])]

def process_input_similarity(bf_training, idx):
  input = bf_training/np.sum(bf_training + 1e-5, axis=2)[..., None]
  cosim = [[1 - np.abs(scipy.spatial.distance.cosine(bf_training[k][i], bf_training[k][idx])) for i in range(len(bf_training[k]))] for k in range(len(bf_training))]
  input = np.moveaxis(np.stack((input.reshape(input.shape[0], Wrg, Waz, input.shape[2]), np.broadcast_to(np.asarray(cosim).reshape(input.shape[0], Wrg, Waz, 1), (input.shape[0], Wrg, Waz, input.shape[2]))), axis=-1), -2, 1)

  return input

def compute_NLSAR_dissimilarity(cov, Wp, rg_ref=None, az_ref=None):
  Nr, Na, _, _ = cov.shape
  
  # Lg
  if rg_ref == None:
      rg_ref=int(Nr/2)
  if az_ref == None:
      az_ref=int(Na/2)
  Cref = cov[rg_ref+Wp, az_ref+Wp] # padded
  Lg = [[(np.multiply(np.abs(np.linalg.det(Cref)), np.abs(np.linalg.det(cov[i,j])))) / (np.abs(np.linalg.det((Cref+cov[i,j])/2))**(2)) for j in range(Na+2*Wp)] for i in range(Nr+2*Wp)]
  
  # sim
  sim = np.zeros((Nr, Na))
  for i in range(Nr):
      for j in range(Na):
          range_rg = range(i, i+2*Wp+1) # padded
          range_az = range(j, j+2*Wp+1) # padded
          sim[i,j] = np.sum(-np.log10([[Lg[k, l] for l in range_az] for k in range_rg]))
  
  return sim

# similarity 
wr, wa = 5, 5
Wp = 4
impad = np.pad(I_interp, ((Wp, Wp), (Wp, Wp), (0, 0)), 'wrap') # best, or also 'minimum'
Cov_sim, _ = generate_covariance_matrix(impad, az_ax, rg_ax, wa, wr, gaussian=True)
similarity = compute_NLSAR_dissimilarity(Cov_sim, Wp, rg_ref=int(Wrg/2)*Waz, az_ref=int(Waz/2))

# input = process_input_similarity(bf_training, int(Wrg/2)*Waz + int(Waz/2))
input = bf_training/np.sum(bf_training + 1e-5, axis=2)[..., None]
#neighbors_simil = [process_input_similarity(neighbors_training[k], int(Wrg/2)*Waz + int(Waz/2)) for k in range(len(input))]
indices = np.arange(input.shape[0])
X_train, X_test, neighbors_train, neighbors_test, indices_train, indices_test = train_test_split(input, neighbors_training, indices, test_size=test_size, random_state=42)
print(X_train.shape, X_test.shape)
print(neighbors_train.shape, neighbors_test.shape)

# pytorch variables
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
neighbors_train = torch.from_numpy(neighbors_train.astype(np.float32))
neighbors_test = torch.from_numpy(neighbors_test.astype(np.float32))

# create dataset
class Data(Dataset):
    def __init__(self, X_t, neighbors_t, indices_t, Wrg, Waz):
        self.X = X_t
        self.neighbors = neighbors_t
        self.indices = indices_t
        self.len = self.X.shape[0]
        self.Wrg = Wrg
        self.Waz = Waz

    def __getitem__(self, index):
        # neigh = self.neighbors[index].reshape(self.neighbors[index].shape[0], self.Wrg*2-1, self.Waz*2-1, self.neighbors[index].shape[2])
        # neigh = torch.movedim(neigh/torch.sum(neigh+1e-5, axis=-1)[..., None], -1, 1)
        # patches = neigh.unfold(2, Wrg, 1).unfold(3, Waz, 1)
        # patches = patches.reshape(patches.shape[0], patches.shape[1], -1, 8, 17)
        # cosim = 1 - torch.abs(torch.nn.CosineSimilarity(dim=1)(patches, patches[:, :, :, int(Wrg/2), int(Waz/2)][..., None, None]))
        # patches_sim = torch.cat((patches, torch.unsqueeze(cosim, 1)), axis=1)
        # return self.X[index], torch.movedim(patches_sim, 2, 1), self.indices[index]

        neigh = self.neighbors[index].reshape(self.Wrg*2-1, self.Waz*2-1, self.neighbors[index].shape[1])
        neigh = torch.movedim(neigh/torch.sum(neigh+1e-5, axis=-1)[..., None], -1, 0)
        patches = neigh.unfold(1, self.Wrg, 1).unfold(2, self.Waz, 1)
        patches = patches.reshape(patches.shape[0], -1, self.Wrg, self.Waz)
        cosim = 1 - torch.abs(torch.nn.CosineSimilarity(dim=0)(patches, patches[:, :, int(self.Wrg/2), int(self.Waz/2)][..., None, None]))
        patches_sim = torch.stack((patches, cosim[None].expand(patches.shape[0], patches.shape[1], self.Wrg, self.Waz)), dim=-1)
        return self.X[index], torch.movedim(patches_sim, 1, 0), self.indices[index]

    def __len__(self):
        return self.len

data = Data(X_train, neighbors_train, indices_train, Wrg, Waz)
trainloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=Data(X_test, neighbors_test, indices_test, Wrg, Waz), batch_size=batch_size, shuffle=True)

print(f"Shape of data: {data.X.shape}")


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

        self.linearw = nn.Linear(Wrg*Waz, Wrg*Waz, bias=False)
        self.conv2d = nn.Conv2d(input, input, kernel_size=(Wrg, Waz), bias=False)
        self.conv3d = nn.Conv3d(input, input, kernel_size=(Wrg, Waz, 2), bias=False)

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

    def forward(self, x, w):
        w = F.softmax(self.linearw(w.reshape(-1)), dim=-1)
        c = F.softmax(self.conv2d(x * w.reshape(Wrg, Waz)), dim=0)
        z = self.encoder(torch.squeeze(c))
        x = self.decoder(z)
        return x, z, c, w

# parameters
input_conv = tuple(X_train[0].shape)  # size of the window
input_dim = X_train[0].shape[0]    # number of variables
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

# Hyperparameters
learning_rate = exp_hyperparams["learning_rate"][int(experiment[keys.index("learning_rate")])]
# learning_rate = 0.001
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def transfo_simu_bf_pipeline(profile, kz, transfo_value, transfo_type=0, device='cpu', Nlook=60, pos_0=128, Nz=512, z=z):
  if transfo_type:
    p_trans = torch.zeros(len(profile), Nz, dtype=torch.cfloat, device=device)
    # dilation
    if transfo_value<1.0:
      p_trans[:,pos_0-int(pos_0*(int(np.ceil(Nz*transfo_value)))/Nz):pos_0+(int(np.ceil(Nz*transfo_value)))-int(pos_0*(int(np.ceil(Nz*transfo_value)))/Nz)] = stretch((profile + 1e-15).type(torch.cfloat), 1./transfo_value)
    else:
      p_trans = stretch((profile + 1e-15).type(torch.cfloat), 1/transfo_value)[:,int(pos_0*int(np.ceil(Nz*transfo_value))/Nz)-pos_0:int(pos_0*int(np.ceil(Nz*transfo_value))/Nz)+Nz-pos_0]
  else:
    # translation
    p_trans = torch.roll(profile, int(transfo_value), dims=1)
    
  p2 = p_trans/(torch.sum(p_trans + 1e-5, axis=1))[..., None]
  Az = torch.exp(1j*torch.matmul(torch.from_numpy(kz)[..., None], torch.from_numpy(z)[None])).to(device)
  pz = torch.matmul(torch.matmul(Az, torch.diag_embed(torch.sqrt(p2+1e-5))),
                    (torch.randn(Az.shape[0], Nz, Nlook, device=device) +
                      1j*torch.randn(Az.shape[0], Nz, Nlook, device=device)) /
                      np.sqrt(2)) + np.sqrt(epsilon)*(torch.randn(Az.shape[0], Nim, Nlook, device=device) +
                                                      1j*torch.randn(Az.shape[0], Nim, Nlook, device=device))/np.sqrt(2)
  res = torch.matmul(torch.transpose(torch.conj(Az), 1, 2), torch.mean(pz, axis=-1)[..., None]) #, pz) # mean with Nlook, when Nlook=1, no change
  p_bf = torch.abs(res)**2 / (torch.sum(torch.abs(res)**2, dim=1, keepdims=True) + 1e-5)

  return torch.squeeze(p_bf) #p_bf

# Training  - CORRECTED SQUARED FINAL NORMALIZED translations Nz/16 normal
torch.manual_seed(0)
stime = time()
nb_epochs = exp_hyperparams["nb_epochs"][int(experiment[keys.index("nb_epochs")])]
# nb_epochs = 20
alpha = exp_hyperparams["alpha"][int(experiment[keys.index("alpha")])]
# alpha = 0.01
epsilon = 1e-1 # thermal noise
Nsimu = 1 # Nlook
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
    # get the inputs; data is a list of [inputs, indices]
    inputs, neighbors, indices_inputs = data
    inputs, neighbors = inputs.to(device), neighbors.to(device)
    # print(i)
    # print(torch.isnan(inputs).any())

    # zero the gradient buffers
    optimizer.zero_grad()

    # get the output
    output, _, convoluted = net(inputs)
    # print(torch.isnan(output).any())
    # compute the transformations
    A_tensor = torch.tensor(A_training[indices_inputs], dtype=torch.cfloat, device=device)
    scale = np.random.uniform(0.1, 1.3, len(output))
    trans_bool = np.random.randint(0, 2, len(output))
    x2_tmp = torch.zeros(len(output), Nz, dtype=torch.cfloat, device=device)
    for k in range(len(output)):
      if trans_bool[k]:
        # dilation
        if scale[k]<1.0:
          x2_tmp[k][pos_0-int(pos_0*(int(np.ceil(Nz*scale[k])))/Nz):pos_0+(int(np.ceil(Nz*scale[k])))-int(pos_0*(int(np.ceil(Nz*scale[k])))/Nz)] = stretch((output[k]+1e-15).type(torch.cfloat), 1./scale[k])
        else:
          x2_tmp[k] = stretch((output[k]+1e-15).type(torch.cfloat), 1/scale[k])[int(pos_0*int(np.ceil(Nz*scale[k]))/Nz)-pos_0:int(pos_0*int(np.ceil(Nz*scale[k]))/Nz)+Nz-pos_0]
      else:
        # translation
        # tmp = np.random.randint(int(Nz/16))-10
        # x2[k] = output[k] @ (torch.diag(torch.ones(Nz-abs(tmp), device=device), diagonal=tmp) + torch.diag(torch.ones(abs(tmp), device=device), diagonal=np.sign(tmp)*(abs(tmp)-Nz))) # reversible
        scale[k] = int(np.random.normal(0, 0.25*Nz/16))
        x2_tmp[k] = torch.roll(output[k], int(scale[k]))
    #x2 = torch.stack(list(map(torch.roll, torch.unbind(output, 0), [np.random.randint(Nz) for k in range(len(outputs))])), 0)

    x2 = x2_tmp/(torch.sum(x2_tmp + 1e-5, axis=1))[..., None]
    # x2 = torch.stack([x2_tmp[k]/torch.sum(x2_tmp[k]+1e-5) for k in range(len(output))])
    # print(torch.isnan(x2).any())
    # zsim_tmp = torch.stack([A_tensor[k] @ torch.diag(torch.sqrt(x2[k]+1e-5)).cfloat() @
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
    res_tmp = torch.matmul(torch.transpose(torch.conj(A_tensor), 1, 2), torch.mean(zsim_tmp, axis=-1)[...,None])
    # bf_tmp = torch.squeeze(torch.stack([torch.abs(torch.transpose(torch.conj(A_tensor[k]), 0, 1) @ torch.mean(zsim_tmp[k], axis=-1)[...,None])**2 /
    #                       (torch.sum(torch.abs(torch.transpose(torch.conj(A_tensor[k]), 0, 1) @ torch.mean(zsim_tmp[k], axis=-1)[...,None])**2, axis=0)+1e-5) for k in range(len(output))]))
    bf_tmp = torch.squeeze(torch.abs(res_tmp)**2 / (torch.sum(torch.abs(res_tmp)**2, dim=1, keepdims=True) + 1e-5))
    # DIM 512*100 --> moyenne ? pout 512*1
    # print(torch.isnan(bf_tmp).any())
    inputs_reg = torch.zeros(inputs.shape, dtype=torch.float, device=device)
    idx = int(Wrg/2)*Waz + int(Waz/2)
    with torch.no_grad():
      for k in range(len(output)):
        out_neighbors, _, _ = net(neighbors[k])
        reg_tmp = transfo_simu_bf_pipeline(out_neighbors, kz_neighbors[indices_inputs[k]], scale[k], transfo_type=trans_bool[k], device=device, Nlook=Nsimu).reshape((Wrg, Waz, bf_tmp.shape[1]))
        cosim_reg = 1 - torch.abs(torch.nn.CosineSimilarity(dim=2)(reg_tmp, bf_tmp[k][None, None, ...]))
        inputs_reg[k] = torch.movedim(torch.stack((reg_tmp, torch.unsqueeze(cosim_reg, 2).expand(Wrg, Waz, bf_tmp.shape[1])), dim=-1), -2, 0)
        # inputs_reg[k] = torch.cat(bf_neighbors[:idx], bf_tmp[k], bf_neighbors[idx:], 0).reshape((bf_tmp.shape[1], Wrg, Waz))

    # print(torch.isnan(inputs_reg).any())
    inputs_reg[:,:,int(Wrg/2),int(Waz/2),:] = F.pad(bf_tmp[..., None], pad=(0, 1), mode='constant', value=0) # batch, 512, 8, 17, 2
    # print(bf_tmp.shape)
    # predict from transformed beamforming profile
    x3, _, _ = net(inputs_reg.to(torch.float).to(device))
    # print(torch.isnan(x3).any())

    # optimisation - loss: data attachment term Adiag(p)A'-R + regularization
    R_tensor = torch.tensor(R_training[indices_inputs], dtype=torch.cfloat, device=device)
    # cov_loss = [A_tensor[k] @ torch.diag(output[k]).cfloat() @ torch.transpose(torch.conj(A_tensor[k]), 0, 1) + epsilon * torch.eye(Nim, device=device)
    #                                     for k in range(len(inputs))]
    # corr_loss = [torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-5)) @ cov_loss[k] @ torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-5))
    #                                     for k in range(len(inputs))]
    # loss1 = (torch.sum(torch.stack([torch.square(torch.linalg.norm(corr_loss[k] - R_tensor[k], ord='fro'))
    #                                     for k in range(len(inputs))])))
    # loss2 = alpha * torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3+1e-2) - 10*torch.log10(x2+1e-2), ord=2, dim=1)))
    # loss = loss1 + loss2
    cov_loss = torch.matmul(torch.matmul(A_tensor, torch.diag_embed(output).cfloat()),
                            torch.transpose(torch.conj(A_tensor), 1, 2)) + epsilon * torch.eye(Nim, dtype=torch.cfloat, device=device).reshape((1, Nim, Nim)).repeat(output.shape[0], 1, 1)
    corr_loss = torch.matmul(torch.matmul(torch.diag_embed(1./torch.sqrt(torch.diagonal(cov_loss, dim1=-2, dim2=-1)+1e-5)), cov_loss),
                             torch.diag_embed(1./torch.sqrt(torch.diagonal(cov_loss, dim1=-2, dim2=-1)+1e-5)))
    loss1 = (torch.sum(torch.square(torch.linalg.norm(corr_loss - R_tensor, ord='fro', dim=(-1, -2)))))
    loss2 = alpha * torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3+1e-2) - 10*torch.log10(x2+1e-2), ord=2, dim=1)))
    loss = loss1 + loss2
    # print(torch.isnan(loss).any())

    loss.backward()
    optimizer.step()    # Does the update
    # print(loss.item())
    # statistics
    running_loss += loss.item()
    running_loss1 += loss1.item()
    running_loss2 += loss2.item()
    nb_loss += 1

    # for name, param in net.named_parameters():
    # # print(param)
    #   print(name, param.grad.norm())

  print(f'[{epoch + 1}] Train loss: {running_loss/nb_loss:.7f}; Data attachment: {running_loss1/nb_loss:.7f}; Regularization: {running_loss2/nb_loss:.7f}; Time: {time()-stime:.7f}')
  
  if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
    # test loss
    with torch.no_grad():
      running_loss_val = 0.0
      running_loss_val1 = 0.0
      running_loss_val2 = 0.0
      nb_loss_val = 0

      for j, (X_test_b, neighbors_test_b, indices_test_b) in enumerate(testloader, 0):
        # calculate outputs by running matrices through the network
        outputs_test, _, _ = net(X_test_b.to(device))

        # compute the transformations
        A_tensor_test = torch.tensor(A_training[indices_test_b], dtype=torch.cfloat, device=device)

        # transformation
        scale_test = np.random.uniform(0.1, 1.3, len(outputs_test))
        trans_bool_test = np.random.randint(0, 2, len(outputs_test))
        x2_test = torch.zeros(len(outputs_test), Nz, dtype=torch.cfloat, device=device)
        for k in range(len(outputs_test)):
          if trans_bool_test[k]:
            # dilation
            if scale_test[k]<1.0:
              x2_test[k][pos_0-int(pos_0*(int(np.ceil(Nz*scale_test[k])))/Nz):pos_0+(int(np.ceil(Nz*scale_test[k])))-int(pos_0*(int(np.ceil(Nz*scale_test[k])))/Nz)] = stretch((outputs_test[k]+1e-15).type(torch.cfloat), 1./scale_test[k])
            else:
              x2_test[k] = stretch((outputs_test[k]+1e-15).type(torch.cfloat), 1./scale_test[k])[int(pos_0*int(np.ceil(Nz*scale_test[k]))/Nz)-pos_0:int(pos_0*int(np.ceil(Nz*scale_test[k]))/Nz)+Nz-pos_0]
            # x2_test[k] = x2_test[k]/torch.sum(x2_test[k]+1e-5)
          else:
            # translation
          #   tmp = np.random.randint(int(Nz/16))-10
          #   x2_test[k] = outputs_test[k] @ torch.diag(torch.ones(Nz-abs(tmp), device=device), diagonal=tmp) + torch.diag(torch.ones(abs(tmp), device=device), diagonal=np.sign(tmp)*(abs(tmp)-Nz))) # reversible
            scale_test[k] = int(np.random.normal(0, 0.25*Nz/16))
            x2_test[k] = torch.roll(outputs_test[k], int(scale_test[k]))
            # x2_test[k] = x2_test[k]/torch.sum(x2_test[k]+1e-5)

        x2_test = x2_test/(torch.sum(x2_test + 1e-5, axis=1))[..., None]
        # zsim_test = torch.stack([A_tensor_test[k] @ torch.diag(torch.sqrt(x2_test[k]+1e-15)).cfloat() @
        #                          (torch.randn(Nz, Nlook, device=device) +
        #                           1j*torch.randn(Nz, Nlook, device=device))/np.sqrt(2) +
        #                          np.sqrt(epsilon)*(torch.randn(Nim, Nlook, device=device) +
        #                                            1j*torch.randn(Nim, Nlook, device=device))/np.sqrt(2)
        #                          for k in range(len(outputs_test))])
        zsim_test = torch.matmul(torch.matmul(A_tensor_test, torch.diag_embed(torch.sqrt(x2_test+1e-5))),
                          (torch.randn(outputs_test.shape[0], Nz, Nsimu, device=device) +
                          1j*torch.randn(outputs_test.shape[0], Nz, Nsimu, device=device)) /
                          np.sqrt(2)) + np.sqrt(epsilon)*(torch.randn(outputs_test.shape[0], Nim, Nsimu, device=device) +
                                                          1j*torch.randn(outputs_test.shape[0], Nim, Nsimu, device=device))/np.sqrt(2)
        res_test = torch.matmul(torch.transpose(torch.conj(A_tensor_test), 1, 2), torch.mean(zsim_test, axis=-1)[..., None])
        bf_test = torch.squeeze(torch.abs(res_test)**2 / (torch.sum(torch.abs(res_test)**2, dim=1, keepdims=True) +1e-5))

        test_reg = torch.zeros(X_test_b.shape, dtype=torch.float, device=device)
        idx = int(Wrg/2)*Waz + int(Waz/2)
        for k in range(len(outputs_test)):
          out_neighbors_test, _, _ = net(neighbors_test_b[k].to(device))
          reg_test = transfo_simu_bf_pipeline(out_neighbors_test, kz_neighbors[indices_test_b[k]], scale_test[k], transfo_type=trans_bool_test[k], device=device, Nlook=Nsimu).reshape((Wrg, Waz, bf_test.shape[1]))
          cosim_test = 1 - torch.abs(torch.nn.CosineSimilarity(dim=2)(reg_test, bf_test[k][None, None, ...]))
          test_reg[k] = torch.movedim(torch.stack((reg_test, torch.unsqueeze(cosim_test, 2).expand(Wrg, Waz, bf_test.shape[1])), axis=-1), -2, 0)
          # inputs_reg[k] = torch.cat(bf_neighbors[:idx], bf_tmp[k], bf_neighbors[idx:], 0).reshape((bf_tmp.shape[1], Wrg, Waz))

        test_reg[:,:,int(Wrg/2),int(Waz/2)] = F.pad(bf_test[..., None], pad=(0, 1), mode='constant', value=0) # batch, 513, 8, 17
        
        # predict from translated beamforming profile
        x3_test, _, _ = net(test_reg.to(torch.float).to(device))

        R_tensor_test = torch.tensor(R_training[indices_test_b], dtype=torch.cfloat, device=device)
        # cov_loss = [A_tensor_test[k] @ torch.diag(outputs_test[k]).cfloat() @ torch.transpose(torch.conj(A_tensor_test[k]), 0, 1) + epsilon * torch.eye(Nim, device=device)
        #                                     for k in range(len(X_test_b))]
        # corr_loss = [torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-5)) @ cov_loss[k] @ torch.diag(1./torch.sqrt(torch.diag(cov_loss[k])+1e-5))
        #                                     for k in range(len(X_test_b))]
        # test_loss1 = (torch.sum(torch.stack([torch.square(torch.linalg.norm(corr_loss[k] - R_tensor_test[k], ord='fro'))
        #                                     for k in range(len(X_test_b))])))
        # test_loss2 = alpha * torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3_test+1e-2) - 10*torch.log10(x2_test+1e-2), ord=2, dim=1)))
        # test_loss = test_loss1 + test_loss2
        cov_loss = torch.matmul(torch.matmul(A_tensor_test, torch.diag_embed(outputs_test).cfloat()),
                            torch.transpose(torch.conj(A_tensor_test), 1, 2)) + epsilon * torch.eye(Nim, dtype=torch.cfloat, device=device).reshape((1, Nim, Nim)).repeat(outputs_test.shape[0], 1, 1)
        corr_loss = torch.matmul(torch.matmul(torch.diag_embed(1./torch.sqrt(torch.diagonal(cov_loss, dim1=-2, dim2=-1)+1e-5)), cov_loss),
                                torch.diag_embed(1./torch.sqrt(torch.diagonal(cov_loss, dim1=-2, dim2=-1)+1e-5)))
        test_loss1 = (torch.sum(torch.square(torch.linalg.norm(corr_loss - R_tensor_test, ord='fro', dim=(-1, -2)))))
        test_loss2 = alpha * torch.sum(torch.square(torch.linalg.norm(10*torch.log10(x3_test+1e-2) - 10*torch.log10(x2_test+1e-2), ord=2, dim=1)))
        test_loss = test_loss1 + test_loss2

        running_loss_val += test_loss.item()
        running_loss_val1 += test_loss1.item()
        running_loss_val2 += test_loss2.item()
        nb_loss_val +=1
      
      test_losses.append(running_loss_val/nb_loss_val)
      test_losses1.append(running_loss_val1/nb_loss_val)
      test_losses2.append(running_loss_val2/nb_loss_val)
      running_loss_val = 0.0
      running_loss_val1 = 0.0
      running_loss_val2 = 0.0 # useful?

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
      axs[i%3,i%4].plot(z, inputs[idx][:, int(Wrg/2), int(Waz/2), 0].cpu().detach().numpy(), color='tab:orange', label='bf')
      axs[i%3,i%4].plot(z, output[idx].cpu().detach().numpy(), color='tab:blue', label='predicted')

    fig.tight_layout()
    plt.legend()
    # plt.show()
            
    run[f"training_val/{epoch + 1}"].upload(fig)
    plt.close()

  epoch += 1

total_time = time() - stime
print(f'Finished Training at epoch number {epoch} in {total_time} seconds')



# save model real EI 100
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
# }, './ei_real_slc_100.pth')


print(0)

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

print(1)
## 100 epochs
fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[0][:, int(Wrg/2), int(Waz/2), 0].cpu().detach().numpy(), label='bf')
plt.plot(z, bf_tmp[0].cpu().detach().numpy(), label='bf(x2)')
plt.plot(z, output[0].cpu().detach().numpy(), label='output')
plt.plot(z, x2[0].cpu().detach().numpy(), label='x2')
plt.plot(z, x3[0].cpu().detach().numpy(), label='x3')
# plt.plot(z, labels[0].cpu().detach().numpy(), label='refs', c='k')
plt.legend()
#plt.show()
    
run[f"results/simulated_0"].upload(fig)
plt.close()

fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[10][:, int(Wrg/2), int(Waz/2), 0].cpu().detach().numpy(), label='bf')
plt.plot(z, bf_tmp[10].cpu().detach().numpy(), label='bf(x2)')
plt.plot(z, output[10].cpu().detach().numpy(), label='output')
plt.plot(z, x2[10].cpu().detach().numpy(), label='x2')
plt.plot(z, x3[10].cpu().detach().numpy(), label='x3')
# plt.plot(z, labels[10].cpu().detach().numpy(), label='refs', c='k')
plt.legend()
#plt.show()
    
run[f"results/simulated_10"].upload(fig)
plt.close()
print(2)
def bf_1d_slc(I_def, z, kz, rg_selected, az_selected):
    kz_ra = kz[rg_selected, az_selected,:].reshape(1, -1)
    A = np.exp(-1j*z.reshape(-1,1).dot(kz_ra)) # conjugate transpose

    return np.abs(A @ I_def[rg_selected, az_selected,:])**2

def bf_1d_slc_window(I_def, z, kz, rg_selected, az_selected, Wrg, Waz):
    range_rg = np.clip(range(rg_selected+math.ceil(-Wrg/2), rg_selected+math.ceil(Wrg/2)), 0, I_def.shape[0]-1)
    range_az = np.clip(range(az_selected+math.ceil(-Waz/2), az_selected+math.ceil(Waz/2)), 0, I_def.shape[1]-1)

    return np.asarray([[bf_1d_slc(I_def, z, kz, i, j) for i in range_rg] for j in range_az]).reshape(-1, z.shape[0])

bf_def_predicted = bf_def.copy()
bf_input = np.asarray([bf_1d_slc_window(I_def, z, kz, i, az_out_sel, Wrg, Waz).reshape(Wrg*Waz, -1) for i in range(len(rg_ax))])
bf_input = bf_input/np.sum(bf_input + 1e-5, axis=2)[..., None]
cosim_real = [[1 - np.abs(scipy.spatial.distance.cosine(bf_input[k][i], bf_input[k][int(Wrg/2)*Waz + int(Waz/2)])) for i in range(len(bf_input[k]))] for k in range(len(bf_input))]
bf_def_torch = torch.from_numpy(np.moveaxis(np.stack((bf_input.reshape(bf_input.shape[0], Wrg, Waz, bf_input.shape[2]), np.broadcast_to(np.asarray(cosim_real).reshape(bf_input.shape[0], Wrg, Waz, 1), (bf_input.shape[0], Wrg, Waz, bf_input.shape[2]))), axis=-1), -2, 1).astype(np.float32))
for i in range(Nr):
  with torch.no_grad():
    # calculate predicted profile
    bf_def_predicted[:,i] = net(bf_def_torch[i].to(device))[0].cpu().detach().numpy()


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
print(3)
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


print(4)
params = {
    "rg_out_camp": rg_out_camp,
    "az_out_sel": az_out_sel,
    "interp_goal": interp_goal,
    "test_size": test_size,
    "batch_size": batch_size,
    "alpha": alpha,
    "learning_rate": learning_rate,
    "net_parameters": net.parameters,
    "nb_epochs": nb_epochs,
    "epsilon": epsilon,
    "Wrg": Wrg,
    "Waz": Waz,
}

run["parameters"] = params
run["parameters/az_ax"].extend(az_ax.tolist())
run["parameters/rg_ax"].extend(rg_ax.tolist())

results = {
    "total_time_training": total_time
}

run["results/arrays"] = results
run["results/losses"].extend(losses)
run["results/losses_att"].extend(losses1)
run["results/losses_reg"].extend(losses2)
run["results/test_losses"].extend(test_losses)
run["results/test_losses_att"].extend(test_losses1)
run["results/test_losses_reg"].extend(test_losses2)
run["results/bf_def_predicted"].extend(bf_def_predicted.reshape(-1).tolist())


run.stop()


