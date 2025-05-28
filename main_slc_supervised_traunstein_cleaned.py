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

from scipy.ndimage import gaussian_filter1d

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(0)


### PARAMETERS
Nz = 512 # number of discrete heights
z = np.linspace(-20, 30, Nz) # heights
epsilon = 1e-2 # thermal noise

# simulated data
Nlook = 60 # number of looks in the simulated data
Nsamples = 10000 # number of simulated samples
Nim = 6 # number of simulated images in the stack
kz = np.linspace(0.2, (Nim-1)*0.2 + 0.2, num=Nim)/2
A = np.exp(-1j*kz.reshape(-1,1)*z.reshape(-1,1).transpose())
A_simu = None
az_out_sel = 1500

master = np.load('./17SARTOM/slc_17sartom0102LHH.npy')
print(master.shape)
az_ax = np.linspace(0, master.shape[0], master.shape[0], endpoint=False)
rg_ax = np.linspace(0, master.shape[1], master.shape[1], endpoint=False)

mat = np.zeros((master.shape[1], master.shape[0], 6), dtype=master.dtype) # range, azimuth, tracks
mat[:,:,0] = np.transpose(master)
mat[:,:,1] = np.transpose(np.load('./17SARTOM/slc_17sartom0104LHH.npy'))
mat[:,:,2] = np.transpose(np.load('./17SARTOM/slc_17sartom0106LHH.npy'))
mat[:,:,3] = np.transpose(np.load('./17SARTOM/slc_17sartom0108LHH.npy'))
mat[:,:,4] = np.transpose(np.load('./17SARTOM/slc_17sartom0110LHH.npy'))
mat[:,:,5] = np.transpose(np.load('./17SARTOM/slc_17sartom0112LHH.npy'))

kz = np.zeros((master.shape[1], master.shape[0], 6), dtype=master.dtype) # range, azimuth, tracks
kz[:,:,0] = np.zeros((master.shape[1], master.shape[0]))
kz[:,:,1] = np.transpose(np.load('./17SARTOM/kz_17sartom0104LHH.npy'))
kz[:,:,2] = np.transpose(np.load('./17SARTOM/kz_17sartom0106LHH.npy'))
kz[:,:,3] = np.transpose(np.load('./17SARTOM/kz_17sartom0108LHH.npy'))
kz[:,:,4] = np.transpose(np.load('./17SARTOM/kz_17sartom0110LHH.npy'))
kz[:,:,5] = np.transpose(np.load('./17SARTOM/kz_17sartom0112LHH.npy'))

phadem = np.zeros((master.shape[1], master.shape[0], 6), dtype=master.dtype) # range, azimuth, tracks
phadem[:,:,0] = np.zeros((master.shape[1], master.shape[0]))
phadem[:,:,1] = np.transpose(np.load('./17SARTOM/phadem_17sartom0104LHH.npy'))
phadem[:,:,2] = np.transpose(np.load('./17SARTOM/phadem_17sartom0106LHH.npy'))
phadem[:,:,3] = np.transpose(np.load('./17SARTOM/phadem_17sartom0108LHH.npy'))
phadem[:,:,4] = np.transpose(np.load('./17SARTOM/phadem_17sartom0110LHH.npy'))
phadem[:,:,5] = np.transpose(np.load('./17SARTOM/phadem_17sartom0112LHH.npy'))

# DEM error processings
Nr, Na, N = mat.shape
I_def = np.zeros(mat.shape, dtype=mat.dtype)
rem_dem_flag = True
if rem_dem_flag:
    for n in range(N):
        I_def[:,:,n] = mat[:,:,n] * np.exp(1j*phadem[:,:,n])

I_def = I_def[:,:,:6]
kz = kz[:,:,:6]
Nr, Na, N = I_def.shape
kz_interp, I_interp = -kz, I_def
Nim = kz.shape[2]

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
    
    
Wrg = 9
Waz = 9
t = time()
Cov_def, Corr_def = generate_covariance_matrix(I_interp, az_ax, rg_ax, Waz, Wrg)
print(f"Covariance computed in: {time() - t}")


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

bf_def = bf(Corr_def, kz_interp, az_out_sel)

   

##### REAL ########

np.random.seed(random_seed)
training_coord = np.load('Traunstein_training_coord_slc.npy')
print(f"Number of training coordinates: {len(training_coord)}")

bf_training = np.load('Traunstein_bf_training_slc_Nim6.npy')
ground_truth = np.load('Traunstein_groundtruth_sup_Nim15.npy')
print(f"Shape of bf_training: {bf_training.shape}")
print(f"Shape of ground_truth: {ground_truth.shape}")

# variables
test_size = 0.25
batch_size = 32

# def compute_NLSAR_dissimilarity(cov, Wp, rg_ref=None, az_ref=None):
#     Nr, Na, _, _ = cov.shape
#     # Lg
#     if rg_ref == None:
#         rg_ref=int(Nr/2)
#     if az_ref == None:
#         az_ref=int(Na/2)
#     Cref = cov[rg_ref, az_ref] # already padded
#     Lg = [[(np.multiply(np.abs(np.linalg.det(Cref)), np.abs(np.linalg.det(cov[i,j])))) / (np.abs(np.linalg.det((Cref+cov[i,j])/2))**(2)) for j in range(Na)] for i in range(Nr)]

#     # sim
#     sim = np.zeros((Nr-2*Wp, Na-2*Wp))
#     for i in range(Nr-2*Wp):
#         for j in range(Na-2*Wp):
#             range_rg = range(i, i+2*Wp+1) # padded
#             range_az = range(j, j+2*Wp+1) # padded
#             sim[i,j] = np.sum(-np.log10([[Lg[k][l] for l in range_az] for k in range_rg]))
    
#     return sim

# def compute_NLSAR_dissimilarity_window(cov, coord_selected, Wp, Wrg, Waz, rg_ref=None, az_ref=None):
#     range_rg = np.clip(range(coord_selected[0]+math.ceil(-Wrg/2), coord_selected[0]+math.ceil(Wrg/2)+2*Wp), 0, I_def.shape[0]-1)
#     range_az = np.clip(range(coord_selected[1]+math.ceil(-Waz/2), coord_selected[1]+math.ceil(Waz/2)+2*Wp), 0, I_def.shape[1]-1)

#     return compute_NLSAR_dissimilarity(cov[np.ix_(range_rg, range_az)], Wp)

# # similarity 
# wr, wa = 3, 3
# Wp = 1
# impad = np.pad(I_interp, ((Wp, Wp), (Wp, Wp), (0, 0)), 'wrap') # minimum
# Cov_sim, Corr_sim = generate_covariance_matrix(impad, az_ax, rg_ax, wa, wr, gaussian=True)
# dissimilarity = np.asarray([compute_NLSAR_dissimilarity_window(Cov_sim, training_coord[i], Wp, Wrg, Waz) for i in range(Nsamples)])
# dissimilarity = (np.max(dissimilarity) - dissimilarity)/np.max(np.max(dissimilarity) - dissimilarity)

input = bf_training/np.sum(bf_training + 1e-5, axis=2)[..., None]
input = np.moveaxis(input.reshape(input.shape[0], Wrg, Waz, input.shape[2]), -1, 1)
gtruth = np.clip(gaussian_filter1d(ground_truth, 5, axis=1), 0, None)
gtruth = gtruth/np.sum(gtruth + 1e-5, axis=1)[..., None]
indices = np.arange(input.shape[0])
X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(input, gtruth, indices, test_size=test_size, random_state=42)
print(X_train.shape, X_test.shape)

# pytorch variables
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

# create dataset
class Data(Dataset):
    def __init__(self, X_t, Y_t, indices_t, Wrg, Waz):
        self.X = X_t
        self.Y = Y_t
        self.indices = indices_t
        self.len = self.X.shape[0]
        self.Wrg = Wrg
        self.Waz = Waz

    def __getitem__(self, index):
        return self.X[index], self.indices[index], self.Y[index]

    def __len__(self):
        return self.len

data = Data(X_train, Y_train, indices_train, Wrg, Waz)
trainloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=Data(X_test, Y_test, indices_test, Wrg, Waz), batch_size=batch_size, shuffle=True)

print(f"Shape of data: {data.X.shape}")


# Neural network architecture
class Net(nn.Module):
    def __init__(self, input, H1, H2, H3, H4, H5, H6, H7, output, input_conv, Wrg, Waz):
        super(Net,self).__init__()
        # encoder
        self.linear1 = nn.Linear(input,H1,bias=False)
        self.linear2 = nn.Linear(H1,H2,bias=False)
        self.linear3 = nn.Linear(H2,H3,bias=False)
        # self.linear3 = nn.Linear(H1,H3,bias=False)
        self.linear4 = nn.Linear(H3,H4,bias=False)

        # decoder
        self.linear5 = nn.Linear(H4,H5,bias=False)
        self.linear6 = nn.Linear(H5,H6,bias=False)
        self.linear7 = nn.Linear(H6,H7,bias=False)
        # self.linear7 = nn.Linear(H5,H7,bias=False)
        self.linear8 = nn.Linear(H7,output,bias=False)

        torch.nn.init.uniform_(self.conv2d.weight)

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
        x = F.softmax(self.linear8(x), dim=-1)
        return x

    def forward(self, x):
        c = torch.mean(x.reshape(x.shape[0], x.shape[1], -1), dim=-1)
        z = self.encoder(c)
        x = self.decoder(z)
        return x, z, torch.squeeze(c)

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
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# Training  
torch.manual_seed(0)
stime = time()
nb_epochs = 100
epsilon = 1e-2 # thermal noise

losses = []
test_losses = []
epoch = 0

while epoch < nb_epochs:
  running_loss = 0.0
  nb_loss = 0

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, indices]
    inputs, indices_inputs, labels = data # inputs (Wrg, Waz), sim (Wrg, Waz)
    inputs, labels = inputs.to(device), labels.to(device)
    # zero the gradient buffers
    optimizer.zero_grad()

    # get the output
    output, _, convoluted = net(inputs)
    # compute the transformations
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()    # Does the update

    # statistics
    running_loss += loss.item()
    nb_loss += 1

  print(f'[{epoch + 1}] Train loss: {running_loss/nb_loss:.7f}; Time: {time()-stime:.7f}')
  
  if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
    # test loss
    with torch.no_grad():
      running_loss_val = 0.0
      nb_loss_val = 0

      for j, (X_test_b, indices_test_b, Y_test_b) in enumerate(testloader, 0):
        # calculate outputs by running matrices through the network
        outputs_test, _, _ = net(X_test_b.to(device))
        test_loss = criterion(outputs_test, Y_test_b.to(device))

        running_loss_val += test_loss.item()
        nb_loss_val +=1
      
      test_losses.append(running_loss_val/nb_loss_val)
      running_loss_val = 0.0

    # train loss
    if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
      print(f'[{epoch + 1}, {i + 1:5d}] Train loss: {running_loss/nb_loss:.7f};  Test loss: {test_loss.item():.7f}; Time: {time()-stime:.7f}')
    losses.append(running_loss/nb_loss)
    running_loss = 0.0

  if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
    # plot results
    fig, axs = plt.subplots(3,4,figsize=(15,10),sharey=True)
    fig2, axs2 = plt.subplots(3,4,figsize=(15,10),sharey=True)
    for i in range(12):
      idx = np.random.randint(len(inputs))
      # plot prediction
      axs[i%3,i%4].plot(z, inputs[idx][:, int(Wrg/2), int(Waz/2)].cpu().detach().numpy(), color='tab:orange', label='bf slc') ## TODO: reshape?
      axs[i%3,i%4].plot(z, output[idx].cpu().detach().numpy(), color='tab:blue', label='predicted')
      axs[i%3,i%4].plot(z, convoluted[idx].cpu().detach().numpy(), color='tab:green', label='averaged bf')
      axs[i%3,i%4].plot(z, labels[idx].cpu().detach().numpy(), label='ref', c='k')

    fig.tight_layout()
    axs[0,0].legend(bbox_to_anchor=(6., 1.))

    fig2.tight_layout()
    plt.show()

  epoch += 1

total_time = time() - stime
print(f'Finished Training at epoch number {epoch} in {total_time} seconds')

# print loss statistics - with translation 2 and 0.1 1.3 dilation 200 epochs alpha 0.1 true losses
fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses), 'r', label='Train loss')
ax2.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses), 'b', label='Test loss')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
plt.show()

fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(nb_epochs//4, nb_epochs, len(losses)-len(losses)//4), np.array(losses[len(losses)//4:]), 'r', label='Train loss')
ax2.plot(np.linspace(nb_epochs//4, nb_epochs, len(losses)-len(losses)//4), np.array(test_losses[len(losses)//4:]), 'b', label='Test loss')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
plt.show()

fig,ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(np.linspace(nb_epochs//2, nb_epochs, len(losses)-len(losses)//2), np.array(losses[len(losses)//2:]), 'r', label='Train loss')
ax2.plot(np.linspace(nb_epochs//2, nb_epochs, len(losses)-len(losses)//2), np.array(test_losses[len(losses)//2:]), 'b', label='Test loss')
fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
plt.show()


## 100 epochs
fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[0][:, int(Wrg/2), int(Waz/2)].cpu().detach().numpy(), label='bf slc')
plt.plot(z, output[0].cpu().detach().numpy(), label='output')
plt.plot(z, convoluted[0].cpu().detach().numpy(), label='averaged bf')
plt.plot(z, labels[0].cpu().detach().numpy(), label='ref', c='k')
plt.legend()
plt.show()

fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[10][:, int(Wrg/2), int(Waz/2)].cpu().detach().numpy(), label='bf slc')
plt.plot(z, output[10].cpu().detach().numpy(), label='output')
plt.plot(z, convoluted[10].cpu().detach().numpy(), label='averaged bf')
plt.plot(z, labels[10].cpu().detach().numpy(), label='ref', c='k')
plt.legend()
plt.show()


def bf_1d_slc(I_def, z, kz, rg_selected, az_selected):
    kz_ra = kz[rg_selected, az_selected,:].reshape(1, -1)
    A = np.exp(-1j*z.reshape(-1,1).dot(kz_ra)) # conjugate transpose

    return np.abs(A @ I_def[rg_selected, az_selected,:])**2

def bf_1d_slc_window(I_def, z, kz, rg_selected, az_selected, Wrg, Waz):
    range_rg = np.clip(range(rg_selected+math.ceil(-Wrg/2), rg_selected+math.ceil(Wrg/2)), 0, I_def.shape[0]-1)
    range_az = np.clip(range(az_selected+math.ceil(-Waz/2), az_selected+math.ceil(Waz/2)), 0, I_def.shape[1]-1)

    return np.asarray([[bf_1d_slc(I_def, z, kz, i, j) for j in range_az] for i in range_rg]).reshape(-1, z.shape[0])

# bf_def_predicted = bf_def.copy()
bf_input = np.asarray([bf_1d_slc_window(I_interp, z, kz_interp, i, az_out_sel, Wrg, Waz) for i in range(len(rg_ax))])
bf_input = bf_input/np.sum(bf_input + 1e-5, axis=2)[..., None]
bf_input = np.moveaxis(bf_input.reshape(bf_input.shape[0], Wrg, Waz, bf_input.shape[2]), -1, 1)
# for i in range(Nr):
with torch.no_grad():
    # calculate predicted profile
    bf_def_predicted = np.transpose(net(torch.from_numpy(bf_input.astype(np.float32)).to(device))[0].cpu().detach().numpy())

# final reconstructed slice -- log alpha 0.1 100 epochs dilation 0.1 1.3 translation 2 fully simplified corrected varying
fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def[:, 1000:2000], extent=[1000,1000+bf_def[:, 1000:2000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def[:, 2000:3000], extent=[2000,2000+bf_def[:, 2000:3000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def_predicted[:, 1000:2000], extent=[1000,1000+bf_def[:, 1000:2000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def_predicted[:, 2000:3000], extent=[2000,2000+bf_def[:, 2000:3000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN')
plt.colorbar()
plt.show()


# final reconstructed slice (dB)
fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def[:, 1000:2000]))
plt.imshow(tmp, extent=[1000,1000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming (dB)')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def[:, 2000:3000]))
plt.imshow(tmp, extent=[2000,2000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming (dB)')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def_predicted[:, 1000:2000])+10e-15)
plt.imshow(tmp, extent=[1000,1000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN (dB)')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def_predicted[:, 2000:3000])+10e-15)
plt.imshow(tmp, extent=[2000,2000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN (dB)')
plt.colorbar()
plt.show()