import neptune
from neptune.utils import stringify_unsupported
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
from scipy.ndimage import gaussian_filter1d
from src.ste_io import rrat


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

f = open("hyperparameters_traunstein.json")
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

mat = np.zeros((master.shape[1], master.shape[0], 15), dtype=master.dtype) # range, azimuth, tracks
mat[:,:,0] = np.transpose(master)
mat[:,:,1] = np.transpose(np.load('./17SARTOM/slc_17sartom0104LHH.npy'))
mat[:,:,2] = np.transpose(np.load('./17SARTOM/slc_17sartom0106LHH.npy'))
mat[:,:,3] = np.transpose(np.load('./17SARTOM/slc_17sartom0108LHH.npy'))
mat[:,:,4] = np.transpose(np.load('./17SARTOM/slc_17sartom0110LHH.npy'))
mat[:,:,5] = np.transpose(np.load('./17SARTOM/slc_17sartom0112LHH.npy'))
mat[:,:,6] = np.transpose(np.load('./17SARTOM/slc_17sartom0114LHH.npy'))
mat[:,:,7] = np.transpose(np.load('./17SARTOM/slc_17sartom0116LHH.npy'))
mat[:,:,8] = np.transpose(np.load('./17SARTOM/slc_17sartom0118LHH.npy'))
mat[:,:,9] = np.transpose(np.load('./17SARTOM/slc_17sartom0120LHH.npy'))
mat[:,:,10] = np.transpose(np.load('./17SARTOM/slc_17sartom0122LHH.npy'))
mat[:,:,11] = np.transpose(np.load('./17SARTOM/slc_17sartom0124LHH.npy'))
mat[:,:,12] = np.transpose(np.load('./17SARTOM/slc_17sartom0126LHH.npy'))
mat[:,:,13] = np.transpose(np.load('./17SARTOM/slc_17sartom0128LHH.npy'))
mat[:,:,14] = np.transpose(np.load('./17SARTOM/slc_17sartom0129LHH.npy'))

kz = np.zeros((master.shape[1], master.shape[0], 15), dtype=master.dtype) # range, azimuth, tracks
kz[:,:,0] = np.zeros((master.shape[1], master.shape[0]))
kz[:,:,1] = np.transpose(np.load('./17SARTOM/kz_17sartom0104LHH.npy'))
kz[:,:,2] = np.transpose(np.load('./17SARTOM/kz_17sartom0106LHH.npy'))
kz[:,:,3] = np.transpose(np.load('./17SARTOM/kz_17sartom0108LHH.npy'))
kz[:,:,4] = np.transpose(np.load('./17SARTOM/kz_17sartom0110LHH.npy'))
kz[:,:,5] = np.transpose(np.load('./17SARTOM/kz_17sartom0112LHH.npy'))
kz[:,:,6] = np.transpose(np.load('./17SARTOM/kz_17sartom0114LHH.npy'))
kz[:,:,7] = np.transpose(np.load('./17SARTOM/kz_17sartom0116LHH.npy'))
kz[:,:,8] = np.transpose(np.load('./17SARTOM/kz_17sartom0118LHH.npy'))
kz[:,:,9] = np.transpose(np.load('./17SARTOM/kz_17sartom0120LHH.npy'))
kz[:,:,10] = np.transpose(np.load('./17SARTOM/kz_17sartom0122LHH.npy'))
kz[:,:,11] = np.transpose(np.load('./17SARTOM/kz_17sartom0124LHH.npy'))
kz[:,:,12] = np.transpose(np.load('./17SARTOM/kz_17sartom0126LHH.npy'))
kz[:,:,13] = np.transpose(np.load('./17SARTOM/kz_17sartom0128LHH.npy'))
kz[:,:,14] = np.transpose(np.load('./17SARTOM/kz_17sartom0129LHH.npy'))

phadem = np.zeros((master.shape[1], master.shape[0], 15), dtype=master.dtype) # range, azimuth, tracks
phadem[:,:,0] = np.zeros((master.shape[1], master.shape[0]))
phadem[:,:,1] = np.transpose(np.load('./17SARTOM/phadem_17sartom0104LHH.npy'))
phadem[:,:,2] = np.transpose(np.load('./17SARTOM/phadem_17sartom0106LHH.npy'))
phadem[:,:,3] = np.transpose(np.load('./17SARTOM/phadem_17sartom0108LHH.npy'))
phadem[:,:,4] = np.transpose(np.load('./17SARTOM/phadem_17sartom0110LHH.npy'))
phadem[:,:,5] = np.transpose(np.load('./17SARTOM/phadem_17sartom0112LHH.npy'))
phadem[:,:,6] = np.transpose(np.load('./17SARTOM/phadem_17sartom0114LHH.npy'))
phadem[:,:,7] = np.transpose(np.load('./17SARTOM/phadem_17sartom0116LHH.npy'))
phadem[:,:,8] = np.transpose(np.load('./17SARTOM/phadem_17sartom0118LHH.npy'))
phadem[:,:,9] = np.transpose(np.load('./17SARTOM/phadem_17sartom0120LHH.npy'))
phadem[:,:,10] = np.transpose(np.load('./17SARTOM/phadem_17sartom0122LHH.npy'))
phadem[:,:,11] = np.transpose(np.load('./17SARTOM/phadem_17sartom0124LHH.npy'))
phadem[:,:,12] = np.transpose(np.load('./17SARTOM/phadem_17sartom0126LHH.npy'))
phadem[:,:,13] = np.transpose(np.load('./17SARTOM/phadem_17sartom0128LHH.npy'))
phadem[:,:,14] = np.transpose(np.load('./17SARTOM/phadem_17sartom0129LHH.npy'))

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
    kz_interp, I_interp = interpolate_cubic(-kz, I_def, interp_goal)
    Nim = int(interp_goal)
else:
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
    
    
Wrg = exp_hyperparams["Wrg"][int(experiment[keys.index("Wrg")])]
Waz = exp_hyperparams["Waz"][int(experiment[keys.index("Waz")])]
t = time()
Cov_def, Corr_def = generate_covariance_matrix(I_interp, az_ax, rg_ax, Waz, Wrg)
print(f"Covariance computed in: {time() - t}")
  
# LiDAR filtering
DTM = rrat('./Traunstein_lidar_4Zoe/dtm_mean_Traunstein_2018_17tmpsar_0116_t01L.rat')
CHM = rrat('./Traunstein_lidar_4Zoe/chm_mean_Traunstein_2018_17tmpsar_0116_t01L.rat')
print(DTM.shape)

SR_CHM = CHM[9000:11000, 500:4500].copy()
SR_CHM[SR_CHM < 0] = None
SR_CHM[SR_CHM > 80] = None

SR_DTM = DTM[9000:11000, 500:4500].copy()
SR_DTM[SR_DTM<500] = None
SR_DTM[SR_DTM>1000] = None
print(SR_DTM.shape)

Wrg, Waz = 9,9
Lrg = np.round(Wrg/2)
Laz = np.round(Waz/2)
mean_filter_mask = np.ones((int(2*Laz+1), int(2*Lrg+1)))/((2*Lrg+1)*(2*Laz+1))
nan_mask_CHM = np.isnan(SR_CHM)
nan_mask_DTM = np.isnan(SR_DTM)

SR_CHM_filtered = np.transpose(scipy.signal.convolve2d(np.where(nan_mask_CHM, 0, SR_CHM), mean_filter_mask, mode='same')/scipy.signal.convolve2d(~nan_mask_CHM, mean_filter_mask, mode='same'))
SR_DTM_filtered = np.transpose(scipy.signal.convolve2d(np.where(nan_mask_DTM, 0, SR_DTM), mean_filter_mask, mode='same')/scipy.signal.convolve2d(~nan_mask_DTM, mean_filter_mask, mode='same'))


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



# # only 2 gaussians
# def create_p_sample2(x):
#   # sampling from Gaussians
#   ground_canopy_reg = np.random.uniform(0, 1) # ratio ground vs canopy
#   mean0 = np.random.uniform(-5,5)
#   std0 = np.random.uniform(0.1,2)
#   ground_peak = scipy.stats.norm.pdf(x, mean0, std0)
#   gauss_list = ground_canopy_reg*ground_peak/max(ground_peak) # ground reflection

#   std1 = np.random.uniform(1,4)
#   mean1 = np.random.uniform(max(mean0+std0+2*std1, -2),20)
#   crown_peak = scipy.stats.norm.pdf(x, mean1, std1)
#   gauss_list += (1-ground_canopy_reg)*crown_peak/max(crown_peak)

#   return gauss_list, [mean0, std0, mean1, std1, ground_canopy_reg]

# def create_p_sample_fromheight(x, mean0, mean1, ground_canopy_reg):
#   # sampling from Gaussians
#   std0 = np.random.uniform(0.1,2)
#   ground_peak = scipy.stats.norm.pdf(x, mean0, std0)
#   gauss_list = ground_canopy_reg*ground_peak/max(ground_peak) # ground reflection

#   std1 = np.random.uniform(1,4)
#   crown_peak = scipy.stats.norm.pdf(x, mean1, std1)
#   gauss_list += (1-ground_canopy_reg)*crown_peak/max(crown_peak)

#   return gauss_list, [mean0, std0, mean1, std1, ground_canopy_reg]

# ### simulate reflectivity vectors
# p = []
# parameters = []
# np.random.seed(random_seed)
# t0 = time()
# for i in range(Nsamples):
#   if i%1000 == 0:
#     print(i)
#   pdf, param = create_p_sample2(z)
#   p.append(pdf)
#   parameters.append(param)
# print(f"Time taken for profile simulation: {time() - t0}")

# # normalisation - to obtain APA normalised
# p_norm = np.transpose(np.asarray(p)/np.sum(p, axis=1).reshape(-1, 1))


# def create_neighbors(parameter, Wrg, Waz, z, freq=10, octaves=1, index=0):
#   mean0, _, mean1, _, ground_canopy_reg = parameter
#   ground = np.asarray([[snoise2(x / freq, y / freq, octaves, base=(index+1)) for y in range(Waz)] for x in range(Wrg)]) * np.abs(5-np.abs(mean0))
#   ground = ground - ground[int(Wrg/2), int(Waz/2)] + mean0
#   crown = np.asarray([[snoise2(x / freq, y / freq, octaves, base=(index+1)*10) for y in range(Waz)] for x in range(Wrg)]) * min(20-np.abs(mean1), np.abs(5-np.abs(mean1)))
#   crown = crown - crown[int(Wrg/2), int(Waz/2)] + mean1
#   ratio = np.asarray([[snoise2(x / freq, y / freq, octaves, base=(index+1)*50) for y in range(Waz)] for x in range(Wrg)]) * min(1-np.abs(ground_canopy_reg), np.abs(ground_canopy_reg))
#   ratio = ratio - ratio[int(Wrg/2), int(Waz/2)] + ground_canopy_reg

#   return np.clip([[create_p_sample_fromheight(z, ground[i,j], crown[i,j], ratio[i,j])[0] for j in range(Waz)] for i in range(Wrg)], 0, None)

# p_neighbors = np.asarray([create_neighbors(parameters[i], Wrg, Waz, z, index=i) for i in range(Nsamples)])

# # compute real A matrices randomly for each sample
# def geometry_simulation(z, kz, Nsamples, random_seed):
#   Nr, Na, _ = kz.shape
#   A = []
#   np.random.seed(random_seed)
#   r_a_coord = np.transpose(np.random.randint([[Nr], [Na]], size=(2, Nsamples)))
#   for i in range(Nsamples):
#     A.append(np.transpose(np.exp(-1j*z.reshape(-1,1).dot(kz[r_a_coord[i][0], r_a_coord[i][1], :].reshape(1,-1)))))

#   return np.asarray(A), r_a_coord

# A_simu, r_a_coord = geometry_simulation(z, kz_interp, Nsamples, random_seed)

# def create_z_samples(p, A, Nlook, epsilon):
#   # create independent samples from theoretic p profiles
#   zsim = []
#   Nim = np.asarray(A).shape[1]

#   for i in range(p.shape[1]):
#     C = A[i] @ np.diag(p[:, i]) @ np.conj(np.transpose(A[i])) + epsilon * np.eye(Nim)
#     L = np.linalg.cholesky(C)
#     noise = np.random.randn(Nim, Nlook)/np.sqrt(2) + 1j*np.random.randn(Nim, Nlook)/np.sqrt(2)
#     zsim.append(L @ noise)

#   return zsim

# def create_z_samples_neighbors(p, r_a_coord_index, kz, z, Nlook, epsilon=1e-2):
#   range_rg = np.clip(range(r_a_coord_index[0]+math.ceil(-Wrg/2), r_a_coord_index[0]+math.ceil(Wrg/2)), 0, kz.shape[0]-1)
#   range_az = np.clip(range(r_a_coord_index[1]+math.ceil(-Waz/2), r_a_coord_index[1]+math.ceil(Waz/2)), 0, kz.shape[1]-1)
#   Nim = kz.shape[2]

#   # create independent samples from theoretic p profiles
#   zsim = np.zeros((len(range_rg), len(range_az), Nim, Nlook), dtype=np.complex64)
#   for i in range(len(range_rg)):
#     for j in range(len(range_az)):
#       A = np.transpose(np.exp(-1j*z.reshape(-1,1).dot(kz[range_rg[i], range_az[j], :].reshape(1,-1))))
#       C = A @ np.diag(p[i, j]/np.sum(p[i, j])) @ np.conj(np.transpose(A)) + epsilon * np.eye(Nim)
#       L = np.linalg.cholesky(C)
#       noise = np.random.randn(Nim, Nlook)/np.sqrt(2) + 1j*np.random.randn(Nim, Nlook)/np.sqrt(2)
#       zsim[i,j] = L @ noise

#   return zsim

# # create independent samples from theoretic p profiles
# t0 = time()
# zsim = create_z_samples(p_norm, A_simu, Nlook, epsilon)
# print(f"Time taken to simulate measures: {time() - t0}")

# # compute correlation
# def simulated_correlation(zsim, W):
#   corr = []
#   for ns in range(len(zsim)):
#     cov = zsim[ns] @ np.conj(np.transpose(zsim[ns])) / W
#     D = np.diag(1./np.sqrt(np.diag(cov)))
#     corr.append(D @ cov @ D)
#   return corr

# corr = simulated_correlation(zsim, Nlook)

# # beamforming
# pbf = [np.real(np.sum(np.conjugate(A_simu[i]) * (corr[i] @ A_simu[i]), axis=0))/Nim**2 for i in range(len(zsim))]

# def generate_covariance_matrix_simulated(param, index, r_a_coord_idx, kz, Wx, Wy, Wrg, Waz, Wp, z, gaussian=False):
#   p_neighbor = create_neighbors(param, Wrg+2*Wp, Waz+2*Wp, z, index=index)
#   p_z = np.squeeze(create_z_samples_neighbors(p_neighbor, r_a_coord_idx, kz, z, 1, Wrg+2*Wp, Waz+2*Wp, epsilon))
  
#   Ny, Nx, N = p_z.shape  
#   Lx = np.round(Wx/2)
#   Ly = np.round(Wy/2)
#   mean_filter_mask = np.ones((int(2*Ly+1), int(2*Lx+1)))/((2*Lx+1)*(2*Ly+1)) # 2*+1 for odd box 
#   if gaussian==True:
#       mean_filter_mask = gaussian_kernel(int(2*Ly+1), int(2*Lx+1), sigma=1.)

#   Cov = np.ones((Ny, Nx, N, N), dtype=complex)
#   for n in range(N):
#       In = F[:,:,n]
#       for m in range(n, N):
#           Im = F[:,:,m]
#           Cnm = scipy.signal.convolve2d(In * np.conjugate(Im), mean_filter_mask, mode='same')
          
#           # coherence
#           Cov[:, :, n, m] = Cnm
#           Cov[:, :, m, n] = np.conj(Cnm)
              
#   return Cov

# def generate_bf_slc(param, index, r_a_coord_idx, kz, Wrg, Waz, z):
#   p_neighbor = create_neighbors(param, Wrg, Waz, z, index=index)
  
#   range_rg = np.clip(range(r_a_coord_idx[0]+math.ceil(-Wrg/2), r_a_coord_idx[0]+math.ceil(Wrg/2)), 0, kz.shape[0]-1)
#   range_az = np.clip(range(r_a_coord_idx[1]+math.ceil(-Waz/2), r_a_coord_idx[1]+math.ceil(Waz/2)), 0, kz.shape[1]-1)
#   Nim = kz.shape[2]
#   Nz = len(z)
#   Nlook = 1

#   # create independent samples from theoretic p profiles
#   bf = np.zeros((len(range_rg), len(range_az), Nz))
#   for i in range(len(range_rg)):
#     for j in range(len(range_az)):
#       A = np.transpose(np.exp(-1j*z.reshape(-1,1).dot(kz[range_rg[i], range_az[j], :].reshape(1,-1))))
#       C = A @ np.diag(p_neighbor[i, j]/np.sum(p_neighbor[i, j])) @ np.conj(np.transpose(A)) + epsilon * np.eye(Nim)
#       L = np.linalg.cholesky(C)
#       noise = np.random.randn(Nim, Nlook)/np.sqrt(2) + 1j*np.random.randn(Nim, Nlook)/np.sqrt(2)
#       pz = np.squeeze(L @ noise)
#       bf[i,j] = np.abs(np.transpose(np.conj(A)) @ pz)**2

#   return bf


   
   

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
batch_size = exp_hyperparams["batch_size"][int(experiment[keys.index("batch_size")])]

def compute_NLSAR_dissimilarity(cov, Wp, rg_ref=None, az_ref=None):
    Nr, Na, _, _ = cov.shape
    # Lg
    if rg_ref == None:
        rg_ref=int(Nr/2)
    if az_ref == None:
        az_ref=int(Na/2)
    Cref = cov[rg_ref, az_ref] # already padded
    Lg = [[(np.multiply(np.abs(np.linalg.det(Cref)), np.abs(np.linalg.det(cov[i,j])))) / (np.abs(np.linalg.det((Cref+cov[i,j])/2))**(2)) for j in range(Na)] for i in range(Nr)]

    # sim
    sim = np.zeros((Nr-2*Wp, Na-2*Wp))
    for i in range(Nr-2*Wp):
        for j in range(Na-2*Wp):
            range_rg = range(i, i+2*Wp+1) # padded
            range_az = range(j, j+2*Wp+1) # padded
            sim[i,j] = np.sum(-np.log10([[Lg[k][l] for l in range_az] for k in range_rg]))
    
    return sim

def compute_NLSAR_dissimilarity_window(cov, coord_selected, Wp, Wrg, Waz, rg_ref=None, az_ref=None):
    range_rg = np.clip(range(coord_selected[0]+math.ceil(-Wrg/2), coord_selected[0]+math.ceil(Wrg/2)+2*Wp), 0, I_def.shape[0]-1)
    range_az = np.clip(range(coord_selected[1]+math.ceil(-Waz/2), coord_selected[1]+math.ceil(Waz/2)+2*Wp), 0, I_def.shape[1]-1)

    return compute_NLSAR_dissimilarity(cov[np.ix_(range_rg, range_az)], Wp)

# similarity 
wr, wa = 3, 3
Wp = 1
impad = np.pad(I_interp, ((Wp, Wp), (Wp, Wp), (0, 0)), 'wrap') # minimum
Cov_sim, Corr_sim = generate_covariance_matrix(impad, az_ax, rg_ax, wa, wr, gaussian=True)
dissimilarity = np.asarray([compute_NLSAR_dissimilarity_window(Cov_sim, training_coord[i], Wp, Wrg, Waz) for i in range(Nsamples)])
dissimilarity = (np.max(dissimilarity) - dissimilarity)/np.max(np.max(dissimilarity) - dissimilarity)

# input = process_input_similarity(bf_training, int(Wrg/2)*Waz + int(Waz/2))
input = bf_training/np.sum(bf_training + 1e-5, axis=2)[..., None]
input = np.moveaxis(input.reshape(input.shape[0], Wrg, Waz, input.shape[2]), -1, 1)
gtruth = np.clip(gaussian_filter1d(ground_truth, 5, axis=1), 0, None)
gtruth = gtruth/np.sum(gtruth + 1e-5, axis=1)[..., None]
#neighbors_simil = [process_input_similarity(neighbors_training[k], int(Wrg/2)*Waz + int(Waz/2)) for k in range(len(input))]
indices = np.arange(input.shape[0])
X_train, X_test, Y_train, Y_test, indices_train, indices_test, dissimilarity_train, dissimilarity_test = train_test_split(input, gtruth, indices, dissimilarity, test_size=test_size, random_state=42)
print(X_train.shape, X_test.shape)
print(dissimilarity_train.shape, dissimilarity_test.shape)

# pytorch variables
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))
dissimilarity_train = torch.from_numpy(dissimilarity_train.astype(np.float32))
dissimilarity_test = torch.from_numpy(dissimilarity_test.astype(np.float32))

# create dataset
class Data(Dataset):
    def __init__(self, X_t, Y_t, indices_t, dis_t, Wrg, Waz):
        self.X = X_t
        self.Y = Y_t
        self.indices = indices_t
        self.dis = dis_t
        self.len = self.X.shape[0]
        self.Wrg = Wrg
        self.Waz = Waz

    def __getitem__(self, index):
        return self.X[index], self.dis[index], self.indices[index], self.Y[index]

    def __len__(self):
        return self.len

data = Data(X_train, Y_train, indices_train, dissimilarity_train, Wrg, Waz)
trainloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=Data(X_test, Y_test, indices_test, dissimilarity_test, Wrg, Waz), batch_size=batch_size, shuffle=True)

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

        self.linearw = nn.Linear(Wrg*Waz, Wrg*Waz, bias=False)
        self.linearw1 = nn.Linear(Wrg*Waz, 100, bias=False)
        self.linearw2 = nn.Linear(100, 50, bias=False)
        self.linearw3 = nn.Linear(50, 100, bias=False)
        self.linearw4 = nn.Linear(100, Wrg*Waz, bias=False)
        self.conv2dw = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv2d = nn.Conv2d(input, input, kernel_size=(Wrg, Waz), bias=False)
        self.conv3d = nn.Conv3d(input, input, kernel_size=(Wrg, Waz, 2), bias=False)

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
        # x = self.linear8(x)
        # x = F.softmax((x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True), dim=-1)
        x = F.leaky_relu(self.linear8(x))
        x = x / (x.sum(dim=-1, keepdim=True)+1e-15)
        return x

    def forward(self, x, d):
        # # w = F.sigmoid(self.conv2dw(d.reshape(d.shape[0], 1, Wrg, Waz)))
        # # w = F.sigmoid(self.linearw(d.reshape(d.shape[0], -1)))
        # w = F.leaky_relu(self.linearw1(d.reshape(d.shape[0], -1)))
        # # w = F.leaky_relu(self.linearw2(d))
        # # w = F.leaky_relu(self.linearw3(w))
        # w = F.softmax(self.linearw4(w).view(w.shape[0], -1), dim=-1).view(w.shape[0], Wrg, Waz)
        # w = self.linearw4(w).view(w.shape[0], Wrg, Waz)
        # w = torch.squeeze(self.conv2dw(d.reshape(d.shape[0], 1, Wrg, Waz)))
        # w = w / (w.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)+1e-15)
        w = d
        c = torch.sum(torch.sum(x * w.reshape(w.shape[0], 1, Wrg, Waz), dim=3), dim=2)/w.sum(dim=2).sum(dim=1).reshape(w.shape[0], 1)# + x.mean(dim=3).mean(dim=2) # self.conv2d
        c = c / (c.sum(dim=-1, keepdim=True)+1e-15)
        # c = torch.mean(x.reshape(x.shape[0], x.shape[1], -1), dim=-1)
        z = self.encoder(torch.squeeze(c))
        x = self.decoder(z)
        return x, z, torch.squeeze(c), w.reshape(w.shape[0], Wrg, Waz)

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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# Training  - CORRECTED SQUARED FINAL NORMALIZED translations Nz/16 normal
torch.manual_seed(0)
stime = time()
nb_epochs = exp_hyperparams["nb_epochs"][int(experiment[keys.index("nb_epochs")])]
# nb_epochs = 20
epsilon = 1e-2 # thermal noise

losses = []
test_losses = []
epoch = 0

while epoch < nb_epochs:
  running_loss = 0.0
  nb_loss = 0

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, indices]
    inputs, sim, indices_inputs, labels = data # inputs (Wrg, Waz), sim (Wrg, Waz)
    inputs, sim, labels = inputs.to(device), sim.to(device), labels.to(device)
    # print(i)
    # print(torch.isnan(inputs).any())

    # zero the gradient buffers
    optimizer.zero_grad()

    # get the output
    output, _, convoluted, weights = net(inputs, sim)
    # print(torch.isnan(output).any())
    # compute the transformations
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()    # Does the update

    # statistics
    running_loss += loss.item()
    nb_loss += 1

    # for name, param in net.named_parameters():
    # # print(param)
    #   print(name, param.grad.norm())

  print(f'[{epoch + 1}] Train loss: {running_loss/nb_loss:.7f}; Time: {time()-stime:.7f}')
  
  if epoch%(nb_epochs//max(min(20, nb_epochs//2), 1)) == 0:
    # test loss
    with torch.no_grad():
      running_loss_val = 0.0
      nb_loss_val = 0

      for j, (X_test_b, sim_test_b, indices_test_b, Y_test_b) in enumerate(testloader, 0):
        # calculate outputs by running matrices through the network
        outputs_test, _, _, _ = net(X_test_b.to(device), sim_test_b.to(device))
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
      
      axtmp = axs2[i%3,i%4].imshow(weights[idx].cpu().detach().numpy())
      plt.colorbar(axtmp, ax=axs2[i%3,i%4])

    fig.tight_layout()
    axs[0,0].legend(bbox_to_anchor=(6., 1.))

    fig2.tight_layout()
    # plt.show()
            
    run[f"training_val/{epoch + 1}"].upload(fig)
    run[f"training_weights/{epoch + 1}"].upload(fig2)
    plt.close('all')

  epoch += 1

total_time = time() - stime
print(f'Finished Training at epoch number {epoch} in {total_time} seconds')



# save model slc sup 100
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'train_loss': losses,
#     'val_loss': test_losses,
# }, './Traunstein_sup_real_slc_100.pth')


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
plt.plot(z, inputs[0][:, int(Wrg/2), int(Waz/2)].cpu().detach().numpy(), label='bf slc')
plt.plot(z, output[0].cpu().detach().numpy(), label='output')
plt.plot(z, convoluted[0].cpu().detach().numpy(), label='averaged bf')
plt.plot(z, labels[0].cpu().detach().numpy(), label='ref', c='k')
plt.legend()
#plt.show()
    
run[f"results/simulated_0"].upload(fig)
plt.close()

fig = plt.figure(figsize=(20,20))
plt.imshow(weights[0].cpu().detach().numpy())
plt.xlabel('azimuth [m]')
plt.ylabel('range [m]')
plt.colorbar()
#plt.show()
    
run[f"results/weights_0"].upload(fig)
plt.close()

fig = plt.figure(figsize=(20,15))
plt.plot(z, inputs[10][:, int(Wrg/2), int(Waz/2)].cpu().detach().numpy(), label='bf slc')
plt.plot(z, output[10].cpu().detach().numpy(), label='output')
plt.plot(z, convoluted[10].cpu().detach().numpy(), label='averaged bf')
plt.plot(z, labels[10].cpu().detach().numpy(), label='ref', c='k')
plt.legend()
#plt.show()
    
run[f"results/simulated_10"].upload(fig)
plt.close()

fig = plt.figure(figsize=(20,20))
plt.imshow(weights[10].cpu().detach().numpy())
plt.xlabel('azimuth [m]')
plt.ylabel('range [m]')
plt.colorbar()
#plt.show()
    
run[f"results/weights_10"].upload(fig)
plt.close()



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
dissimilarity_real = np.asarray([compute_NLSAR_dissimilarity_window(Cov_sim, [i, az_out_sel], Wp, Wrg, Waz) for i in range(len(rg_ax))])
dissimilarity_real = (np.max(dissimilarity_real) - dissimilarity_real)/np.max(np.max(dissimilarity_real) - dissimilarity_real)
# bf_def_torch = torch.from_numpy(np.moveaxis(np.stack((bf_input.reshape(bf_input.shape[0], Wrg, Waz, bf_input.shape[2]), np.broadcast_to(np.asarray(cosim_real).reshape(bf_input.shape[0], Wrg, Waz, 1), (bf_input.shape[0], Wrg, Waz, bf_input.shape[2]))), axis=-1), -2, 1).astype(np.float32))
# for i in range(Nr):
with torch.no_grad():
    # calculate predicted profile
    bf_def_predicted = np.transpose(net(torch.from_numpy(bf_input.astype(np.float32)).to(device), torch.from_numpy(dissimilarity_real.astype(np.float32)).to(device))[0].cpu().detach().numpy())

# final reconstructed slice -- log alpha 0.1 100 epochs dilation 0.1 1.3 translation 2 fully simplified corrected varying
fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def[:, 1000:2000], extent=[1000,1000+bf_def[:, 1000:2000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_1000_2000"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def[:, 2000:3000], extent=[2000,2000+bf_def[:, 2000:3000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_2000_3000"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def_predicted[:, 1000:2000], extent=[1000,1000+bf_def[:, 1000:2000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_predicted_1000_2000"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
plt.imshow(bf_def_predicted[:, 2000:3000], extent=[2000,2000+bf_def[:, 2000:3000].shape[1],min(z),max(z)], origin='lower')
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_predicted_2000_3000"].upload(fig)
plt.close()


# final reconstructed slice (dB)
fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def[:, 1000:2000]))
plt.imshow(tmp, extent=[1000,1000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming (dB)')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_1000_2000_log"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def[:, 2000:3000]))
plt.imshow(tmp, extent=[2000,2000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after beamforming (dB)')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_2000_3000_log"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def_predicted[:, 1000:2000])+10e-15)
plt.imshow(tmp, extent=[1000,1000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN (dB)')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_predicted_1000_2000_log"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10(np.abs(bf_def_predicted[:, 2000:3000])+10e-15)
plt.imshow(tmp, extent=[2000,2000+tmp.shape[1],min(z),max(z)], origin='lower')
plt.clim(np.max(tmp)-20, np.max(tmp))
axes=plt.gca()
axes.set_aspect(4)
plt.title('HH after NN (dB)')
plt.colorbar()
#plt.show()
    
run[f"results/bf_def_predicted_2000_3000_log"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10((np.abs(bf_def_predicted[:, 1000:2000])+10e-15)/np.max(np.abs(bf_def_predicted[:, 1000:2000])+10e-15))
im=plt.imshow(tmp, extent=[1000,1000+bf_def[:, 1000:2000].shape[1],min(z),max(z)], origin='lower')
plt.plot(np.arange(1000, 2000), SR_DTM_filtered[1000:2000,az_out_sel]-SR_DTM_filtered[ 1000:2000,az_out_sel], 'w', linewidth=3.0)
plt.plot(np.arange(1000, 2000), SR_CHM_filtered[1000:2000, az_out_sel], 'w', linewidth=3.0)
plt.clim(-20, 0)
axes=plt.gca()
axes.set_aspect(4)
plt.locator_params(nbins=5)
plt.xlabel('range [bin]')
plt.ylabel('height z [m]')
plt.colorbar(im,fraction=0.046, pad=0.01)
#plt.show()
    
run[f"results/bf_def_predicted_1000_2000_log_dem"].upload(fig)
plt.close()

fig = plt.figure(figsize=(30,5))
tmp = 10*np.log10((np.abs(bf_def_predicted[:, 2000:3000])+10e-15)/np.max(np.abs(bf_def_predicted[:, 2000:3000])+10e-15))
im=plt.imshow(tmp, extent=[2000,2000+bf_def[:, 2000:3000].shape[1],min(z),max(z)], origin='lower')
plt.plot(np.arange(2000, 3000), SR_DTM_filtered[ 2000:3000,az_out_sel]-SR_DTM_filtered[ 2000:3000,az_out_sel], 'w', linewidth=3.0)
plt.plot(np.arange(2000, 3000), SR_CHM_filtered[ 2000:3000, az_out_sel], 'w', linewidth=3.0)
plt.clim(-20, 0)
axes=plt.gca()
axes.set_aspect(4)
plt.locator_params(nbins=5)
plt.xlabel('range [bin]')
plt.ylabel('height z [m]')
plt.colorbar(im,fraction=0.046, pad=0.01)
#plt.show()
    
run[f"results/bf_def_predicted_2000_3000_log_dem"].upload(fig)
plt.close()



params = {
    "az_out_sel": az_out_sel,
    "interp_goal": interp_goal,
    "test_size": test_size,
    "batch_size": batch_size,
    # "alpha": alpha,
    "learning_rate": learning_rate,
    "net_parameters": net.parameters,
    "nb_epochs": nb_epochs,
    "epsilon": epsilon,
    "Wrg": Wrg,
    "Waz": Waz,
}

run["parameters"] = stringify_unsupported(params)
run["parameters/az_ax"].extend(az_ax.tolist())
run["parameters/rg_ax"].extend(rg_ax.tolist())

results = {
    "total_time_training": total_time
}

run["results/arrays"] = results
run["results/losses"].extend(losses)
run["results/test_losses"].extend(test_losses)
run["results/bf_def_predicted"].extend(bf_def_predicted.reshape(-1).tolist())


run.stop()


