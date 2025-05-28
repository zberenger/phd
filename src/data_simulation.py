import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from time import time


def create_p_samples(x):
  # sampling from Gaussians
  ground_canopy_reg = np.random.uniform(0, 1) # ratio ground vs canopy
  mean0 = np.random.uniform(-5,5)
  std0 = np.random.uniform(0.1,2)
  ground_peak = norm.pdf(x, mean0, std0)
  gauss_list = ground_canopy_reg*ground_peak/max(ground_peak) # ground reflection

  std1 = np.random.uniform(0.5,4)
  mean1 = np.random.uniform(max(mean0+std0+2*std1, -2),20)
  crown_peak = norm.pdf(x, mean1, std1)
  gauss_list += (1-ground_canopy_reg)*crown_peak/max(crown_peak)

  param = [mean0, std0, mean1, std1, ground_canopy_reg]

  return gauss_list, param


def profile_simulation(z, Nsamples, random_seed=0):
  # create Nsamples profiles p
  p = []
  parameters = []
  np.random.seed(random_seed)
  for i in range(Nsamples):
    pdf, param = create_p_samples(z)
    p.append(pdf)
    parameters.append(param)

  return p, parameters


def geometry_simulation(z, kz, Nsamples, data_shape, random_seed):
  # associate each Nsamples profile with a steering matrix
  Nr, Na, _ = data_shape
  A = []
  np.random.seed(random_seed)
  r_a_coord = np.transpose(np.random.randint([[Nr], [Na]], size=(2, Nsamples)))
  for i in range(Nsamples):
    A.append(np.transpose(np.exp(-1j*z.reshape(-1,1).dot(kz[r_a_coord[i][0], r_a_coord[i][1], :].reshape(1,-1)))))

  return A, r_a_coord


def normalize_simulated_data(p):
  return np.transpose(np.asarray(p)/np.sum(p, axis=1).reshape(-1, 1)) # samples in the columns
  

def create_z_samples(p, A, Nlook, epsilon):
  # create Nlook independent samples from theoretic p profiles
  zsim = []
  Nim = np.asarray(A).shape[1]

  for i in range(p.shape[1]):
    C = A[i] @ np.diag(p[:, i]) @ np.conj(np.transpose(A[i])) + epsilon * np.eye(Nim) # correlation matrix
    L = np.linalg.cholesky(C) # cholesky step
    noise = np.random.randn(Nim, Nlook)/np.sqrt(2) + 1j*np.random.randn(Nim, Nlook)/np.sqrt(2) # generate random speckle
    zsim.append(L @ noise)

  return zsim


def simulated_correlation(zsim, W):
  # compute correlation from simulated measurement given a window W (number of looks)
  corr = [] # correlation list
  cov = [] # covariance list

  for ns in range(len(zsim)):
    cov_tmp = zsim[ns] @ np.conj(np.transpose(zsim[ns])) / W # covariance
    D = np.diag(1./np.sqrt(np.diag(cov_tmp))) # normalization factor
    corr.append(D @ cov_tmp @ D)
    cov.append(cov_tmp)
  return cov, corr


def beamforming_simulated(p_norm, A_simu, Nlook, epsilon, Nim):
  # compute beamforming profiles
  t0 = time()
  zsim = create_z_samples(p_norm, A_simu, Nlook, epsilon)
  print('Measures simulated in: ', time() - t0)

  # compute covariance and correlation from the simulated data, with a given number of looks Nlook
  t0 = time()
  cov, corr = simulated_correlation(zsim, Nlook)
  print('Correlation matrices computed in: ', time() - t0)

  # beamforming
  pbf = [np.real(np.sum(np.conjugate(A_simu[i]) * (corr[i] @ A_simu[i]), axis=0))/Nim**2 for i in range(len(zsim))]

  return pbf, cov