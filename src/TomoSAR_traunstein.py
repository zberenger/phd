### IMPORTS ###
import numpy as np
random_seed = 0
np.random.seed(random_seed)
from time import time
import argparse

## DISPLAY
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
mpl.rc('image', cmap='jet')
plt.rcParams.update({'font.size': 30})

## TORCH
import torch
import torch.nn as nn
torch.manual_seed(42)
use_gpu = False

## LOCAL FILES
from generate_covariance_matrix import generate_covariance_matrix, display_COV
from spectral_estimation import *
from data_simulation import *
from utils import *
from model import preprocessing_data_for_nn, Net, training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', dest='data_path', default='Biosar2_L_band_demo_data.mat')
    parser.add_argument('--basedir', dest='basedir', type=str, default='./', help='base dir')
    parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=0, help='gpu flag, 1 for GPU and 0 for CPU')

    args = parser.parse_args()

    data_path = args.data_path
    basedir = args.basedir

    device = "cpu"
    if args.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    ### FIXED PARAMETERS
    Nz = 512 # number of discrete heights
    min_z = -20
    max_z = 30

    Nlook = 60 # number of looks in the simulated data
    Nsamples = 10000 # number of simulated samples
    Nim = 6 # number of simulated images in the stack
    epsilon = 1e-2 # thermal noise
    az_out_sel = 1500 # select one azimuth value for results display

   
    ### AIRBORNE DATA
    z = np.linspace(min_z, max_z, Nz) # heights
    # filename = args.basedir + "Biosar2_L_band_demo_data.mat"
    # I, kz, rg_ax, az_ax = load_mat_data(filename)
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
    I = I_def
    kz = -kz
    Nr, Na, N = I.shape

    Wrg, Waz = 9, 9 # 60 looks
    start_time = time()
    Cov, Corr = generate_covariance_matrix(I, az_ax, rg_ax, Waz, Wrg)
    print('Covariance computed in: {}'.format(time() - start_time))
    #display_COV(Corr)

    start_time = time()            
    bf = beamforming_az(Corr, az_out_sel, z=z, kz=kz, rg_ax=rg_ax)
    print('Beamforming computed in: {}'.format(time() - start_time))

    start_time = time()            
    cp = capon_az(Corr, az_out_sel, z=z, kz=kz, rg_ax=rg_ax)
    print('Capon computed in: {}'.format(time() - start_time))


    # ### SIMULATION
    # # simulate reflectivity vectors composed of 2 Gaussians
    # start_time = time()
    # p, parameters = profile_simulation(z, Nsamples, random_seed=random_seed)
    # print('Profiles simulated in: {}'.format(time() - start_time))

    # # select real A matrices randomly for each sample
    # A_simu, r_a_coord = geometry_simulation(z, kz, Nsamples, I.shape, random_seed)

    # # normalisation step - to obtain the covariance matrix Adiag(p)A normalised
    # p_norm = normalize_simulated_data(p, A_simu)

    # # beamforming on simualted data
    # pbf, cov = beamforming_simulated(p_norm, A_simu, Nlook, epsilon, Nim)
    # # plot_ref_and_one(p_norm, pbf, z, norm=True)


    ### REAL DATA
    np.random.seed(random_seed)
    training_coord = np.load('Traunstein_training_coord_slc.npy')
    print(f"Number of training coordinates: {len(training_coord)}")

    bf_training = np.load('Traunstein_bf_training_slc_Nim6.npy')
    ground_truth = np.load('Traunstein_groundtruth_sup_Nim15.npy')
    print(f"Shape of bf_training: {bf_training.shape}")
    print(f"Shape of ground_truth: {ground_truth.shape}")
    gtruth = np.clip(gaussian_filter1d(ground_truth, 5, axis=1), 0, None)
    gtruth = gtruth/np.sum(gtruth + 1e-5, axis=1)[..., None]

    pbf = np.asarray([beamforming_rg_az(Corr, z, kz[idx[0], idx[1]], idx[0], idx[1]) for idx in training_coord])
    pbf0 = pbf/np.sum(pbf + 1e-5, axis=1)[..., None]
    print(f"Shape of pbf: {pbf0.shape}")
    print(np.max(pbf0))

    input = bf_training/np.sum(bf_training + 1e-5, axis=2)[..., None]
    input = np.moveaxis(input.reshape(input.shape[0], Wrg, Waz, input.shape[2]), -1, 1)
    pbf = np.mean(input.reshape(input.shape[0], input.shape[1], -1), axis=-1)
    print(f"Shape of pbf: {pbf.shape}")
    print(np.max(pbf))

    plt.plot(z, pbf0[0])
    plt.plot(z, pbf[0])
    plt.savefig('results/pbf0.png')
    plt.show(block = False)


    ### NEURAL NETWORK
    trainloader, testloader, X_train, Y_train, X_test, Y_test, indices_train, indices_test = preprocessing_data_for_nn(np.asarray(pbf), gtruth, test_size=0.25, batch_size=32)
    print('X_train shape: {}; X_test shape: {}; Y_train shape: {}; Y_test shape: {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))
    input_dim = X_train[0].reshape(-1).shape[0]    # number of variables
    output_dim = Y_train[0].reshape(-1).shape[0]    # "number of classes"
    latent_space_size = 5
    net = Net(input_dim, output_dim, latent_space_size=latent_space_size)
    print('Net parameters: {}'.format(net.parameters))

    # Hyperparameters
    nb_epochs = 100
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    stime = time()
    net, losses, test_losses, epoch = training(trainloader, X_test, Y_test, net, nb_epochs=nb_epochs, criterion=criterion, optimizer=optimizer, display=True)
    total_time = time() - stime
    print('Finished training at epoch number {} in {} seconds'.format(epoch, total_time))

    ### PLOTS
    plot_losses(losses, test_losses, nb_epochs)

    with torch.no_grad():
        # calculate predicted profile
        outputs_test, _ = net(X_test.reshape((X_test.shape[0], -1)))
        
        # calculate predicted airborne tomogram
        bf_def_predicted = torch.transpose(torch.stack([net(torch.from_numpy(bf[:,i].astype(np.float32)).reshape(-1))[0] for i in range(len(rg_ax))]), 0, 1)
    plot_ref_and_two(np.transpose(gtruth[indices_test.astype(int)]), np.array(pbf)[indices_test.astype(int)], outputs_test.detach().numpy(), z, label_one='beamforming', label_two='predicted', norm_one=True, norm_two=False)
    plot_dB(bf, z)
    plot_dB(cp, z)
    plot_dB(bf_def_predicted.detach().numpy(), z)


## AT THE END
plt.show()

