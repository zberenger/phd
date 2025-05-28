### IMPORTS ###
import numpy as np
random_seed = 0
np.random.seed(random_seed)
from time import time
import argparse

## DISPLAY
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
plt.rcParams.update({'font.size': 30})

## TORCH
import torch
import torch.nn as nn
torch.manual_seed(0)
use_gpu = False

## LOCAL FILES
from generate_covariance_matrix import generate_covariance_matrix, display_COV
from spectral_estimation import beamforming_az, capon_az
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
    min_z = -10
    max_z = 30

    Nlook = 60 # number of looks in the simulated data
    Nsamples = 10000 # number of simulated samples
    Nim = 6 # number of simulated images in the stack
    epsilon = 1e-2 # thermal noise
    az_out_sel = 115 # select one azimuth value for results display

   
    ### AIRBORNE DATA
    z = np.linspace(min_z, max_z, Nz) # heights
    filename = args.basedir + "Biosar2_L_band_demo_data.mat"
    I, kz, rg_ax, az_ax = load_mat_data(filename)
    I, kz = I[:,:,:Nim], kz[:,:,:Nim]

    Wrg, Waz = 8, 17 # 60 looks
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

    yue = scipy.io.loadmat('./biosar_profile_az116.mat', squeeze_me=True)#, simplify_cells=True)
    rg_out_camp = yue['rg_out_camp']-1

    ### SIMULATION
    # simulate reflectivity vectors composed of 2 Gaussians
    start_time = time()
    p, parameters = profile_simulation(z, Nsamples, random_seed=random_seed)
    print('Profiles simulated in: {}'.format(time() - start_time))

    # select real A matrices randomly for each sample
    A_simu, r_a_coord = geometry_simulation(z, kz, Nsamples, I.shape, random_seed)

    # normalisation step - to obtain the covariance matrix Adiag(p)A normalised
    p_norm = normalize_simulated_data(p)

    # beamforming on simualted data
    pbf, cov = beamforming_simulated(p_norm, A_simu, Nlook, epsilon, Nim)
    # plot_ref_and_one(p_norm, pbf, z, norm=True)


    ### NEURAL NETWORK
    trainloader, testloader, X_train, Y_train, X_test, Y_test, indices_train, indices_test = preprocessing_data_for_nn(np.asarray(pbf), np.transpose(np.asarray(p_norm)), test_size=0.25, batch_size=32)
    print('X_train shape: {}; X_test shape: {}; Y_train shape: {}; Y_test shape: {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))
    input_dim = X_train[0].reshape(-1).shape[0]    # number of variables
    output_dim = Y_train[0].reshape(-1).shape[0]    # "number of classes"
    latent_space_size = 5
    net = Net(input_dim, output_dim, latent_space_size=latent_space_size)
    print('Net parameters: {}'.format(net.parameters))

    # Hyperparameters
    nb_epochs = 200
    learning_rate = 0.001
    # criterion = nn.CosineEmbeddingLoss()
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
    plot_ref_and_two(p_norm[:, indices_test.astype(int)], np.array(pbf)[indices_test.astype(int)], outputs_test.detach().numpy(), z, label_one='beamforming', label_two='predicted', norm_one=True, norm_two=False)
    plot_dB_dem(bf, z, SR_DTM_filtered[:,az_out_sel], SR_CHM_filtered[:,az_out_sel], rg_out_camp=rg_out_camp)
    plot_dB_dem(cp, z, SR_DTM_filtered[:,az_out_sel], SR_CHM_filtered[:,az_out_sel], rg_out_camp=rg_out_camp)
    plot_dB_dem(bf_def_predicted.detach().numpy(), z, SR_DTM_filtered[:,az_out_sel], SR_CHM_filtered[:,az_out_sel], rg_out_camp=rg_out_camp)
    plot_dB(bf, z)
    plot_dB(cp, z)
    plot_dB(bf_def_predicted.detach().numpy(), z)


## AT THE END
plt.show()

