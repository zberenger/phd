import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_mat_data(filename):
    # load data in our case - change here to load you data
    mat = scipy.io.loadmat(filename, squeeze_me=True)
    print("Data loaded: {}".format(mat.keys()))

    kz = mat['kz']
    k = mat['k']
    I = k[:,:,:,0] ## HH polarization

    rg_ax = np.linspace(0, I.shape[0], I.shape[0], endpoint=False)
    az_ax = np.linspace(0, I.shape[1], I.shape[1], endpoint=False)

    return I, kz, rg_ax, az_ax


def plot_dB(tomo, z, xlabel='range [bin]'):
    # plot tomogram in dB scale
    plt.figure(figsize=(30,5))
    tmp = 10*np.log10(np.abs(tomo+10e-5)/np.max(np.abs(tomo+10e-5)))
    im=plt.imshow(tmp, extent=[0,tomo.shape[1],min(z),max(z)], origin='lower')
    plt.clim(-20, 0)
    axes=plt.gca()
    axes.set_aspect(4)
    plt.locator_params(nbins=5)
    plt.xlabel(xlabel)
    plt.ylabel('height z [m]')
    plt.colorbar(im,fraction=0.046, pad=0.01)
    plt.savefig('results/plot_dB.png')
    plt.show(block = False)

def plot_dB_dem(tomo, z, SR_DTM_filtered, SR_CHM_filtered, rg_out_camp=None, xlabel='range [bin]'):
    # plot tomogram in dB scale, DEM already for az_out_sel
    plt.figure(figsize=(30,5))
    if rg_out_camp.all() != None:
        tmp = 10*np.log10(np.abs(tomo[:,rg_out_camp]+10e-5)/np.max(np.abs(tomo[:,rg_out_camp]+10e-5)))
        im=plt.imshow(tmp, extent=[0,tomo[:,rg_out_camp].shape[1],min(z),max(z)], origin='lower')
        plt.plot(np.arange(len(rg_out_camp)), SR_DTM_filtered[rg_out_camp]-SR_DTM_filtered[rg_out_camp], 'w', linewidth=3.0)
        plt.plot(np.arange(len(rg_out_camp)), SR_CHM_filtered[rg_out_camp], 'w', linewidth=3.0)
    else:
        tmp = 10*np.log10(np.abs(tomo+10e-5)/np.max(np.abs(tomo+10e-5)))
        im=plt.imshow(tmp, extent=[0,tomo.shape[1],min(z),max(z)], origin='lower')
        plt.plot(np.arange(len(tomo.shape[1])), SR_DTM_filtered-SR_DTM_filtered, 'w', linewidth=3.0)
        plt.plot(np.arange(len(tomo.shape[1])), SR_CHM_filtered, 'w', linewidth=3.0)
    
    plt.clim(-20, 0)
    axes=plt.gca()
    axes.set_aspect(4)
    plt.locator_params(nbins=5)
    plt.xlabel('range [bin]')
    plt.ylabel('height z [m]')
    plt.colorbar(im,fraction=0.046, pad=0.01)
    plt.savefig('results/plot_dB_dem.png')
    plt.show(block = False)

def plot_ref(ref, z):
    # plot simulated profiles
    _, axs = plt.subplots(4, 4, figsize=(20,12), sharey=True, constrained_layout=True)
    for i in range(4 * 4):
        idx = np.random.randint(ref.shape[1])
        # plot prediction
        axs[(i//4),i%4].plot(z, ref[:, idx], '-r', label='ground truth')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig('results/plot_ref.png')
    plt.show(block=False)


def plot_ref_and_one(ref, one, z, label_one='beamforming', norm=False):
    if norm==True:
        tmp_loss = np.sum(np.multiply(one, np.transpose(ref)), axis=1) / np.sum(np.multiply(one, one), axis=1)

    _, axs = plt.subplots(4, 4, figsize=(20,12), sharey=True, constrained_layout=True)
    for i in range(4 * 4):
        idx = np.random.randint(ref.shape[1])

        # plot prediction
        axs[(i//4),i%4].plot(z, ref[:, idx], '-r', label='ground truth')
        if norm==True:
            axs[(i//4),i%4].plot(z, np.asarray(one)[idx]*np.max(tmp_loss[idx]), label=label_one)
        else:
            axs[(i//4),i%4].plot(z, np.asarray(one)[idx], label=label_one)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig('results/plot_ref_and_one.png')
    plt.show(block=False)

def plot_ref_and_two(ref, one, two, z, label_one='beamforming', label_two='predicted', norm_one=False, norm_two=False):
    if norm_one==True:
        tmp_loss_one = np.sum(np.multiply(one, np.transpose(ref)), axis=1) / np.sum(np.multiply(one, one), axis=1)
    if norm_two==True:
        tmp_loss_two = np.sum(np.multiply(two, np.transpose(ref)), axis=1) / np.sum(np.multiply(two, two), axis=1)

    _, axs = plt.subplots(4, 4, figsize=(20,12), sharey=True, constrained_layout=True)
    for i in range(4 * 4):
        idx = np.random.randint(ref.shape[1])

        # plot prediction
        axs[(i//4),i%4].plot(z, ref[:, idx], '-r', label='ground truth')
        if norm_one==True:
            axs[(i//4),i%4].plot(z, np.asarray(one)[idx]*np.max(tmp_loss_one[idx]), label=label_one)
        else:
            axs[(i//4),i%4].plot(z, np.asarray(one)[idx], label=label_one)
        if norm_two==True:
            axs[(i//4),i%4].plot(z, np.asarray(two)[idx]*np.max(tmp_loss_two[idx]), label=label_two)
        else:
            axs[(i//4),i%4].plot(z, np.asarray(two)[idx], label=label_two)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig('results/plot_ref_and_two.png')
    plt.show(block=False)

def plot_losses(losses, test_losses, nb_epochs=100):
    plt.figure(figsize=(15,12), constrained_layout=True)
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, nb_epochs, len(losses)), np.array(losses), label='Train loss')
    plt.plot(np.linspace(0, nb_epochs, len(losses)), np.array(test_losses), label='Test loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')

    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(nb_epochs//4*3, nb_epochs, len(losses)-len(losses)//4*3), np.array(losses[len(losses)//4*3:]), label='Train loss')
    plt.plot(np.linspace(nb_epochs//4*3, nb_epochs, len(losses)-len(losses)//4*3), np.array(test_losses[len(losses)//4*3:]), label='Test loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig('results/plot_losses.png')
    plt.show(block=False)
