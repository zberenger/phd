import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def conv2(x, y, mode='same'):
    # same conv2 function as in matlab
    return np.rot90(scipy.signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2) # equivalent to no rotation with a filter

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
    # display covariance matrix
    ni, nj, n, _ = COV.shape
    COV_display = np.zeros((ni*n, nj*n), dtype='c16')
    for line in range(0,n):
        for col in range(0, n):
            COV_display[line*ni:line*ni+ni, col*nj:col*nj+nj] = COV[:,:,line,col]
    plt.figure(figsize=(15,15))
    plt.imshow(np.abs(COV_display))  
    plt.show(block = False)