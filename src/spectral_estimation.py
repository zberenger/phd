import numpy as np

def beamforming_az(COV, az_selected, z, kz, rg_ax):  
    # apply the beamforming algorithm given a matrice COV, along the az_selected azimuth      
    Sp_estimator = np.zeros((z.shape[0], rg_ax.shape[0]))
    
    for r in range(rg_ax.shape[0]):
        kz_r = kz[r, az_selected,:].reshape(1, -1)
        A = np.exp(-1j*z.reshape(-1,1).dot(kz_r))
        
        cov = COV[r,az_selected,:,:]
        Sp_estimator[:,r] = np.real(np.diag(A @ cov @ np.conjugate(A).T))
        
    return Sp_estimator

def capon_az(COV, az_selected, z, kz, rg_ax):
    # apply the Capon's filter given a matrice COV, along the az_selected azimuth
    Sp_estimator = np.zeros((z.shape[0], rg_ax.shape[0]))
    
    for r in range(rg_ax.shape[0]):
        kz_r = kz[r, az_selected,:].reshape(1, -1)
        A = np.exp(-1j*z.reshape(-1,1).dot(kz_r))
        
        cov = COV[r,az_selected,:,:]
        cov_i = np.linalg.inv(cov + 1e-2*np.eye(cov.shape[0]))
        Sp_estimator[:,r] = 1./np.real(np.diag(A @ cov_i @ np.conjugate(A).T))
        
    return Sp_estimator

def beamforming_rg_az(COV, z, kz, rg, az):
    A = np.exp(-1j*z.reshape(-1,1).dot(kz.reshape(1, -1)))

    cov = COV[rg,az,:,:]
    return np.real(np.diag(A @ cov @ np.conjugate(A).T))

def beamforming_slc(I, az_selected, z, kz, rg_ax):  
    # apply the beamforming algorithm given a matrice COV, along the az_selected azimuth      
    Sp_estimator = np.zeros((z.shape[0], rg_ax.shape[0]))
    
    for r in range(rg_ax.shape[0]):
        kz_r = kz[r, az_selected,:].reshape(1, -1)
        A = np.exp(-1j*z.reshape(-1,1).dot(kz_r))
        Sp_estimator[:,r] = np.abs(A @ I[r, az_selected])**2
        
    return Sp_estimator