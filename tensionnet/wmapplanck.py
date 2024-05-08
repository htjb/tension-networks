import numpy as np
import healpy as hp
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise
import cosmopower as cp
from tqdm import tqdm
from scipy.special import loggamma, hyp0f1
from numpy.linalg import slogdet
from scipy.special import ive, iv
import matplotlib.pyplot as plt

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

def jointClGenCP(path):

    """ looks similar to some of the stuff above?? """
    
    cp_nn = cp.cosmopower_NN(restore=True, 
                            restore_filename= path \
                            +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')
    def clf(parameters, lobs, bins):
        
        if type(parameters) == list:
            parameters = np.array(parameters)
        
        if parameters.ndim < 2:
            parameters = np.array([parameters])
        
        params = {'omega_b': parameters[:, 0],
            'omega_cdm': parameters[:, 1],
            'h': parameters[:, -1],
            'n_s': parameters[:, 2],
            'tau_reio': [0.055]*len(parameters[:, 0]),
            'ln10^{10}A_s': parameters[:, 3],
            }
        
        cl = cp_nn.ten_to_predictions_np(params)*1e12*2.7255**2
        lgen = cp_nn.modes

        pnoise = planck_noise(lgen).calculate_noise()
        wnoise = wmap_noise(lgen).calculate_noise()

        pnalm = hp.synalm(pnoise)
        wnalm = hp.synalm(wnoise)

        def rebin(signal, bins):
            indices = bins - 2
            binned_signal = []
            for i in range(len(indices)):
                if indices[i, 0] == indices[i, 1]:
                    binned_signal.append(signal[int(indices[i, 0])])
                else:
                    binned_signal.append(
                        np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
            return np.array(binned_signal)
    
        planck_obs, wmap_obs, cross_obs, binned_theory = [], [], [], []
        for i, cltheory in enumerate(cl):
            alm = hp.synalm(cltheory)

            # calculate ClFF
            pobs = hp.alm2cl(alm+pnalm)
            
            # calculate ClGG
            wobs = hp.alm2cl(alm+wnalm)

            pobs = rebin(pobs, bins)
            wobs = rebin(wobs, bins)

            # calculate ClFG
            cltheory = rebin(cltheory, bins)


            planck_obs.append(pobs)
            wmap_obs.append(wobs)
            binned_theory.append(cltheory)
        pobs = np.array(planck_obs)
        wobs = np.array(wmap_obs)
        binned_theory = np.array(binned_theory)

    
        return pobs, wobs,binned_theory
    return clf

def loghyp0f1(l, x):
    """log(hyp0f1((2*l+1)/2, x**2))"""
    ans1 = np.log(hyp0f1((2*l+1)/2, x**2))
    with np.errstate(divide='ignore'):
        ans2 = np.log(ive((2*l-1)/2,2*x)) + 2*x + loggamma((2*l+1)/2) -(2*l-1)/2*np.log(x)
    ans3 = 2*x - l*np.log(x) + loggamma((2*l+1)/2) - np.log(4*np.pi)/2 
    ans = ans1
    ans = np.where(np.isfinite(ans), ans, ans2)
    ans = np.where(np.isfinite(ans), ans, ans3)
    return ans

def loglikelihood(hatCF, hatCG, C, NF, NG, l):
    """ takes in the observed power spectra, theory, noise and relevant l"""
    D = ((C+NF)*(C+NG) - C**2)/(2*l+1)
    logp = -2*loggamma((2*l+1)/2) - (2*l+1)/2*np.log(4*D/(2*l+1)) - \
        ((C+NG)*hatCF + (C+NF)*hatCG)/(2*D) + \
            (2*l-1)/2*np.log(hatCF*hatCG)
    B = loghyp0f1(l, np.sqrt(hatCF*hatCG)*C/2/D)
    A = np.log(hyp0f1((2*l+1)/2, C**2/(2*D)))
    plt.plot(l, A)
    plt.plot(l, B)
    plt.show()
    return np.sum(logp + B), np.sum(np.isfinite(logp + A))

def bin_planck(bins, lobs):
    l, signal, _, _ = np.loadtxt('cosmology-data/planck_unbinned.txt', unpack=True)
    signal = signal*(2*np.pi)/(l*(l+1))
    indices = bins - 2
    binned_signal = []
    for i in range(len(indices)):
        if indices[i, 0] == indices[i, 1]:
            binned_signal.append(signal[int(indices[i, 0])])
        else:
            binned_signal.append(
                np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
    return np.array(binned_signal)#*(2*np.pi)/(lobs*(lobs+1))