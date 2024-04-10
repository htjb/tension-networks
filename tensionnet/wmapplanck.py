import numpy as np
import camb
import healpy as hp
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise
import cosmopower as cp
from tqdm import tqdm

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

def jointClGen(cosmop=None):

    if not cp:
        pars = camb.CAMBparams()
    else:
        cp_nn = cp.cosmopower_NN(restore=True, 
                                restore_filename= cosmop \
                                +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

    def cl(_, parameters, clexample=None):
        if clexample is None:
            if not cp:
                pars.set_cosmology(ombh2=parameters[0], omch2=parameters[1],
                                    tau=parameters[3], cosmomc_theta=parameters[2]/100,
                                    theta_H0_range=[5, 1000])
                pars.InitPower.set_params(As=np.exp(parameters[5])/10**10, ns=parameters[4])
                pars.set_for_lmax(2500, lens_potential_accuracy=0)
                results = camb.get_background(pars) # computes evolution of background cosmology

                cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
                cl *= 2*np.pi/(np.arange(len(cl))*(np.arange(len(cl))+1))
                cl = cl[1:]
                lgen = np.arange(len(cl))
            else:
                params = {'omega_b': [parameters[0]],
                    'omega_cdm': [parameters[1]],
                    'h': [parameters[-1]],
                    'n_s': [parameters[2]],
                    'tau_reio': [0.055],
                    'ln10^{10}A_s': [parameters[3]],
                    }
                cl = cp_nn.ten_to_predictions_np(params)[0]*1e12*2.7255**2
                lgen = cp_nn.modes
        else:
            cl = clexample
            cl = cl[1:]
            lgen = np.arange(len(cl))

        pnoise = planck_noise(lgen).calculate_noise()
        alm = hp.synalm(cl)
        nalm = hp.synalm(pnoise)
        pobscl = hp.alm2cl(alm+nalm)
        pobscl = np.interp(l, np.arange(len(pobscl)), pobscl)
        pnoise = np.interp(l, np.arange(len(pnoise)), pnoise)

        wnoise = wmap_noise(lgen).calculate_noise()
        nalm = hp.synalm(wnoise)
        wobscl = hp.alm2cl(alm+nalm)

        wobscl = np.interp(lwmap, np.arange(len(wobscl)), wobscl)
        wnoise = np.interp(lwmap, np.arange(len(wnoise)), wnoise)

        return pobscl-pnoise, wobscl-wnoise
    return cl

def jointClGenCP(path):
    
    cp_nn = cp.cosmopower_NN(restore=True, 
                            restore_filename= path \
                            +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')
    def clf(parameters):
        params = {'omega_b': parameters[:, 0],
            'omega_cdm': parameters[:, 1],
            'h': parameters[:, -1],
            'n_s': parameters[:, 2],
            'tau_reio': [0.055]*len(parameters),
            'ln10^{10}A_s': parameters[:, 3],
            }
        
        cl = cp_nn.ten_to_predictions_np(params)*1e12*2.7255**2
        lgen = cp_nn.modes
       
        pnoise = planck_noise(lgen).calculate_noise()
        wnoise = wmap_noise(lgen).calculate_noise()
        pnalm = hp.synalm(pnoise)
        pnoise = np.interp(l, np.arange(len(pnoise)), pnoise)
        wnalm = hp.synalm(wnoise)
        wnoise = np.interp(lwmap, np.arange(len(wnoise)), wnoise)

        
        # calcualte ClFF and ClGG
        cFF, cGG = [], []
        cFG = []
        clTheory, clTheoryNoiseF, clTheoryNoiseG = [], [], []
        for i in tqdm(range(len(cl))):
            alm = hp.synalm(cl[i])

            # calculate ClFF
            pobscl = hp.alm2cl(alm+pnalm)
            pobscl = np.interp(l, np.arange(len(pobscl)), pobscl)

            # calculate ClGG
            wobscl = hp.alm2cl(alm+wnalm)
            wobscl = np.interp(lwmap, np.arange(len(wobscl)), wobscl)

            # calculate ClFG
            cFG.append(hp.alm2cl(alm+pnalm, alm+wnalm))
            cFF.append(pobscl)
            cGG.append(wobscl)

        pobs = np.array(cFF)
        wobs = np.array(cGG)
        crossobs = np.array(cFG)
        return pobs, wobs, crossobs
    return clf