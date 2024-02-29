import numpy as np
import camb
import healpy as hp
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

def jointClGen():

    pars = camb.CAMBparams()

    def cl(_, parameters):
        pars.set_cosmology(ombh2=parameters[0], omch2=parameters[1],
                            tau=parameters[3], cosmomc_theta=parameters[2]/100,
                            theta_H0_range=[5, 1000])
        pars.InitPower.set_params(As=np.exp(parameters[5])/10**10, ns=parameters[4])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
        #cl = np.interp(l, np.arange(len(cl)), cl)
        cl *= 2*np.pi/(np.arange(len(cl))*(np.arange(len(cl))+1))
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