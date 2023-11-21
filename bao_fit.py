import numpy as np
import matplotlib.pyplot as plt
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import camb


pars = camb.CAMBparams()

z = np.array([0.38, 0.51, 0.698])

d12 = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
d16 = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])
d12cov = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH_covtot.txt')
d16cov = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH_covtot.txt')

"""d12dm = d12[::2]
d12dh = d12[1::2]
d16dm = d16[::2]
d16dh = d16[1::2]"""

def narrow_prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.0211, 0.0235)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.108, 0.131)(cube[1]) # omegach2
    theta[2] = UniformPrior(1.038, 1.044)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.938, 1)(cube[4]) # ns
    theta[5] = UniformPrior(2.95, 3.25)(cube[5]) # log(10^10*As)
    return theta

def wide_prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
    theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
    theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
    return theta

def likelihood(theta):
    try:
        pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=theta[3], cosmomc_theta=theta[2]/100,
                            theta_H0_range=[5, 1000])
        pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        da = (1+z) * results.angular_diameter_distance(z)
        dh = 3e5/results.hubble_parameter(z) # 1/Mpc
        rs = results.get_derived_params()['rdrag'] # Mpc

        datad12 = [da[0]/rs, dh[0]/rs, da[1]/rs, dh[1]/rs]
        datad16 = [da[2]/rs, dh[2]/rs]

        #L1 = -0.5*np.log(2*np.pi*np.linalg.det(d12cov)) \
        L1 = -0.5*(d12[:, 1] - datad12).T @ np.linalg.inv(d12cov) @ (d12[:, 1] - datad12)
        #L2 = -0.5*np.log(2*np.pi*np.linalg.det(d16cov)) \
        L2 = -0.5*(d16[:, 1] - datad16).T @ np.linalg.inv(d16cov) @ (d16[:, 1] - datad16)

        logl = L1 + L2

        return logl, []
    except:
        return 1e-300, []

prior = wide_prior
file = 'test/'
RESUME = False
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

"""for i in range(10):
    print(likelihood(prior(np.random.uniform(0, 1, 6))))
sys.exit(1)"""


run_poly(prior, likelihood, file, RESUME=RESUME, nDims=6)#, nlive=200*6)
