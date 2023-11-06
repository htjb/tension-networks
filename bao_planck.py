import numpy as np
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import camb
import matplotlib.pyplot as plt
from cmbemu.eval import evaluate


def load_planck():

    """
    Function to load in the planck power spectrum data.

    Returns
    -------
    p: power spectrum
    ps: the error on the power spectrum
    l_real: the multipoles
    """

    tt = np.loadtxt('TT_power_spec.txt', delimiter=',', dtype=str)

    l_real, p, ps, ns = [], [], [], []
    for i in range(len(tt)):
        if tt[i][0] == 'Planck binned      ':
            l_real.append(tt[i][2].astype('float')) # ell
            p.append(tt[i][4].astype('float')) # power spectrum
            ps.append(tt[i][6].astype('float')) # positive error
            ns.append(tt[i][5].astype('float')) # negative error
    p, ps, l_real = np.array(p), np.array(ps), np.array(l_real)
    return p, ps, l_real

p, _, l_real = load_planck()
power_cov = np.loadtxt('planck_mock_cov.txt')
inv_cov = np.linalg.inv(power_cov)
predictor = evaluate(base_dir='cmbemu_model/', l=l_real)

pars = camb.CAMBparams()

z = np.array([0.38, 0.51, 0.698])

d12 = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
d16 = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])
d12cov = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH_covtot.txt')
d16cov = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH_covtot.txt')

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

def joint_likelihood(theta):
    try:
        cl, _ = predictor(theta)
        Lplanck = -0.5 * np.einsum('i,ij,j', p - cl, inv_cov, p - cl)

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

        Lbaod12 = -0.5*(d12[:, 1] - datad12).T @ np.linalg.inv(d12cov) @ (d12[:, 1] - datad12)
        Lbaod16 = -0.5*(d16[:, 1] - datad16).T @ np.linalg.inv(d16cov) @ (d16[:, 1] - datad16)

        return Lplanck+Lbaod12+Lbaod16, []
    except:
        return 1e-300, []
    
file = 'Planck_bao_chains_wide/'
RESUME = True
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(wide_prior, joint_likelihood, file, RESUME=RESUME, nDims=6)

from anesthetic import read_chains

joint = read_chains('Planck_bao_chains_wide/test')
planck = read_chains('Planck_chains_wide/test')
bao = read_chains('BAO_chains_wide/test')

R = joint.logZ(10000) - planck.logZ(10000) - bao.logZ(10000)
R = R.values
print(np.mean(R), np.std(R))
