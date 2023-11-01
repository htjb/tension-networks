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

pars = camb.CAMBparams()
pars.set_for_lmax(2500, lens_potential_accuracy=0)

predictor = evaluate(base_dir='cmbemu_model/', l=l_real)

def prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.0211, 0.0235)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.108, 0.131)(cube[1]) # omegach2
    theta[2] = UniformPrior(1.038, 1.044)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.938, 1)(cube[4]) # ns
    theta[5] = UniformPrior(2.95, 3.25)(cube[5]) # log(10^10*As)
    return theta

def likelihood(theta):
    #try:
        """pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=theta[3], cosmomc_theta=theta[2]/100)
        pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        results = camb.get_background(pars) # computes evolution of background cosmology

        cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:, 0]
        cl = np.interp(l_real, np.arange(len(cl)), cl)"""
        cl, _ = predictor(theta)

        plt.plot(_, cl, c='r')
        plt.plot(l_real, p, c='k')
        plt.show()

        #L = -0.5*(p -cl).T @ inv_cov @ (p - cl)
        Lein = -0.5 * np.einsum('i,ij,j', p - cl, inv_cov, p - cl)

        return Lein, []
    #except:
    #    return 1e-300, []
    
import time
for i in range(5):
    s = time.time()
    print(likelihood(prior(np.random.uniform(0, 1, 6))))
    print(time.time()-s)
sys.exit(1)

file = 'Planck_chains/'
RESUME = False
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, likelihood, file, RESUME=RESUME, nDims=6)