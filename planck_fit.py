import numpy as np
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import camb
import matplotlib.pyplot as plt
from cmbemu.eval import evaluate
from scipy.stats import chi2


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

#power_cov = np.loadtxt('planck_mock_cov.txt')
#inv_cov = np.linalg.inv(power_cov)

predictor = evaluate(base_dir='cmbemu_model_wide/', l=l_real)

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

# from montepython https://github.com/brinckmann/montepython_public/blob/3.6/montepython/likelihoods/fake_planck_bluebook/fake_planck_bluebook.data
theta_planck = np.array([10, 7.1, 5.0]) # in arcmin
sigma_T = np.array([68.1, 42.6, 65.4]) # in muK arcmin

theta_planck *= np.array([np.pi/60/180])
sigma_T *= np.array([np.pi/60/180])

from scipy.special import logsumexp

nis = []
for i in range(len(sigma_T)):
    # from montepython code https://github.com/brinckmann/montepython_public/blob/3.6/montepython/likelihood_class.py#L1096
    ninst = 1/sigma_T[i]**2 + \
        np.exp(-l_real*(l_real+1)*theta_planck[i]**2/(8*np.log(2))) #one over ninst
    nis.append(ninst)
ninst = np.array(nis).T
ninst = np.sum(ninst, axis=1)
noise = 1/ninst
noise *= (l_real*(l_real+1)/(2*np.pi))

def likelihood(theta):
    cl, _ = predictor(theta)

    cl += noise
    
    x = (2*l_real + 1)* p/cl
    L = (-chi2(len(l_real) - 6).logpdf(x) - np.log((2*l_real + 1)/cl)).sum()

    return L, []
    
"""import time
for i in range(5):
    s = time.time()
    print(likelihood(prior(np.random.uniform(0, 1, 6))))
    print(time.time()-s)
sys.exit(1)"""

file = 'Planck_chains_wide/'
RESUME = False
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(wide_prior, likelihood, file, RESUME=RESUME, nDims=6)