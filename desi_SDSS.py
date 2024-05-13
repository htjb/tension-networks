import numpy as np
import matplotlib.pyplot as plt
from tensionnet.robs import run_poly
from tensionnet.bao import DESI_BAO, SDSS_BAO
from pypolychord.priors import UniformPrior, LogUniformPrior
import camb

prior_mins = [0.005, 0.001, 0.8, 1.61, 0.5]
prior_maxs = [0.1, 0.99, 1.2, 3.91, 0.9]
desi_baos = DESI_BAO(data_location='cosmology-data/bao_data/', 
            prior_mins=prior_mins, prior_maxs=prior_maxs)
prior = desi_baos.prior
desi_likelihood = desi_baos.loglikelihood()

file = 'All_the_BAOs/DESI/'
RESUME = True
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, desi_likelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)

sdss_baos = SDSS_BAO(data_location='cosmology-data/bao_data/', 
            prior_mins=prior_mins, prior_maxs=prior_maxs)
sdss_likelihood = sdss_baos.loglikelihood()

file = 'All_the_BAOs/SDSS/'
RESUME = True
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, sdss_likelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)


def joint_loglikelihood(theta):
    return desi_likelihood(theta)[0] + sdss_likelihood(theta)[0], []


file = 'All_the_BAOs/DESI_SDSS/'
RESUME = True
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, joint_loglikelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)

##############################################################################
############################### Calculate R ##################################
##############################################################################

from anesthetic import read_chains

joint = read_chains('All_the_BAOs/DESI_SDSS/test')
desi = read_chains('All_the_BAOs/DESI/test')
sdss = read_chains('All_the_BAOs/SDSS/test')

R = (joint.logZ - desi.logZ - sdss.logZ)
print(R.mean(), R.std())