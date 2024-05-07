from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
import pypolychord
from cmblike.data import get_data
from cmblike.noise import wmap_noise, planck_noise
from cmblike.cmb import CMB
import numpy as np
from tensionnet.wmapplanck import jointClGenCP, loglikelihood

nDims = 5
nDerived = 0

RESUME = False
BASE_DIR = 'clean-wmap-planck-02052024/'

file = 'wmap_planck_joint_fit_cp/'
pwmap, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
pplanck = np.loadtxt('cosmology-data/planck_binned_like_wmap.txt')
wnoise = wmap_noise(lwmap).calculate_noise()
pnoise = planck_noise(lwmap).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

def prior(cube):
    theta = np.zeros_like(cube)
    for i in range(nDims):
        theta[i] = UniformPrior(prior_mins[i], prior_maxs[i])(cube[i])
    return theta

generator = jointClGenCP(path='/Users/harrybevins/Documents/Software/cosmopower')
wmap_raw_data = np.loadtxt('cosmology-data/wmap_binned.txt')
bins = np.array([wmap_raw_data[:, 1], wmap_raw_data[:, 2]]).T

def likelihood(theta):
    pobs, wobs, _, cltheory = generator(theta, lwmap, bins)
    return loglikelihood(pobs, wobs, cltheory, pnoise, wnoise, lwmap), []

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file + '/'

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)
