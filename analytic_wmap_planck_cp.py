from pypolychord.settings import PolyChordSettings
import pypolychord
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise
from cmblike.cmb import CMB
import numpy as np
from tensionnet.wmapplanck import jointClGenCP
from scipy.stats import wishart

nDims = 5
nDerived = 0

PRETEND_DATA = True
RESUME = False
BASE_DIR = 'analytic-joint-planck-wmap/'
file = 'planck-wmap-analytic/'

p, l = get_data(base_dir='cosmology-data/').get_planck()
pwmap, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
pnoise = planck_noise(l).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

import healpy as hp
palm = hp.synalm(p)
wmapalm = hp.synalm(pwmap)
wnoisealm = hp.synalm(wnoise)
pnoisealm = hp.synalm(pnoise)

clcross = hp.alm2cl(palm+pnoisealm, wmapalm+wnoisealm)

hatCl = np.array([p+pnoise, clcross], [clcross, pwmap+wnoise])


parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

generator = jointClGenCP(cmbs.path_to_cp)

def likelihood(parameters):
    pobs, wobs, crossobs = \
                generator(np.array([parameters]))
    V = np.array([[pobs[0], crossobs[0]], [crossobs[0], wobs[0]]])
    logpdf = wishart.logpdf((2*l+1)*hatCl, 2*l+1, V)
    return logpdf - 2*np.log(2*l+1), []

prior = cmbs.prior

print(likelihood(prior(np.random.uniform(0, 1, nDims))))
sys.exit(1)

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from fgivenx import plot_contours, plot_lines
import matplotlib.pyplot as plt
from anesthetic import read_chains

def signal():
    def signal_func(_, parameters):
        cl, sample = cmbs.get_samples(l, parameters, noise=planck_noise)
        return sample*(l*(l+1))/(2*np.pi)
    return signal_func

fig, axes = plt.subplots(1)

samples = read_chains(BASE_DIR + file + 'test')
samples = samples.compress(1000)

names = ['p' + str(i) for i in range(6)]
samples = samples[names].values
sf = signal()
plot_lines(sf, l, samples, axes, color='r')
plt.plot(l, p*(l*(l+1))/(2*np.pi), c='k', label='Planck')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR + file.split('/')[0] + '.png', dpi=300)
plt.show()
