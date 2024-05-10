from pypolychord.settings import PolyChordSettings
import pypolychord
from cmblike.data import get_data
from cmblike.noise import planck_noise
from cmblike.cmb import CMB
import numpy as np
import matplotlib.pyplot as plt 
from tensionnet.utils import rebin, cosmopower_prior

nDims = 5
nDerived = 0

RESUME = True
BASE_DIR = 'clean-wmap-planck-02052024/'
data_label = ''

file = 'fit_wmap_binned_planck_cp_cp_prior_l_above_124/'
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lplanck, signal_planck, _, _ = np.loadtxt(
    'cosmology-data/planck_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

mask = lwmap > 124
lwmap = lwmap[mask]
bins = bins[mask]

pnoise = planck_noise(lwmap).calculate_noise()

planck_binned_like_wmap = rebin(signal_planck, bins)*2*np.pi/(lwmap*(lwmap+1))


parameters, prior_mins, prior_maxs = cosmopower_prior()

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

likelihood = cmbs.get_likelihood(planck_binned_like_wmap, lwmap, 
                                 noise=pnoise, cp=True, bins=bins)
prior = cmbs.prior

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
        cl, sample = cmbs.get_samples(lwmap, parameters, noise=pnoise, cp=True,
                                       bins=bins)
        return sample*(lwmap*(lwmap+1))/(2*np.pi)
    return signal_func

fig, axes = plt.subplots(1)

samples = read_chains(BASE_DIR + file + 'test')
samples = samples.compress(1000)

names = ['p' + str(i) for i in range(5)]
samples = samples[names].values
sf = signal()
plot_lines(sf, lwmap, samples, axes, color='r')
plt.plot(lwmap, planck_binned_like_wmap*(lwmap*(lwmap+1))/(2*np.pi), 
         c='k', label='Planck')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR + file + file.split('/')[0] + '.png', dpi=300)
plt.show()
