from pypolychord.settings import PolyChordSettings
import pypolychord
from cmblike.data import get_data
from cmblike.noise import wmap_noise
from cmblike.cmb import CMB
import numpy as np
from tensionnet.utils import rebin, cosmopower_prior

nDims = 5
nDerived = 0

RESUME = False
BASE_DIR = 'clean-wmap-planck-02052024/'

file = 'wmap_fit_cp_cp_prior/'
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt(
    'cosmology-data/wmap_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

mask = lwmap > 0
lwmap = lwmap[mask]
bins = bins[mask]

wnoise = wmap_noise(lwmap).calculate_noise()

wmap_binned_like_wmap = rebin(wmap_unbinned, bins)*2*np.pi/(lwmap*(lwmap+1))

parameters, prior_mins, prior_maxs = cosmopower_prior()

cmbs = CMB(parameters=parameters, prior_mins=prior_mins,
		           prior_maxs=prior_maxs,
                   path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

likelihood = cmbs.get_likelihood(wmap_binned_like_wmap, lwmap,
                                noise=wnoise, cp=True, bins=bins)
prior = cmbs.prior

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file + '/'

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from fgivenx import plot_contours, plot_lines
import matplotlib.pyplot as plt
from anesthetic import read_chains

def signal():
    def signal_func(_, parameters):
        cl, sample = cmbs.get_samples(lwmap, parameters, 
                noise=wmap_noise, cp=True, bins=bins)
        return sample*(lwmap*(lwmap+1))/(2*np.pi)
    return signal_func

fig, axes = plt.subplots(1)

samples = read_chains(BASE_DIR + file + 'test')
samples = samples.compress()

names = ['p' + str(i) for i in range(5)]
samples = samples[names].values
plot_lines(signal, lwmap, samples, axes, color='r')
plt.plot(lwmap, wmap_binned_like_wmap*(lwmap*(lwmap+1))/(2*np.pi), 
         c='k', label='wmap')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR + 'wmap_fit_cp.png', dpi=300)
plt.show()
