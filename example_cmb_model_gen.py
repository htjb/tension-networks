import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
import cosmopower as cp
import healpy as hp
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

path = '/Users/harrybevins/Documents/Software/cosmopower'

cp_nn = cp.cosmopower_NN(restore=True, 
        restore_filename= path \
        +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(6.3, 6.3))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

parameters = [0.022, 0.12, 0.96, 3.0448, 0.674]

if type(parameters) == list:
    parameters = np.array(parameters)

if parameters.ndim < 2:
    parameters = np.array([parameters])

params = {'omega_b': parameters[:, 0],
    'omega_cdm': parameters[:, 1],
    'h': parameters[:, -1],
    'n_s': parameters[:, 2],
    'tau_reio': [0.055]*len(parameters[:, 0]),
    'ln10^{10}A_s': parameters[:, 3],
    }

cl = cp_nn.ten_to_predictions_np(params)[0]*1e12*2.7255**2
lgen = cp_nn.modes

pnoise = planck_noise(lgen).calculate_noise()
wnoise = wmap_noise(lgen).calculate_noise()

pnalm = hp.synalm(pnoise)
wnalm = hp.synalm(wnoise)

def rebin(signal, bins):
            indices = bins - 2
            binned_signal = []
            for i in range(len(indices)):
                if indices[i, 0] == indices[i, 1]:
                    binned_signal.append(signal[int(indices[i, 0])])
                else:
                    binned_signal.append(
                        np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
            return np.array(binned_signal)

alm = hp.synalm(cl)
axalm = fig.add_subplot(spec[0, :])

plt.axes(axalm)
m = hp.alm2map(alm, nside=2064,)
hp.mollview(m, hold=True, cmap='jet', title='Anisotropies', unit=r'\small$\mu K$')

nalm = hp.synalm(pnoise)
axnoisep = fig.add_subplot(spec[1, 0])
plt.axes(axnoisep)
m = hp.alm2map(nalm, nside=2064,)
hp.mollview(m, hold=True, cmap='jet', title='Planck Noise', unit=r'\small$\mu K$')
obscl = hp.alm2cl(alm+nalm)

wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

obscl = rebin(obscl, bins)

axclp = fig.add_subplot(spec[2, 0])
axclp.plot(lgen, cl*(lgen*(lgen+1))/(2*np.pi), label='Theory')

pnoise = rebin(pnoise, bins)
A = (lwmap*(lwmap+1))/(2*np.pi)
axclp.plot(lwmap, obscl*A, label='Obs. Planck')

noise = wmap_noise(lgen).calculate_noise()

nalm = hp.synalm(noise)
axnoisew = fig.add_subplot(spec[1, 1])
plt.axes(axnoisew)
m = hp.alm2map(nalm, nside=2064,)
hp.mollview(m, hold=True, cmap='jet', title='WMAP Noise', unit=r'\small $\mu K$')
obscl = hp.alm2cl(alm+nalm)

obscl = rebin(obscl, bins)
A = (lwmap*(lwmap+1))/(2*np.pi)

axclw = fig.add_subplot(spec[2, 1])
axclw.plot(lgen, cl*(lgen*(lgen+1))/(2*np.pi))
axclw.plot(lwmap, obscl*A, label='Obs. WMAP')

axclw.legend()
axclp.legend()
axclw.set_xlabel(r'$l$')
axclp.set_xlabel(r'$l$')
axclp.set_ylabel(r'$\frac{l(l+1)}{2\pi} C_l$')
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig('planck_wmap_example.pdf', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()