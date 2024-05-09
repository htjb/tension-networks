import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
import camb
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

p, l = get_data(base_dir='cosmology-data/').get_planck()

def wide_prior(cube):
    # wide prior apart from tau which I left tight
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
    theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
    theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
    return theta

pars = camb.CAMBparams()

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(6.3, 6.3))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

theta = wide_prior(np.random.rand(6))

pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=theta[3], cosmomc_theta=theta[2]/100,
                            theta_H0_range=[5, 1000])
pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
pars.set_for_lmax(2500, lens_potential_accuracy=0)
results = camb.get_background(pars) # computes evolution of background cosmology

cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
#cl = np.interp(l, np.arange(len(cl)), cl)
cl *= 2*np.pi/(np.arange(len(cl))*(np.arange(len(cl))+1))
cl = cl[1:]
lgen = np.arange(len(cl))


pnoise = planck_noise(lgen).calculate_noise()

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

obscl = np.interp(l, np.arange(len(obscl)), obscl)

#cl = np.interp(l, np.arange(len(cl)), cl)
axclp = fig.add_subplot(spec[2, 0])
axclp.plot(lgen, cl*(lgen*(lgen+1))/(2*np.pi), label='Theory')

pnoise = np.interp(l, np.arange(len(pnoise)), pnoise)
A = (l*(l+1))/(2*np.pi)
axclp.plot(l, (obscl - pnoise)*A, label='Obs. Planck')

noise = wmap_noise(lgen).calculate_noise()

nalm = hp.synalm(noise)
axnoisew = fig.add_subplot(spec[1, 1])
plt.axes(axnoisew)
m = hp.alm2map(nalm, nside=2064,)
hp.mollview(m, hold=True, cmap='jet', title='WMAP Noise', unit=r'\small $\mu K$')
obscl = hp.alm2cl(alm+nalm)

wmap, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
obscl = np.interp(lwmap, np.arange(len(obscl)), obscl)
noise = np.interp(lwmap, np.arange(len(noise)), noise)
A = (lwmap*(lwmap+1))/(2*np.pi)

axclw = fig.add_subplot(spec[2, 1])
axclw.plot(lgen, cl*(lgen*(lgen+1))/(2*np.pi))
axclw.plot(lwmap, (obscl - noise)*A, label='Obs. WMAP')

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