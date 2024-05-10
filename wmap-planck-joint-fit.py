from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior
import pypolychord
from cmblike.noise import wmap_noise, planck_noise
from cmblike.cmb import CMB
import numpy as np
from tensionnet.wmapplanck import loglikelihood
from tensionnet.utils import rebin, cosmopower_prior
import matplotlib.pyplot as plt
import cosmopower as cp

path = '/Users/harrybevins/Documents/Software/cosmopower'
cp_nn = cp.cosmopower_NN(restore=True, 
                            restore_filename= path \
                            +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

def gen(parameters, lobs, bins):
    
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

    return rebin(cl*lgen*(lgen+1)/(2*np.pi), bins)*2*np.pi/(lobs*(lobs+1))


nDims = 5
nDerived = 0

RESUME = True
BASE_DIR = 'clean-wmap-planck-02052024/'

file = 'wmap_planck_joint_fit_cp_cp_prior_l_above_124/'
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt(
    'cosmology-data/wmap_unbinned.txt', unpack=True)
lplanck, signal_planck, _, _ = np.loadtxt(
    'cosmology-data/planck_unbinned.txt', unpack=True)

orig_bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

mask = lwmap > 124
lwmap = lwmap[mask]
bins = orig_bins[mask]

pnoise = planck_noise(lwmap).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

planck_binned_like_wmap = rebin(signal_planck, bins)*2*np.pi/(lwmap*(lwmap+1))
wmap_binned_like_wmap = rebin(wmap_unbinned, bins)*2*np.pi/(lwmap*(lwmap+1))

parameters, prior_mins, prior_maxs = cosmopower_prior()

def prior(cube):
    theta = np.zeros_like(cube)
    #for i in range(nDims-1):
    for i in range(nDims):
        theta[i] = UniformPrior(prior_mins[i], prior_maxs[i])(cube[i])
    #theta[-1] = LogUniformPrior(1e-30, 1)(cube[-1])
    return theta

def likelihood(theta):
    cltheory = gen(theta, lwmap, bins)
    """plt.plot(lwmap, cltheory, label='theory')
    plt.plot(lwmap, planck_binned_like_wmap+pnoise, label='planck')
    plt.plot(lwmap, wmap_binned_like_wmap + wnoise, label='wmap')
    plt.legend()
    plt.show()"""
    return loglikelihood(planck_binned_like_wmap + pnoise, 
                        wmap_binned_like_wmap + wnoise, 
                        cltheory, pnoise, wnoise, 
                        lwmap)[0], []#, flag=theta[-1])[0], []

"""for i in range(1):
    print(likelihood(prior(np.random.uniform(0, 1, nDims))))
exit(1)"""

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file + '/'

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

chains = read_chains(BASE_DIR + file + '/test')
"""chains.plot_2d(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'])
plt.show()"""

planck = read_chains(BASE_DIR + 
        'fit_wmap_binned_planck_cp_cp_prior_l_above_124/test')
wmap = read_chains(BASE_DIR + 
        'wmap_fit_cp_cp_prior_l_above_124/test')

Rsamples = chains.logZ(1000) - planck.logZ(1000) - wmap.logZ(1000)
Rs = Rsamples.mean()
errorRs = Rsamples.std()
print('R = {:.2f} +/- {:.2f}'.format(Rs, errorRs))

"""chains = chains.compress()
values = chains.values[:, :nDims]
flagged = []
for i, v in enumerate(values):
    cltheory = gen(v[:-1], lwmap, bins)
    lL, emax = loglikelihood(planck_binned_like_wmap + pnoise,
                            wmap_binned_like_wmap + wnoise,
                            cltheory, pnoise, wnoise,
                            lwmap, flag=v[-1])
    flagged.append(lwmap[~emax])
#    plt.plot(lwmap[~emax], 
#             (planck_binned_like_wmap+pnoise)[~emax]*lwmap[~emax]*(lwmap[~emax]+1)/2/np.pi, 'o')
#    plt.plot(lwmap[~emax], 
#             (wmap_binned_like_wmap+wnoise)[~emax]*lwmap[~emax]*(lwmap[~emax]+1)/2/np.pi, 'o')
#plt.show()

flagged = np.concatenate(flagged)
plt.hist(flagged, bins=50)
plt.xlabel(r'$l$')
plt.ylabel('Count')
plt.savefig('flagged_wmap_planck.png', dpi=300)
plt.show()
"""