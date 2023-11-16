import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
from scipy.stats import chi2, invgamma
from tqdm import tqdm
import camb
import warnings

########## check if cmbemu is available ############
# if it is it assumes you have the model in the working
# directory. else use camb.
####################################################

emulator = False

try:
    if emulator:
        from cmbemu.eval import evaluate
    else:
        warnings.warn('Could not import cmbemu. Will use CAMB instead.')
        pars = camb.CAMBparams()
except:
    warnings.warn('Could not import cmbemu. Will use CAMB instead.')
    pars = camb.CAMBparams()

############### load the planck data ###############
# should be in the working directory
####################################################
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

############ if cmbemu load the emulator ############
# should be in the working directory
#####################################################
if emulator:
    predictor = evaluate(base_dir='cmbemu_model_wide/', l=l_real)

################# define priors #####################
# a lot of this is taken from the quantifying tensions
# paper
#####################################################

def narrow_prior(cube):
    # tight around planck as in the qunatifying tensions paper
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.0211, 0.0235)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.108, 0.131)(cube[1]) # omegach2
    theta[2] = UniformPrior(1.038, 1.044)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.938, 1)(cube[4]) # ns
    theta[5] = UniformPrior(2.95, 3.25)(cube[5]) # log(10^10*As)
    return theta

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

######### Calcualte the noie #############
# noise from planck. I stole some code from montepython
# the noise gets plotted on the summary plot and you can
# compare with figure 5 in the CORE inflation paper
# looks good to me?
##########################################

# from montepython https://github.com/brinckmann/montepython_public/blob/3.6/montepython/likelihoods/fake_planck_bluebook/fake_planck_bluebook.data
theta_planck = np.array([10, 7.1, 5.0]) # in arcmin
sigma_T = np.array([68.1, 42.6, 65.4]) # in muK arcmin

# convert to radians
theta_planck *= np.array([np.pi/60/180])
sigma_T *= np.array([np.pi/60/180])

# calculate the noise for each map and multipole
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


############# define the likelihood ##############
# there are a bunch of different equatiosn here
# that can be selected with the mode paraemter
#################################################

def likelihood(t, nn, mode):

    """
    Calcualte a mock cmb likelihood.

    Parameters:
    -----------
    t: array
        the cosmological parameters

    nn: array
        the noise
    
    mode: str
        the likelihood function
    
    """

    # use emulator if available else camb
    if emulator:
        cl, _ = predictor(t)
    else:
        # camb stuff
        pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=theta[3], cosmomc_theta=theta[2]/100,
                            theta_H0_range=[5, 1000])
        pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
        cl = np.interp(l_real, np.arange(len(cl)), cl)
    
    # if noise then add noise
    if nn is not None:
        cl += nn
    
    # calculate the likelihood
    # there is some confusion here
    # in lewis they refer to posteriors as P(D|M) which is not 
    # right... and similarly they refer to a likelihood as P(M|D)
    # which feels wrong... equation 13 in that paper is quite 
    # notationally fun...

    if mode == 'scipy':
        # scipy chi2 with 2l+1 degrees of freedom
        # this is the scipy version of lewis eq 8 in theory
        # chi2*change of variables
        # seems to give higher likelihood to smaller cl
        x = (2*l_real + 1)* p/cl
        L = (-chi2(2*l_real+1).logpdf(x) - np.log((2*l_real + 1)/cl)).sum()
    elif mode == 'lewis-eq8':
        # is this equation a posterior or a likelihood?
        L = (-1/2*(2*l_real + 1)*(np.log(cl) + p/cl - (2*l_real-1)/(2*l_real + 1)*np.log(p))).sum()

    return L, cl

ns = [None, noise] # loop over this to do with and without noise
MODE = 'scipy' # select the likelihood function
PLANCK = True
nsamples = 100 # number of samples to draw

if PLANCK:
    from anesthetic import read_chains
    root = read_chains(root='/Users/harrybevins/Documents/Resources/data.1902.04029/runs_default/chains/planck')
    root = root.compress(10000)

    names = ['omegabh2', 'omegach2', 'theta', 'tau', 'ns', 'logA']
    planck_chains = root[names].values

######### make a nice plot ############
# plot the likelihoods 
# will save a figure showing the example signals vs data
# coloured by likelihood value
# will plot the distribution of likelihood values
# will plot the noise
# will do noise and no noise case
#######################################
u = np.random.uniform(0, 1, (nsamples, 6)) # for the priors
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for j in range(len(ns)):
    # get models and associated likelihoods
    print('Making models and calculating likelihoods...')
    likes, cls = [], []
    for i in tqdm(range(nsamples)):
        if PLANCK:
            theta = planck_chains[i]
        else:
            theta = wide_prior(u[i])
        l, c = likelihood(theta, ns[j], MODE)
        likes.append(l)
        cls.append(c)
    print('Models made...')

    likes = np.array(likes)
    likes -= likes.max()
    likes = np.exp(likes)
    mask = np.isfinite(likes)
    likes = likes[mask]
    cls = np.array(cls)[mask]

    # order
    idx = np.argsort(likes)#[::-1]
    cls = cls[idx]
    likes = likes[idx]
    if j == 0:
        color_likes = likes.copy()

    cb = axes[0, j].scatter(likes, likes, c=likes, cmap='Blues')
    plt.colorbar(cb, label=r'$\log \mathcal{L}$')
    axes[0, j].cla()
    [axes[0, j].plot(l_real, cls[i], c=plt.get_cmap('Blues')(likes[i]/likes.max())) 
        for i in range(len(cls))]
    axes[0, j].plot(l_real, p, c='r', marker='.', ls='-', label='Planck')
    axes[1, j].hist(likes, bins=20, histtype='step', color='k')
    axes[0, j].set_xlabel(r'$l$')
    axes[0, j].set_ylabel(r'$C_l$')
    axes[1, j].set_xlabel(r'$\log \mathcal{L}$')
    axes[1, j].set_ylabel(r'$N$')
    if ns[j] is not None:
        axes[2, j].plot(l_real, ns[j], c='k')
        axes[2, j].set_xlabel(r'$l$')
        axes[2, j].set_ylabel(r'$N_l$')
        axes[2, j].set_yscale('log')
        axes[2, j].set_xscale('log')
    else:
        axes[2, j].set_axis_off()
plt.tight_layout()
plt.savefig('planck_likelihood_analytic_' + MODE +'.png', dpi=300)
plt.show()
