import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
from scipy.stats import chi2, invgamma
from tqdm import tqdm
import camb
import warnings

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

p *= (2*np.pi)/(l_real*(l_real+1)) # convert to C_l

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
    ninst = 1/sigma_T[i]**2* \
        np.exp(-l_real*(l_real+1)*theta_planck[i]**2/(8*np.log(2))) #one over ninst
    nis.append(ninst)
ninst = np.array(nis).T
ninst = np.sum(ninst, axis=1)
noise = 1/ninst
# noise is in the cl space

#mpn = np.loadtxt('montypython_noise.txt')

############# define the likelihood ##############
# there are a bunch of different equatiosn here
# that can be selected with the mode paraemter
#################################################

def likelihood(t, p):
    # camb stuff
    pars.set_cosmology(ombh2=t[0], omch2=t[1],
                        tau=t[3], cosmomc_theta=t[2]/100,
                        theta_H0_range=[5, 1000])
    pars.InitPower.set_params(As=np.exp(t[5])/10**10, ns=t[4])
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_background(pars) # computes evolution of background cosmology

    cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
    cl = np.interp(l_real, np.arange(len(cl)), cl)
    
    cl *= (2*np.pi)/(l_real*(l_real+1)) # convert to C_l
    # if noise then add noise
    cl += noise

    x = (2*l_real + 1)*p/cl
    L = (chi2(2*l_real+1).logpdf(x)).sum()
    sample = chi2.rvs(df=2*l_real + 1, size=len(l_real))
    
    return L, cl, sample

nsamples = 10 # number of samples to draw

A = (l_real*(l_real+1))/(2*np.pi)

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
for j in range(nsamples):
    L, cl, samples = likelihood(wide_prior(np.random.rand(6)), p)
    
    sampled_cl = cl*samples/(2*l_real + 1)
    axes.plot(l_real, A*(sampled_cl - noise), alpha=0.1, c='k')
    axes.plot(l_real, A*(cl - noise), alpha=0.1, c='g')
axes.plot(l_real, A*p, c='r', label='truth')
axes.plot(l_real, A*noise, c='b', label='noise')
#axes.set_ylim(0, 15000)
axes.legend()
axes.set_xscale('log')
axes.set_yscale('log')
plt.savefig('planck_draws.png', dpi=300)
plt.show()




