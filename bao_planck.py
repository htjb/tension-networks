import numpy as np
from pypolychord.priors import UniformPrior, LogUniformPrior
from pypolychord.settings import PolyChordSettings
import pypolychord
import camb
import matplotlib.pyplot as plt
from scipy.stats import chi2

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
p *= (2*np.pi)/(l_real*(l_real+1))

pars = camb.CAMBparams()

z = np.array([0.38, 0.51, 0.698])

d12 = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
d16 = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])
d12cov = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH_covtot.txt')
d16cov = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH_covtot.txt')


def prior(cube):
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
    theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
    theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
    return theta

# from montepython https://github.com/brinckmann/montepython_public/blob/3.6/montepython/likelihoods/fake_planck_bluebook/fake_planck_bluebook.data
theta_planck = np.array([10, 7.1, 5.0]) # in arcmin
sigma_T = np.array([68.1, 42.6, 65.4]) # in muK arcmin

theta_planck *= np.array([np.pi/60/180])
sigma_T *= np.array([np.pi/60/180])


nis = []
for i in range(len(sigma_T)):
    # from montepython code https://github.com/brinckmann/montepython_public/blob/3.6/montepython/likelihood_class.py#L1096
    ninst = 1/sigma_T[i]**2 + \
        np.exp(-l_real*(l_real+1)*theta_planck[i]**2/(8*np.log(2))) #one over ninst
    nis.append(ninst)
ninst = np.array(nis).T
ninst = np.sum(ninst, axis=1)
noise = 1/ninst
#noise *= (l_real*(l_real+1)/(2*np.pi))

def joint_likelihood(theta):

    pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                        tau=theta[3], cosmomc_theta=theta[2]/100,
                        theta_H0_range=[5, 1000])
    pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_background(pars) # computes evolution of background cosmology

    da = (1+z) * results.angular_diameter_distance(z)
    dh = 3e5/results.hubble_parameter(z) # 1/Mpc
    rs = results.get_derived_params()['rdrag'] # Mpc

    datad12 = [da[0]/rs, dh[0]/rs, da[1]/rs, dh[1]/rs]
    datad16 = [da[2]/rs, dh[2]/rs]

    Lbaod12 = -0.5*(d12[:, 1] - datad12).T @ np.linalg.inv(d12cov) @ (d12[:, 1] - datad12)
    Lbaod16 = -0.5*(d16[:, 1] - datad16).T @ np.linalg.inv(d16cov) @ (d16[:, 1] - datad16)

    cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
    cl = np.interp(l_real, np.arange(len(cl)), cl)

    cl *= (2*np.pi)/(l_real*(l_real+1)) # convert to C_l
    
    cl += noise

    #L = (-1/2*(2*l_real + 1)*(np.log(cl) + p/cl - (2*l_real-1)/(2*l_real + 1)*np.log(p))).sum()

    x = (2*l_real + 1)* p/cl
    Lplanck = (chi2(2*l_real+1).logpdf(x)).sum()

    #print(Lplanck, Lbaod12, Lbaod16)

    return Lplanck+Lbaod12+Lbaod16, []
    
file = 'Planck_bao_chains_wide/'
RESUME = True
nDims=6

#run_poly(prior, joint_likelihood, file, RESUME=RESUME, nDims=6)
settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = file + '/'
#settings.nlive = 25
#settings.num_repeats = 2

output = pypolychord.run_polychord(joint_likelihood, nDims, 0, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

joint = read_chains('Planck_bao_chains_wide/test')
planck = read_chains('Planck_chains_wide/test')
bao = read_chains('BAO_chains_wide/test')

R = joint.logZ(10000) - planck.logZ(10000) - bao.logZ(10000)
R = R.values
print(np.mean(R), np.std(R))
