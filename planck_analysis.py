import numpy as np
import matplotlib.pyplot as plt
import camb
from anesthetic import read_chains

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

samples = read_chains('Planck_chains_wide_test/test')

pars = camb.CAMBparams()

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
noise *= (l_real*(l_real+1)/(2*np.pi))

def signal(l, theta):
    pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                        tau=theta[3], cosmomc_theta=theta[2]/100,
                        theta_H0_range=[5, 1000])
    pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_background(pars) # computes evolution of background cosmology

    cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
    cl = np.interp(l_real, np.arange(len(cl)), cl)
    return cl + noise

from fgivenx import plot_contours, plot_lines
fig, axes = plt.subplots(1)
samples = samples.compress()

names = ['p' + str(i) for i in range(6)]
samples = samples[names].values
#cbar = plot_contours(bao, z, samples, axes)
plot_lines(signal, l_real, samples, axes, color='r')
plt.plot(l_real, p, c='k', label='Planck')
plt.plot(l_real, noise, c='b', label='noise')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig('planck_fit_test.png', dpi=300)
plt.show()