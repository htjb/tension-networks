import numpy as np
from anesthetic import read_chains
import matplotlib.pyplot as plt
import camb

pars = camb.CAMBparams()

z = np.array([0.38, 0.51, 0.698])

d12 = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
d16 = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])
d12cov = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH_covtot.txt')
d16cov = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH_covtot.txt')

def bao(z, theta):
    try:
        pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=theta[3], cosmomc_theta=theta[2]/100,
                            theta_H0_range=(10, 100))
        pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        da = (1+z) * results.angular_diameter_distance(z)
        dh = 3e5/results.hubble_parameter(z) # 1/Mpc
        rs = results.get_derived_params()['rdrag'] # Mpc

        return da/rs#, dh/rs
    except:
        return [0, 0, 0]       

samples = read_chains('BAO_chains/test')
names = ['p'+str(i) for i in range(6)]

samples.plot_2d(['p'+str(i) for i in range(6)])
plt.show()

#[likelihood(samples[names].values[i]) for i in range(len(samples))]
from fgivenx import plot_contours
fig, axes = plt.subplots(1)
samples = samples.compress()
cbar = plot_contours(bao, z, samples, axes)

plt.plot(d12[::2,0], d12[::2,1], 'o', label='DR12 DM', c='b')
#plt.plot(d12[1::2,0], d12[1::2,1], 'o', label='DR12 DH', c='k')
plt.plot(d16[::2,0], d16[::2,1], 'o', label='DR16 DM', c='b')
#plt.plot(d16[1::2,0], d16[1::2,1], 'o', label='DR16 DH', c='g')
plt.show()

