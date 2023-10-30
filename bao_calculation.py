import numpy as np
import matplotlib.pyplot as plt
import camb

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)
results = camb.get_background(pars) # computes evolution of background cosmology

z = np.array([0.38, 0.51, 0.698])

da = (1+z) * results.angular_diameter_distance(z)

dh = 3e5/results.hubble_parameter(z) # 1/Mpc
#rs = results.sound_horizon(z) # Mpc
#rs = 147.78 # Mpc
rs = results.get_derived_params()['rdrag'] # Mpc

plt.plot(z, da/rs, label='DA/rs')
plt.plot(z, dh/rs, label='DH/rs')

d12 = np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
d16 = np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])

d12dm = d12[::2]
d12dh = d12[1::2]
d16dm = d16[::2]
d16dh = d16[1::2]
plt.plot(d12dm[:,0], d12dm[:,1], 'o', label='DR12 DM', c='r')
plt.plot(d12dh[:,0], d12dh[:,1], 'o', label='DR12 DH', c='k')
plt.plot(d16dm[:,0], d16dm[:,1], 'o', label='DR16 DM', c='b')
plt.plot(d16dh[:,0], d16dh[:,1], 'o', label='DR16 DH', c='g')
plt.legend()
plt.xlabel(r'$z$')
plt.ylabel(r'$D/r_s$')

plt.plot()
plt.show()