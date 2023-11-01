import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import camb
from tqdm import tqdm

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

p, error, l_real = load_planck()


cls = []
for i in tqdm(range(200)):
    alm = hp.synalm(p)
    cl = hp.alm2cl(alm)
    
    if i == 0:
        label = 'Simulated'
    else:
        label = None

    plt.plot(l_real, cl, label=label, c='r', alpha=0.1)
    cls.append(cl)
plt.plot(l_real, p, label='Planck Data', marker='.', c='k')
plt.legend()
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.savefig('planck_mock_for_cov.png', dpi=300)
plt.show()

cls = np.array(cls)

cov = np.cov(cls.T)

np.savetxt('planck_mock_cov.txt', cov)

plt.imshow(cov,extent=[l_real[0],l_real[-1],l_real[-1],l_real[0]],cmap='inferno')
plt.colorbar()
plt.xlabel(r'$l$')
plt.ylabel(r'$l$')
plt.tight_layout()
plt.savefig('planck_mock_cov.png', dpi=300)
plt.show()