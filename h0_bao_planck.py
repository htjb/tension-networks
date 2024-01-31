import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from anesthetic import MCMCSamples
from tqdm import tqdm
import camb

pars = camb.CAMBparams()

def derived(parameters):
    H0, rs, omm = [], [], []
    for i in tqdm(range(len(parameters))):
        pars.set_cosmology(ombh2=parameters[i][0], omch2=parameters[i][1],
                                    tau=parameters[i][3], cosmomc_theta=parameters[i][2]/100,
                                    theta_H0_range=[5, 1000])
        pars.InitPower.set_params(As=np.exp(parameters[i][5])/10**10, ns=parameters[i][4])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        H0.append(results.hubble_parameter(0))
        rs.append(results.get_derived_params()['rdrag']) # Mpc
        
        h = H0[-1]/100
        omb = parameters[i][0]/h**2
        omc = parameters[i][1]/h**2
        omm.append((omb+omc))

    H0 = np.array(H0)
    rs = np.array(rs)
    data = np.array(H0*rs)
    data /= 3e5
    data = np.vstack((data, omm, H0)).T

    samples = MCMCSamples(data=data, labels=[r'$\frac{H_0 r_s}{c}$', r'$\Omega_m$', r'$H_0$'])
    return samples

bao = read_chains('bao_fit/test')
bao = bao.compress(1000)
parameters = bao.values[:, :6]
bao_samples = derived(parameters)

axis = bao_samples.plot_2d([0, 1, 2])

planck = read_chains('planck_fit/test')
planck = planck.compress(1000)
parameters = planck.values[:, :6]
planck_samples = derived(parameters)
planck_samples.plot_2d(axis)

joint = read_chains('Planck_bao_fit/test')
joint = joint.compress(1000)
parameters = joint.values[:, :6]
joint_samples = derived(parameters)
joint_samples.plot_2d(axis)

plt.savefig('h0_bao.pdf', bbox_inches='tight')
plt.show()
