import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cmbemu.eval import evaluate
import camb

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
power_cov = np.loadtxt('planck_mock_cov.txt')
inv_cov = np.linalg.inv(power_cov)
predictor = evaluate(base_dir='cmbemu_model/', l=l_real)

pars = camb.CAMBparams()
pars.set_for_lmax(2500, lens_potential_accuracy=0)
z = np.array([0.38, 0.51, 0.698])

def cl_func_gen():
    def power_spec(_, parameters):
        cl, _ = predictor(parameters)
        return cl
    return power_spec

def bao_func():
    def bao(_, parameters):
        pars.set_cosmology(ombh2=parameters[0], omch2=parameters[1],
                            tau=parameters[3], cosmomc_theta=parameters[2]/100)
        pars.InitPower.set_params(As=np.exp(parameters[5])/10**10, ns=parameters[4])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        da = (1+z) * results.angular_diameter_distance(z)
        dh = 3e5/results.hubble_parameter(z) # 1/Mpc
        rs = results.get_derived_params()['rdrag'] # Mpc

        datad12 = [da[0]/rs, dh[0]/rs, da[1]/rs, dh[1]/rs]
        datad16 = [da[2]/rs, dh[2]/rs]
        return np.concatenate((datad12, datad16))
    return bao

def signal_prior(n):
    parameters = np.ones((n, 6))
    parameters[:, 0] = np.random.uniform(0.0211, 0.0235) # omegabh2
    parameters[:, 1] = np.random.uniform(0.108, 0.131) # omegach2
    parameters[:, 2] = np.random.uniform(1.038, 1.044) # 100*thetaMC
    parameters[:, 3] = np.random.uniform(0.01, 0.16) # tau
    parameters[:, 4] = np.random.uniform(0.938, 1) # ns
    parameters[:, 5] = np.random.uniform(2.95, 3.25) # log(10^10*As)
    return parameters

def exp_prior(n):
    """
    The way tensionnet is set up it requires some
    parameters that are unique to each experiment. Here I give an array of
    zeros because the experimetns are just signal plus noise. Doesn't have
    any impact on the results.
    """
    return np.zeros((n, 2))

planck_func = cl_func_gen()
bao_func = bao_func()

from tensionnet.tensionnet import nre

try:
    nrei = nre.load('bao_planck_model.pkl',
                planck_func, bao_func, exp_prior,
                exp_prior, signal_prior)
except:
    nrei = nre(lr=1e-4)
    nrei.build_model(len(l_real) + len(z)*2, 1, 
                        [100]*10, 'sigmoid')

    #nrei.build_compress_model(len(exp2_freq), len(exp1_freq), 1, 
    #                       [len(exp2_freq), len(exp2_freq), len(exp2_freq)//2, 50, 10], 
    #                       [len(exp1_freq), len(exp1_freq), len(exp1_freq)//2, 50, 10], 
    #                       [10, 10, 10, 10, 10],
    #                       'sigmoid')
    nrei.build_simulations(planck_func, bao_func, exp_prior, exp_prior, signal_prior, n=50000)
    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=2000)
    nrei.save('bao_planck_model.pkl')
"""
plt.plot(nrei.loss_history)
plt.plot(nrei.test_loss_history)
plt.yscale('log')
plt.show()"""

nrei.__call__(iters=2000)
r = nrei.r_values
mask = np.isfinite(r)
sigr = tf.keras.layers.Activation('sigmoid')(r[mask])
c = 0
for i in range(len(sigr)):
    if sigr[i] < 0.75:
        c += 1

Rs = [-0.055]
sigma_Rs = [0.283]


fig, axes = plt.subplots(1, 2, figsize=(6.3, 3))
axes[0].hist(r[mask], bins=25, label=f'{c/len(sigr)*100:.2f} % Mis-classified', color='C1')
axes[0].set_yticks([])
axes[0].axvline(Rs[0], ls='--', c='r')
axes[0].axvspan(Rs[0] - sigma_Rs[0], Rs[0] + sigma_Rs[0], alpha=0.1, color='r')
axes[0].set_xlabel(r'$\log R$')
axes[0].set_ylabel('Frequency')
axes[0].legend()


"""from anesthetic import MCMCSamples
samples = MCMCSamples(data=r[mask], columns=['R'])
axes = samples.plot_1d('R', fc='C1', ec='k')
for i,t in enumerate(temperatures):
    plt.axvline(Rs[i], ls='--', label= f'{round(t/0.2, 2)}', color=plt.get_cmap('jet')(i/len(temperatures)))
    plt.axvspan(Rs[i] - sigma_Rs[i], Rs[i] + sigma_Rs[i], alpha=0.1, color=plt.get_cmap('jet')(i/len(temperatures)))
plt.xlim([-5, 25])
plt.xlabel(r'$\log R$')
plt.ylabel('Frequency')
plt.tight_layout()
plt.legend()
plt.savefig('test_r_kde.png', dpi=300)
plt.show()"""

from scipy.stats import ecdf

r  = np.sort(r[mask])
c = ecdf(r)

axes[1].plot(r, c.cdf.evaluate(r)) 
axes[1].axhline(c.cdf.evaluate(Rs[0]), ls='--',
        color='r')
axes[1].axhspan(c.cdf.evaluate(Rs[0] - sigma_Rs[0]), 
        c.cdf.evaluate(Rs[0] + sigma_Rs[0]), 
        alpha=0.1, 
        color='r')
axes[1].set_xlabel(r'$\log R$')
axes[1].set_ylabel(r'$P(\log R < \log R_{obs})$')
plt.tight_layout()
plt.savefig('bao_planck_r_cdf_hist.png', dpi=300)
plt.show()

idx = [int(np.random.uniform(0, len(nrei.labels_test), 1)) for i in range(1000)]

labels_test = nrei.labels_test[idx]

nrei.__call__(iters=nrei.data_test[idx])
p = tf.keras.layers.Activation('sigmoid')(nrei.r_values)
"""plt.hist(p, bins=25)
plt.show()"""

correct1, correct0, wrong1, wrong0, confused1, confused0 = 0, 0, 0, 0, 0, 0
for i in range(len(p)):
    if p[i] > 0.75 and labels_test[i] == 1:
        correct1 += 1
    elif p[i] < 0.25 and labels_test[i] == 0:
        correct0 += 1
    elif p[i] > 0.75 and labels_test[i] == 0:
        wrong0 += 1
    elif p[i] < 0.25 and labels_test[i] == 1:
        wrong1 += 1
    elif p[i] > 0.25 and r[i] < 0.75 and labels_test[i] == 1:
        confused1 += 1
    elif p[i] > 0.25 and r[i] < 0.75 and labels_test[i] == 0:
        confused0 += 1

cm = [[correct0, wrong0, confused0],
        [correct1, wrong1, confused1]]

fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(3):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='k',
                 bbox=dict(facecolor='white', lw=0))
plt.xticks([0, 1, 2], ['Correct', 'Wrong', 'Confused'])
plt.yticks([0, 1], ['In tension', 'Not In Tension'])
plt.tight_layout()
plt.savefig('bao_planck_confusion_matrix.png', dpi=300)
plt.show()

