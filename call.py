import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cmbemu.eval import evaluate
import camb
import matplotlib as mpl
from matplotlib import rc
from scipy.stats import ecdf

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')


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
                            tau=parameters[3], cosmomc_theta=parameters[2]/100,
                            theta_H0_range=[5, 1000])
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
    parameters[:, 0] = np.random.uniform(0.0211, 0.0235, n) # omegabh2
    parameters[:, 1] = np.random.uniform(0.108, 0.131, n) # omegach2
    parameters[:, 2] = np.random.uniform(1.038, 1.044, n) # 100*thetaMC
    parameters[:, 3] = np.random.uniform(0.01, 0.16, n) # tau
    parameters[:, 4] = np.random.uniform(0.938, 1, n) # ns
    parameters[:, 5] = np.random.uniform(2.95, 3.25, n) # log(10^10*As)
    return parameters

def signal_wide_prior(n):
    theta = np.ones((n, 6))
    theta[:, 0] = np.random.uniform(0.01, 0.085, n) # omegabh2
    theta[:, 1] = np.random.uniform(0.08, 0.21, n) # omegach2
    theta[:, 2] = np.random.uniform(0.97, 1.5, n) # 100*thetaMC
    theta[:, 3] = np.random.uniform(0.01, 0.16, n) # tau
    theta[:, 4] = np.random.uniform(0.8, 1.2, n) # ns
    theta[:, 5] = np.random.uniform(2.6, 3.8, n) # log(10^10*As)
    return theta

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

priors = [signal_prior, signal_wide_prior]
nsamples = [50000, 50000]
layers = [[200]*4, [200]*4]
prior_strings = ['Narrow', 'Wide']
file_strings = ['', '_wide']
Rs = [-0.055, 3.032]
sigma_Rs = [0.283, 0.470]
nreis = []
for j in range(len(priors)):
    predictor = evaluate(base_dir='cmbemu_model' + file_strings[j] + '/', l=l_real)
    try:
        nrei = nre.load('bao_planck_model' + file_strings[j] + '.pkl',
                    planck_func, bao_func, exp_prior,
                    exp_prior, signal_prior)
    except:
        nrei = nre(lr=1e-4)
        nrei.build_model(len(l_real) + len(z)*2, 1, 
                            layers[j], 'sigmoid')
        try:
            wide_data = np.loadtxt('planck_bao' + file_strings[j] + '_data.txt')
            wide_labels = np.loadtxt('planck_bao' + file_strings[j] + '_labels.txt')
            nrei.data = wide_data
            nrei.labels = wide_labels
            nrei.simulation_func_A = planck_func
            nrei.simulation_func_B = bao_func
            nrei.prior_function_A = exp_prior
            nrei.prior_function_B = exp_prior
            nrei.shared_prior = priors[j]
        except:
            nrei.build_simulations(planck_func, bao_func, 
                               exp_prior, exp_prior, priors[j], n=nsamples[j])
            np.savetxt('planck_bao' + file_strings[j] + '_data.txt', nrei.data)
            np.savetxt('planck_bao' + file_strings[j] + '_wide_labels.txt', nrei.data)
        model, data_test, labels_test = nrei.training(epochs=1000, batch_size=2000)
        nrei.save('bao_planck_model' + file_strings[j] + '.pkl')
    nreis.append(nrei)

rs, acc = [], []
for i, nrei in enumerate(nreis):
    nrei.__call__(iters=1000)
    r = nrei.r_values
    mask = np.isfinite(r)
    sigr = tf.keras.layers.Activation('sigmoid')(r[mask])
    c = 0
    good_idx = []
    for i in range(len(sigr)):
        if sigr[i] < 0.75:
            c += 1
        else:
            good_idx.append(i)

    r = r[good_idx]
    mask = np.isfinite(r)
    rs.append(r[mask])
    acc.append(c/len(sigr)*100)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3))
for i in range(len(rs)):
    axes[0].hist(rs[i], bins=25, label=prior_strings[i] + f': {acc[i]:.2f} \% Mis-classified', 
                 color='C' + str(i), histtype='step', density=True)
    axes[0].set_yticks([])
    axes[0].axvline(Rs[i], ls='--', c='C' + str(i))
    axes[0].axvspan(Rs[i] - sigma_Rs[i], Rs[i] + sigma_Rs[i], 
                    alpha=0.1, color='C' + str(i))
    axes[0].set_xlabel(r'$\log R$')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    rsort  = np.sort(rs[i])
    c = ecdf(rsort)

    axes[1].plot(rsort, c.cdf.evaluate(rsort)) 
    axes[1].axhline(c.cdf.evaluate(Rs[i]), ls='--',
            color='C' + str(i))
    axes[1].axhspan(c.cdf.evaluate(Rs[i] - sigma_Rs[i]), 
            c.cdf.evaluate(Rs[i] + sigma_Rs[i]), 
            alpha=0.1, 
            color='C' + str(i))
    axes[1].set_xlabel(r'$\log R$')
    axes[1].set_ylabel(r'$P(\log R < \log R_{obs})$')
plt.tight_layout()
plt.savefig('bao_planck_r_cdf_hist_multi_prior.png', dpi=300)
plt.show()

for k, nrei in enumerate(nreis):
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
        elif p[i] > 0.25 and p[i] < 0.75 and labels_test[i] == 1:
            confused1 += 1
        elif p[i] > 0.25 and p[i] < 0.75 and labels_test[i] == 0:
            confused0 += 1

    total_0 = len(labels_test[labels_test == 0])
    total_1 = len(labels_test[labels_test == 1])

    cm = [[correct0/total_0*100, wrong0/total_0*100, confused0/total_0*100],
            [correct1/total_1*100, wrong1/total_1*100, confused1/total_1*100]]

    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(3):
            plt.text(j, i, '{:.3f} \%'.format(cm[i][j]), ha='center', va='center', color='k',
                    bbox=dict(facecolor='white', lw=0))
    plt.xticks([0, 1, 2], ['Correct', 'Wrong', 'Confused'])
    plt.yticks([0, 1], ['In tension', 'Not In Tension'])
    plt.tight_layout()
    plt.savefig('bao_planck_confusion_matrix' + file_strings[i] + '.png', dpi=300)
    plt.show()

