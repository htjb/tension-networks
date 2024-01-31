import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import camb
import matplotlib as mpl
from matplotlib import rc
from scipy.stats import ecdf
from cmblike.data import get_data
from cmblike.noise import planck_noise
from cmblike.cmb import CMB

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')


p, l = get_data(base_dir='cosmology-data/').get_planck()
planck_noise = planck_noise(l).calculate_noise()

from tensionnet.bao import BAO
baos = BAO(data_location='cosmology-data/bao_data/')
z = baos.z

cmbs = CMB()

def cl_func_gen():
    def cl_func(_, parameters):
        cl, sample = cmbs.get_samples(l, parameters, noise=planck_noise)
        return sample
    return cl_func

def bao_func():
    def bao(_, parameters):
        datad12, datad16 = baos.get_camb_model(parameters)
        return np.concatenate((datad12, datad16))
    return bao

def signal_prior(n):
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

nsamples = 100000
layers = [200]*4
Rs, errorRs = 3.428, 0.172

try:
    nrei = nre.load('bao_planck_model.pkl',
                planck_func, bao_func, exp_prior,
                exp_prior, signal_prior)
except:
    nrei = nre(lr=1e-4)
    nrei.build_model(len(l) + len(z)*2, 1, 
                        layers, 'sigmoid')
    try:
        wide_data = np.loadtxt('planck_bao_data.txt')
        wide_labels = np.loadtxt('planck_bao_labels.txt')
        nrei.data = wide_data
        nrei.labels = wide_labels
        nrei.simulation_func_A = planck_func
        nrei.simulation_func_B = bao_func
        nrei.prior_function_A = exp_prior
        nrei.prior_function_B = exp_prior
        nrei.shared_prior = signal_prior
    except:
        nrei.build_simulations(planck_func, bao_func, 
                            exp_prior, exp_prior, signal_prior, n=nsamples)
        np.savetxt('planck_bao_data.txt', nrei.data)
        np.savetxt('planck_bao_labels.txt', nrei.labels)
    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=2000)
    nrei.save('bao_planck_model.pkl')

nrei.__call__(iters=2000)
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
acc = c/len(sigr)*100

fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.3))
axes[0, 0].hist(r[mask], bins=25,density=True)
axes[0, 0].axvline(Rs, ls='--', c='r')
axes[0, 0].set_title('No. Sig. ' + r'$=$ ' + str(len(r[mask])) + '\n' +
                         r'$R_{obs}=$' + str(np.round(Rs, 2)) + r'$\pm$' +
                            str(np.round(errorRs, 2)))
axes[0, 0].axvspan(Rs - errorRs, Rs + errorRs, alpha=0.1, color='r')
axes[0, 0].set_xlabel(r'$\log R$')
axes[0, 0].set_ylabel('Density')

rsort  = np.sort(r[mask])
c = ecdf(rsort)

axes[0, 1].plot(rsort, c.cdf.evaluate(rsort)) 
axes[0, 1].axhline(c.cdf.evaluate(Rs), ls='--',
        color='r')
axes[0, 1].axhspan(c.cdf.evaluate(Rs - errorRs), 
        c.cdf.evaluate(Rs + errorRs), 
        alpha=0.1, 
        color='r')
axes[0, 1].set_xlabel(r'$\log R$')
axes[0, 1].set_ylabel(r'$P(\log R < \log R_{obs})$')
axes[0, 1].set_title(r'$P=$' + str(np.round(c.cdf.evaluate(Rs), 3)) +
                r'$+$' + str(np.round(c.cdf.evaluate(Rs + errorRs) - c.cdf.evaluate(Rs), 3)) +
                r'$(-$' + str(np.round(c.cdf.evaluate(Rs) - c.cdf.evaluate(Rs - errorRs),3)) + r'$)$')


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

axes[1,0].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(3):
        axes[1, 0].text(j, i, '{:.2f} \%'.format(cm[i][j]), ha='center', va='center', color='k',
                bbox=dict(facecolor='white', lw=0), fontsize=10)
axes[1, 0].set_xticks([0, 1, 2], ['Correct', 'Wrong', 'Confused'])
axes[1, 0].set_yticks([0, 1], ['In tension', 'Not In Tension'])

axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig('bao_planck.pdf', bbox_inches='tight')
plt.show()

