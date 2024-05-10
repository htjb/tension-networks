import numpy as np
import matplotlib.pyplot as plt
from cmblike.data import get_data
from tqdm import tqdm
import tensorflow as tf
from tensionnet import wmapplanck
from scipy.stats import ecdf
import os
from tensionnet.utils import cosmopower_prior, plotting_preamble

plotting_preamble()

_, lwmap = get_data(base_dir='cosmology-data/').get_wmap()

nSamples = 50000
joint = wmapplanck.jointClGenCP(path='/Users/harrybevins/Documents/Software/cosmopower')

parameters, prior_mins, prior_maxs = cosmopower_prior()

BASE_DIR = 'clean-wmap-planck-02052024/'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
load_data = True

def prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(len(parameters))]).T

if load_data:
    planckExamples = np.load(BASE_DIR + 'planck-wmap-planck-examples-50000.npy')
    wmapExamples = np.load(BASE_DIR + 'planck-wmap-wmap-examples-50000.npy')
else:
    wd = np.loadtxt('cosmology-data/wmap_binned.txt')
    bins = np.array([wd[:, 1], wd[:, 2]]).T
    pe, we= [], []
    for i in tqdm(range(nSamples//100)):
        samples = prior(100)
        pobs, wobs, cltheory = joint(samples, lwmap, bins)
        # pobs  = (5, 45)
        pe.append(pobs)
        we.append(wobs)
    planckExamples = np.vstack(pe)
    wmapExamples = np.vstack(we)
    print(planckExamples.shape, wmapExamples.shape)
    np.save(BASE_DIR + 'planck-wmap-planck-examples-50000.npy', planckExamples)
    np.save(BASE_DIR + 'planck-wmap-wmap-examples-50000.npy', wmapExamples)

mask = lwmap > 124
lwmap = lwmap[mask]
planckExamples = planckExamples[:, mask]
wmapExamples = wmapExamples[:, mask]

from tensionnet.tensionnet import nre

from anesthetic import read_chains
chains = read_chains(BASE_DIR + 'wmap_planck_joint_fit_cp_cp_prior_l_above_124/test')
planck = read_chains(BASE_DIR + 'fit_wmap_binned_planck_cp_cp_prior_l_above_124/test')
wmap = read_chains(BASE_DIR + 'wmap_fit_cp_cp_prior_l_above_124/test')

Rsamples = chains.logZ(1000) - planck.logZ(1000) - wmap.logZ(1000)
Rs = Rsamples.mean()
errorRs = Rsamples.std()
print('R = {:.2f} +/- {:.2f}'.format(Rs, errorRs))

RETRAIN = True
if RETRAIN:
    import os
    if os.path.exists(BASE_DIR + 'wmap_planck_nre.pkl'):
        os.remove(BASE_DIR + 'wmap_planck_nre.pkl')

try:
    nrei = nre.load(BASE_DIR + 'wmap_planck_nre.pkl', shared_prior=prior,
                     simulation_func_A=None, simulation_func_B=None)
except FileNotFoundError:

    nrei = nre(lr=1e-4)
    nrei.build_model(len(lwmap) + len(lwmap), 
                    [25]*5, 'sigmoid')

    splitIdx = np.arange(len(wmapExamples))
    np.random.shuffle(splitIdx)
    trainIdx = splitIdx[:int(len(wmapExamples)*0.75)]
    validationIdx = splitIdx[int(len(wmapExamples)*0.75):int(len(wmapExamples)*0.8)]
    testIdx = splitIdx[int(len(wmapExamples)*0.8):]

    print('Splitting data...')
    trainWmap = wmapExamples[trainIdx]
    testWmap = wmapExamples[testIdx]
    validationWmap = wmapExamples[validationIdx]
    trainPlanck = planckExamples[trainIdx]
    testPlanck = planckExamples[testIdx]
    validationPlanck = planckExamples[validationIdx]

    print('Normalising data...')
    normtrainwmapExamples = (trainWmap -
                             np.mean(trainWmap, axis=0))/ \
                                np.std(trainWmap, axis=0)
    normtestwmapExamples = (testWmap -
                            np.mean(trainWmap, axis=0))/ \
                                np.std(trainWmap, axis=0)
    normtrainplanckExamples = (trainPlanck -
                               np.mean(trainPlanck, axis=0))/ \
                                np.std(trainPlanck, axis=0)
    normtestplanckExamples = (testPlanck -
                              np.mean(trainPlanck, axis=0))/ \
                                np.std(trainPlanck, axis=0)
    normvalidationPlanck = (validationPlanck -
                            np.mean(trainPlanck, axis=0))/ \
                                np.std(trainPlanck, axis=0)
    normvalidationWmap = (validationWmap -
                          np.mean(trainWmap, axis=0))/ \
                                np.std(trainWmap, axis=0)

    print('Shuffling and stacking training and test data...')
    matchedtrainData = np.hstack([normtrainwmapExamples, normtrainplanckExamples])
    matchedtrainLabels = np.ones(len(matchedtrainData))
    matchedtestData = np.hstack([normtestwmapExamples, normtestplanckExamples])
    matchedtestLabels = np.ones(len(matchedtestData))
    data_validation = np.hstack([normvalidationWmap, normvalidationPlanck])

    idx = np.arange(len(matchedtrainData))
    np.random.shuffle(idx)
    shuffledtrainPlanck = normtrainplanckExamples[idx]
    shuffledtrainData = np.hstack([normtrainwmapExamples, shuffledtrainPlanck])
    shuffledtrainLabels = np.zeros(len(shuffledtrainData))

    data_train = np.vstack([matchedtrainData, shuffledtrainData])
    labels_train = np.hstack([matchedtrainLabels, shuffledtrainLabels])

    idx = np.arange(len(matchedtestData))
    np.random.shuffle(idx)
    shuffledtestPlanck = normtestplanckExamples[idx]
    shuffledtestData = np.hstack([normtestwmapExamples, shuffledtestPlanck])
    shuffledtestLabels = np.zeros(len(shuffledtestData))

    data_test = np.vstack([matchedtestData, shuffledtestData])
    labels_test = np.hstack([matchedtestLabels, shuffledtestLabels])

    idx = np.arange(len(data_train))
    np.random.shuffle(idx)
    data_train = data_train[idx]
    labels_train = labels_train[idx]

    idx = np.arange(len(data_test))
    np.random.shuffle(idx)
    data_test = data_test[idx]
    labels_test = labels_test[idx]
    #########

    nrei.data_test = data_test
    nrei.labels_test = labels_test
    nrei.data_train = data_train
    nrei.labels_train = labels_train
    nrei.prior_function_A = None
    nrei.prior_function_B = None
    nrei.shared_prior = prior

    nrei.simulation_func_A = None
    nrei.simulation_func_B = None

    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=1000)
    nrei.save(BASE_DIR + 'wmap_planck_nre.pkl')

plt.plot(nrei.loss_history, label='Training Loss')
plt.plot(nrei.test_loss_history, label='Validation Loss')
plt.legend()
plt.show()

nrei.__call__(iters=data_validation)
r = nrei.r_values
print(len(data_validation))
print(len(r))
mask = np.isfinite(r)
print(len(r[mask]))

fig, axes = plt.subplots(1, 2, figsize=(6.3, 4))
axes[0].hist(r[mask], bins=25, density=True)
axes[0].set_xlabel(r'$\log R$')
axes[0].set_ylabel('Density')

rsort  = np.sort(r[mask])
c = ecdf(rsort)

axes[1].plot(rsort, c.cdf.evaluate(rsort)) 
axes[1].axhline(c.cdf.evaluate(Rs), ls='--',
        color='r')
axes[1].axhspan(c.cdf.evaluate(Rs - errorRs), 
        c.cdf.evaluate(Rs + errorRs), 
        alpha=0.1, 
        color='r')
axes[1].set_xlabel(r'$\log R$')
axes[1].set_ylabel(r'$P(\log R < \log R^\prime)$')

axes[0].axvline(Rs, ls='--', c='r')
axes[0].axvspan(Rs - errorRs, Rs + errorRs, alpha=0.1, color='r')

axes[1].axhline(c.cdf.evaluate(Rs), ls='--',
            color='r')
axes[1].axhspan(c.cdf.evaluate(Rs - errorRs),
            c.cdf.evaluate(Rs + errorRs),
            alpha=0.1,
            color='r')

plt.tight_layout()
plt.savefig(BASE_DIR + 'wmap_planck.pdf', bbox_inches='tight')
plt.show()
