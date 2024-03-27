import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
import camb
import healpy as hp
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise
from tqdm import tqdm
from random import shuffle
import matplotlib as mpl
from matplotlib import rc
import tensorflow as tf
from tensionnet import wmapplanck
from scipy.stats import ecdf

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

nSamples = 50000
joint = wmapplanck.jointClGenCP(path='/Users/harrybevins/Documents/Software/cosmopower')

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

BASE_DIR = 'cosmopower-stuff/'
load_data = False

def prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(len(parameters))]).T

def exp_prior(n):
    """
    The way tensionnet is set up it requires some
    parameters that are unique to each experiment. Here I give an array of
    zeros because the experimetns are just signal plus noise. Doesn't have
    any impact on the results.
    """
    return np.zeros((n, 2))

if load_data:
    planckExamples = np.load(BASE_DIR + 'planckExamplesCPRNRE.npy')
    wmapExamples = np.load(BASE_DIR + 'wmapExamplesCPRNRE.npy')
    samples = np.load(BASE_DIR + 'samplesCPRNRE.npy')
else:
    pe, we, samps = [], [], []
    for i in range(nSamples//100):
        print(str(i) + '/' + str(nSamples//100))
        samples = prior(100)
        planckExamples, wmapExamples = joint(samples)
        pe.append(planckExamples)
        we.append(wmapExamples)
        samps.append(samples)
    planckExamples = np.vstack(pe)
    wmapExamples = np.vstack(we)
    samples = np.vstack(samps)
    np.save(BASE_DIR + 'planckExamplesCPRNRE.npy', planckExamples)
    np.save(BASE_DIR + 'wmapExamplesCPRNRE.npy', wmapExamples)
    np.save(BASE_DIR + 'samplesCPRNRE.npy', samples)

sys.exit(1)
from tensionnet.tensionnet import nre

layers = [200]*4

from anesthetic import read_chains
chains = read_chains('wmap_planck_fit/test')
planck = read_chains('Planck_fit/test')
wmap = read_chains('wmap_fit/test')

Rsamples = chains.logZ(1000) - planck.logZ(1000) - wmap.logZ(1000)
Rs = np.mean(Rsamples)
errorRs = np.std(Rsamples)

nrei = nre(lr=1e-4)
nrei.build_model(len(l) + len(lwmap), 1, 
                    layers, 'sigmoid')
try:
    nrei = nrei.load(BASE_DIR + 'wmap_planck_nre.pkl', prior_function_A = exp_prior,
                     prior_function_B=exp_prior, shared_prior=prior,
                     simulation_func_A=None, simulation_func_B=None)
except FileNotFoundError:

    splitIdx = np.arange(len(wmapExamples))
    np.random.shuffle(splitIdx)
    trainIdx = splitIdx[:int(len(wmapExamples)*0.8)]
    testIdx = splitIdx[int(len(wmapExamples)*0.8):]

    print('Splitting data...')
    trainWmap = wmapExamples[trainIdx]
    testWmap = wmapExamples[testIdx]
    trainPlanck = planckExamples[trainIdx]
    testPlanck = planckExamples[testIdx]
    #trainParams = samples[trainIdx]
    #testParams = samples[testIdx]

    np.savetxt('cosmopower-stuff/train_wmap_RNRE.txt', trainWmap)
    np.savetxt('cosmopower-stuff/test_wmap_RNRE.txt', testWmap)
    np.savetxt('cosmopower-stuff/train_planck_RNRE.txt', trainPlanck)
    np.savetxt('cosmopower-stuff/test_planck_RNRE.txt', testPlanck)
    #np.savetxt('cosmopower-stuff/train_params_RNRE.txt', trainParams)
    #np.savetxt('cosmopower-stuff/test_params_RNRE.txt', testParams)
    np.savetxt('cosmopower-stuff/train_wmap_mean_RNRE.txt', np.mean(trainWmap, axis=0))
    np.savetxt('cosmopower-stuff/train_wmap_std_RNRE.txt', np.std(trainWmap, axis=0))
    np.savetxt('cosmopower-stuff/train_planck_mean_RNRE.txt', np.mean(trainPlanck, axis=0))
    np.savetxt('cosmopower-stuff/train_planck_std_RNRE.txt', np.std(trainPlanck, axis=0))
    #np.savetxt('cosmopower-stuff/train_params_mean_RNRE.txt', np.mean(trainParams, axis=0))
    #np.savetxt('cosmopower-stuff/train_params_std_RNRE.txt', np.std(trainParams, axis=0))

    print('Normalising data...')
    normtrainwmapExamples = (trainWmap -np.mean(trainWmap, axis=0))/np.std(trainWmap, axis=0)
    normtestwmapExamples = (testWmap -np.mean(trainWmap, axis=0))/np.std(trainWmap, axis=0)
    normtrainplanckExamples = (trainPlanck -np.mean(trainPlanck, axis=0))/np.std(trainPlanck, axis=0)
    normtestplanckExamples = (testPlanck -np.mean(trainPlanck, axis=0))/np.std(trainPlanck, axis=0)
    #normtrainParams = (trainParams -np.mean(trainParams, axis=0))/np.std(trainParams, axis=0)
    #normtestParams = (testParams -np.mean(trainParams, axis=0))/np.std(trainParams, axis=0)

    print('Shuffling and stacking training and test data...')
    matchedtrainData = np.hstack([normtrainwmapExamples, normtrainplanckExamples])
    matchedtrainLabels = np.ones(len(matchedtrainData))
    matchedtestData = np.hstack([normtestwmapExamples, normtestplanckExamples])
    matchedtestLabels = np.ones(len(matchedtestData))

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
    nrei.prior_function_A = exp_prior
    nrei.prior_function_B = exp_prior
    nrei.shared_prior = prior

    nrei.simulation_func_A = None
    nrei.simulation_func_B = None

    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=2000)
    nrei.save(BASE_DIR + 'wmap_planck_nre.pkl')

test_data =[]
for i in tqdm(range(500)):
    theta = wide_prior(np.random.rand(6))
    pl, w = joint(None, theta)
    test_data.append(np.concatenate((pl, w)))
test_data = np.array(test_data)

nrei.__call__(iters=test_data)
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
#axes[0, 0].axvline(Rs, ls='--', c='r')
axes[0, 0].set_title('No. Sig. ' + r'$=$ ' + str(len(r[mask])))
#axes[0, 0].axvspan(Rs - errorRs, Rs + errorRs, alpha=0.1, color='r')
axes[0, 0].set_xlabel(r'$\log R$')
axes[0, 0].set_ylabel('Density')

rsort  = np.sort(r[mask])
c = ecdf(rsort)

axes[0, 1].plot(rsort, c.cdf.evaluate(rsort)) 
"""axes[0, 1].axhline(c.cdf.evaluate(Rs), ls='--',
        color='r')
axes[0, 1].axhspan(c.cdf.evaluate(Rs - errorRs), 
        c.cdf.evaluate(Rs + errorRs), 
        alpha=0.1, 
        color='r')"""
axes[0, 1].set_xlabel(r'$\log R$')
axes[0, 1].set_ylabel(r'$P(\log R < \log R_{obs})$')
"""axes[0, 1].set_title(r'$P=$' + str(np.round(c.cdf.evaluate(Rs), 3)) +
                r'$+$' + str(np.round(c.cdf.evaluate(Rs + errorRs) - c.cdf.evaluate(Rs), 3)) +
                r'$(-$' + str(np.round(c.cdf.evaluate(Rs) - c.cdf.evaluate(Rs - errorRs),3)) + r'$)$')
"""

idx = [int(np.random.uniform(0, len(nrei.labels_test), 1)) for i in range(1000)]
labels_test = nrei.labels_test[idx]
nrei.__call__(iters=nrei.data_test[idx])
p = tf.keras.layers.Activation('sigmoid')(nrei.r_values)

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

"""nrei.__call__(iters=np.array([np.concatenate([praw, wmapraw])]))
r = nrei.r_values"""

axes[0, 0].axvline(Rs, ls='--', c='r')
axes[0, 0].axvspan(Rs - errorRs, Rs + errorRs, alpha=0.1, color='r')

axes[0, 1].axhline(c.cdf.evaluate(Rs), ls='--',
            color='r')
axes[0, 1].axhspan(c.cdf.evaluate(Rs - errorRs),
            c.cdf.evaluate(Rs + errorRs),
            alpha=0.1,
            color='r')

plt.tight_layout()
plt.savefig('wmap_planck.pdf', bbox_inches='tight')
plt.show()

