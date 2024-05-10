import numpy as np
import matplotlib.pyplot as plt
import pypolychord
import cosmopower as cp
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings
from tensionnet.wmapplanck import jointClGenCP
from cmblike.noise import planck_noise, wmap_noise
from tensionnet.utils import rebin, cosmopower_prior, narrow_cosmopower_prior
from cmblike.cmb import CMB
from tensionnet.wmapplanck import loglikelihood as jointlikelihood
import os
from anesthetic import read_chains
from tqdm import tqdm
import tensorflow as tf
from scipy.stats import ecdf
import os
from tensionnet.utils import plotting_preamble
from tensionnet.tensionnet import nre
import yaml

plotting_preamble()

##########################################################################
############################ Constants ###################################
##########################################################################
print('Setting Constants...')
config = yaml.load(open('mock-cmb.yaml', 'r'), Loader=yaml.FullLoader)

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

ndims = config['polychord']['ndims']
nderived = config['polychord']['nderived']
RESUME = config['polychord']['resume']
RETRAIN = config['nre']['retrain']
BASE_DIR = config['base_dir']

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


##########################################################################
#################### Define the generator function #######################
################### for theoretical CMB power spectra ####################
##########################################################################

print('Defining generator function...')
path = config['cosmopower_path']
cp_nn = cp.cosmopower_NN(restore=True, 
                restore_filename= path \
                +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

def gen(parameters, lobs, bins):
    
    if type(parameters) == list:
            parameters = np.array(parameters)
        
    if parameters.ndim < 2:
        parameters = np.array([parameters])
    
    params = {'omega_b': parameters[:, 0],
        'omega_cdm': parameters[:, 1],
        'h': parameters[:, 4],
        'n_s': parameters[:, 2],
        'tau_reio': [0.055]*len(parameters[:, 0]),
        'ln10^{10}A_s': parameters[:, 3],
        }
    
    cl = cp_nn.ten_to_predictions_np(params)[0]*1e12*2.7255**2
    lgen = cp_nn.modes

    return rebin(cl*lgen*(lgen+1)/(2*np.pi), bins)*2*np.pi/(lobs*(lobs+1))

##########################################################################
#################### Generate pretend data ###############################
##########################################################################

print('Generating/loading data...')
generator = jointClGenCP(config['cosmopower_path'])
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]
pnoise = planck_noise(lwmap).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

if config['polychord']['resume']:
    pobs = np.load(BASE_DIR + 'pobs.npy')
    wobs = np.load(BASE_DIR + 'wobs.npy')
else:
    if config['mock_data']['params']:
        samples = config['mock_data']['params']
        pobs, wobs, cltheory = generator(samples, lwmap, bins)
    else:
        lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt(
            'cosmology-data/wmap_unbinned.txt', unpack=True)
        lplanck, signal_planck, _, _ = np.loadtxt(
            'cosmology-data/planck_unbinned.txt', unpack=True)

        pobs = rebin(signal_planck, bins)*2*np.pi/(lwmap*(lwmap+1))
        wobs = rebin(wmap_unbinned, bins)*2*np.pi/(lwmap*(lwmap+1))

    np.save(BASE_DIR + 'pobs.npy', pobs)
    np.save(BASE_DIR + 'wobs.npy', wobs)


lcut = config['lcut']
if lcut:
    mask = lwmap > lcut
    lwmap = lwmap[mask]
    bins = bins[mask]
    pobs = pobs[mask]
    wobs = wobs[mask]
    pnoise = pnoise[mask]
    wnoise = wnoise[mask]

############################################################################
######################## Define likelihoods ################################
############################################################################
print('Defining likelihoods...')
if config['prior']['name'] == 'cosmopower_prior':
    parameters, prior_mins, prior_maxs = cosmopower_prior()
elif config['prior']['name'] == 'narrow_cosmopower_prior':
    parameters, prior_mins, prior_maxs = narrow_cosmopower_prior()

cmbs = CMB(parameters=parameters, prior_mins=prior_mins,
		           prior_maxs=prior_maxs,
                   path_to_cp=config['cosmopower_path'])

wmaplikelihood = cmbs.get_likelihood(wobs, lwmap,
                                noise=wnoise, cp=True, bins=bins)
prior = cmbs.prior

plancklikelihood = cmbs.get_likelihood(pobs, lwmap,
                                noise=pnoise, cp=True, bins=bins)

if config['joint_likelihood']['flag']:
    print('Flagged likelihood not implemented yet.')
    exit()

def joint_likelihood(theta):
    cltheory = gen(theta, lwmap, bins)
    return jointlikelihood(pobs + pnoise, 
                        wobs + wnoise, 
                        cltheory, pnoise, wnoise, 
                        lwmap)[0], []

##########################################################################
############################## WMAP fit ##################################
##########################################################################
print('Running WMAP Fit...')
file = 'WMAP'

settings = PolyChordSettings(ndims, nderived) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file + '/'

output = pypolychord.run_polychord(wmaplikelihood, ndims, 
                                   nderived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(ndims)]
output.make_paramnames_files(paramnames)

##########################################################################
############################## Planck fit ##################################
##########################################################################
print('Running Planck Fit...')
file = 'Planck'

settings = PolyChordSettings(ndims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file + '/'

output = pypolychord.run_polychord(plancklikelihood, ndims, 
                                   nderived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(ndims)]
output.make_paramnames_files(paramnames)


##########################################################################
############################## joint fit ##################################
##########################################################################
print('Running Joint Fit...')
file = 'Joint'

settings = PolyChordSettings(ndims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file + '/'

output = pypolychord.run_polychord(joint_likelihood, ndims, 
                                   nderived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(ndims)]
output.make_paramnames_files(paramnames)

#########################################################################
############################# Calculate R ###############################
#########################################################################
print('Calculating R...')
wmap_chains = read_chains(BASE_DIR + 'WMAP/test')
planck_chains = read_chains(BASE_DIR + 'Planck/test')
joint_chains = read_chains(BASE_DIR + 'Joint/test')

wmap_Z = wmap_chains.legZ(1000)
planck_Z = planck_chains.legZ(1000)
joint_Z = joint_chains.legZ(1000)

Rs = joint_Z - wmap_Z - planck_Z

R = Rs.mean()
errorR = Rs.std()

print('R = ', R, '+/-', errorR)

#########################################################################
############################### NRE  ####################################
#########################################################################
print('Running NRE...')
nSamples = config['nre']['nsamples']

load_data = config['nre']['load_data']

def nre_prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(len(parameters))]).T

if load_data:
    planckExamples = np.load(BASE_DIR + 'planck-examples-'
                              + str(int(nSamples)) + '.npy')
    wmapExamples = np.load(BASE_DIR + 'wmap-examples-'
                              + str(int(nSamples)) + '.npy')
else:
    pe, we= [], []
    for i in tqdm(range(nSamples//100)):
        samples = nre_prior(100)
        pobs, wobs, cltheory = generator(samples, lwmap, bins)
        pe.append(pobs)
        we.append(wobs)
    planckExamples = np.vstack(pe)
    wmapExamples = np.vstack(we)
    print(planckExamples.shape, wmapExamples.shape)
    np.save(BASE_DIR + 'planck-wmap-planck-examples-'
                              + str(int(nSamples)) + '.npy', planckExamples)
    np.save(BASE_DIR + 'planck-wmap-wmap-examples-'
                              + str(int(nSamples)) + '.npy', wmapExamples)

if RETRAIN:
    import os
    if os.path.exists(BASE_DIR + 'wmap_planck_nre.pkl'):
        os.remove(BASE_DIR + 'wmap_planck_nre.pkl')

try:
    nrei = nre.load(BASE_DIR + 'wmap_planck_nre.pkl', shared_prior=prior,
                     simulation_func_A=None, simulation_func_B=None)
except FileNotFoundError:

    nrei = nre(lr=config['nre']['lr'])
    nrei.build_model(len(lwmap) + len(lwmap), 
                    config['nre']['nhidden'],
                    config['nre']['activation'],)

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
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(BASE_DIR + 'loss_history_nre.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

nrei.__call__(iters=data_validation)
r = nrei.r_values
mask = np.isfinite(r)

fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.3))
axes[0, 0].hist(r[mask], bins=25,density=True)
#axes[0, 0].axvline(Rs, ls='--', c='r')
axes[0, 0].set_title('No. Sig. ' + r'$=$ ' + str(len(r[mask])))
axes[0, 0].set_xlabel(r'$\log R$')
axes[0, 0].set_ylabel('Density')

rsort  = np.sort(r[mask])
c = ecdf(rsort)

axes[0, 1].plot(rsort, c.cdf.evaluate(rsort)) 
axes[0, 1].axhline(c.cdf.evaluate(R), ls='--',
        color='r')
axes[0, 1].axhspan(c.cdf.evaluate(R - errorR), 
        c.cdf.evaluate(R + errorR), 
        alpha=0.1, 
        color='r')
axes[0, 1].set_xlabel(r'$\log R$')
axes[0, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')

axes[0, 0].axvline(R, ls='--', c='r')
axes[0, 0].axvspan(R - errorR, R + errorR, alpha=0.1, color='r')

axes[0, 1].axhline(c.cdf.evaluate(R), ls='--',
            color='r')
axes[0, 1].axhspan(c.cdf.evaluate(R - errorR),
            c.cdf.evaluate(R + errorR),
            alpha=0.1,
            color='r')

plt.tight_layout()
plt.savefig(BASE_DIR + 'wmap_planck.pdf', bbox_inches='tight')
#plt.show()
plt.close()

