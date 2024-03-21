import torch
import numpy as np
import healpy as hp
import camb
import matplotlib.pyplot as plt
from cmblike.noise import planck_noise, wmap_noise
from cmblike.data import get_data

from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import likelihood_nn
from sbi import analysis as analysis
from tensionnet import wmapplanck
from sbi import utils

import pickle


wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

NUM_NETS = 12
DATA_NORM = 'independent'
HIDDEN_LAYERS = 1
REPEAT = 1
NOTAU = True

if NOTAU:
    prior = utils.BoxUniform(low=torch.tensor([0.01, 0.08, 0.97, 0.8, 2.6]),
                            high=torch.tensor([0.085, 0.21, 1.5, 1.2, 3.8]))
else:
    prior = utils.BoxUniform(low=torch.tensor([0.01, 0.08, 0.97, 0.01, 0.8, 2.6]),
                                high=torch.tensor([0.085, 0.21, 1.5, 0.16, 1.2, 3.8]))

density_estimator_build_fun = likelihood_nn(
    model="maf", hidden_features=50, hidden_layers=HIDDEN_LAYERS,
    num_transforms=NUM_NETS, z_score_x=None, z_score_theta='independent',
    use_batch_norm=True,
)


inference = SNLE(prior=prior, density_estimator=density_estimator_build_fun)
try:
    x = np.loadtxt('planck-wmap-nle-examples.txt').astype(np.float32)
    theta = np.loadtxt('planck-wmap-nle-params.txt').astype(np.float32)
except FileNotFoundError:
    nSamples = 25000
    joint = wmapplanck.jointClGen()
    clLibrary = np.load('cl_library/cls.npy')
    paramsLibrary = np.load('cl_library/params.npy')

    idx = np.arange(nSamples)
    np.random.shuffle(idx)
    clLibrary = clLibrary[idx]
    paramsLibrary = list(paramsLibrary[idx])
    paramsLibrary = [list(item) for item in paramsLibrary]

    from tqdm import tqdm
    x, theta = [], []
    for i in tqdm(range(nSamples)):
        x.append(np.concatenate(joint(None, paramsLibrary[i], clexample=clLibrary[i])).astype(np.float32))
        theta.append(paramsLibrary[i].astype(np.float32))
    
    np.savetxt('planck-wmap-nle-examples.txt', x)
    np.savetxt('planck-wmap-nle-params.txt', theta)

if NOTAU:
    theta = np.delete(theta, 3, axis=1)


if DATA_NORM == 'independent':
    planck = x[:, :len(praw)] 
    wmap = x[:, len(praw):]

    planck = (planck - np.mean(planck, axis=0)) / np.std(planck, axis=0)
    wmap = (wmap - np.mean(wmap, axis=0)) / np.std(wmap, axis=0)
    x = np.hstack([planck.astype(np.float32), wmap.astype(np.float32)])
elif DATA_NORM == 'custom':
    # custom just means structured... bad naming
    planck = x[:, :len(praw)] 
    wmap = x[:, len(praw):]

    planck = (planck - np.mean(planck)) / np.std(planck)
    wmap = (wmap - np.mean(wmap)) / np.std(wmap)
    x = np.hstack([planck.astype(np.float32), wmap.astype(np.float32)])
elif DATA_NORM == 'covariance':
    covariance = np.cov(x.T)
    invL = np.linalg.inv(np.linalg.cholesky(covariance))
    x = np.dot((x - np.mean(x, axis=0)), invL).astype(np.float32)

x = torch.tensor(x)
theta = torch.tensor(theta)

inference = inference.append_simulations(theta, x)
density_estimator = inference.train()

if REPEAT > 1:
    with open('planck_wmap_likelihood_with_' + DATA_NORM + '_'+ 
            'data_norm_plus_batch_norm_' + str(NUM_NETS) +
            '_nets_number' + str(REPEAT) + '_' + str(HIDDEN_LAYERS) + '_hls.pkl', 'wb') as f:
        pickle.dump(density_estimator, f)
else:
    with open('planck_wmap_likelihood_with_' + DATA_NORM + '_'+ 
          'data_norm_plus_batch_norm_' + str(NUM_NETS) +'_nets_' + str(HIDDEN_LAYERS) + '_hls.pkl', 'wb') as f:
        pickle.dump(density_estimator, f)
print(x, theta)
print(density_estimator.log_prob(x, theta).max())

plt.plot(np.array(inference._summary['training_log_probs']), label='Train')
plt.plot(np.array(inference._summary['validation_log_probs']), label='test')
plt.yscale('log')
plt.legend()
plt.savefig('planck_wmap_likelihood_with_' + DATA_NORM + '_data_norm' +
            '_plus_batch_norm_' + str(NUM_NETS) +'_nets_number' + str(REPEAT) + '_' + 
            str(HIDDEN_LAYERS) + '_hls_loss.png', dpi=300,
            bbox_inches='tight')
plt.show()
