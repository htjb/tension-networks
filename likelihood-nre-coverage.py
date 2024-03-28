import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from tensionnet import wmapplanck
from cmblike.data import get_data
from cmblike.cmb import CMB

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

simulator = wmapplanck.jointClGenCP(path='/Users/harrybevins/Documents/Software/cosmopower')

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

def prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(len(parameters))]).T

model = tf.keras.models.load_model('cosmopower-stuff/cosmopower_joint_likelihood.keras')

# hmmm cant work this out....

from anesthetic import read_chains

chains = read_chains('cosmopower-stuff/wmap_planck_nre_no_tau/test')

train_planck_mean = np.loadtxt('cosmopower-stuff/train_planck_mean.txt')
train_planck_std = np.loadtxt('cosmopower-stuff/train_planck_std.txt')
train_wmap_mean = np.loadtxt('cosmopower-stuff/train_wmap_mean.txt')
train_wmap_std = np.loadtxt('cosmopower-stuff/train_wmap_std.txt')
train_params_mean = np.loadtxt('cosmopower-stuff/train_params_mean.txt')
train_params_std = np.loadtxt('cosmopower-stuff/train_params_std.txt')

data = chains.values[:, :5]
idx = random.sample(range(len(data)), 1000)
data = data[idx]
f = []
for j in tqdm(range(len(data))):
    print('Data point:', j)

    # samples from P(x |theta_j)
    # samples vary because of noise   
    sims = [simulator(np.array([data[j]])) for _ in range(10)]

    # do some sorting and normalisation
    test_planck = np.array([s[0] for s in sims]).reshape(-1, praw.shape[0])
    test_wmap = np.array([s[1] for s in sims]).reshape(-1, wmapraw.shape[0])
    test_samples = np.concatenate([test_planck, test_wmap], axis=1)
    for i in range(len(test_samples)):
        test_samples[i, :len(praw)] = (test_samples[i, :len(praw)] - train_planck_mean) / train_planck_std
        test_samples[i, len(praw):] = (test_samples[i, len(praw):] - train_wmap_mean) / train_wmap_std
    
    # randomly picking a simulation as my reference data
    idx = random.sample(range(len(sims)), 1)[0]
    reference = model(tf.convert_to_tensor(np.array([[*test_samples[idx, len(praw):],
                    *data[j], *test_samples[idx, :len(praw)]]]).astype('float32')))

    # removing reference from {x_ij}
    test_samples = np.delete(test_samples, idx, axis=0)
    
    # comparing P(x_ij|theta_i) and P(x_j|theta_i)
    # hmmm i mean technically im not comparing the likelihoods here...
    ff = np.mean([1 if model(tf.convert_to_tensor(np.array([[*test_samples[t, len(praw):], 
                *data[j], *test_samples[t, :len(praw)]]]).astype('float32'))) < 
                  reference else 0 for t in range(1, len(test_samples))])
    f.append(ff)
print(f)

ecp = []
alpha=np.linspace(0, 1, 100)
for j in range(len(alpha)):
    e = np.mean([1 if f[i] < (1 - alpha[j]) else 0 for i in range(len(data))])
    ecp.append(e)
ecp = np.array(ecp)

plt.plot(1- alpha, ecp, label='ECP')
plt.plot(1- alpha, 1 - alpha, ls='--', c='k')
plt.legend()
plt.xlabel('1 - alpha')
plt.ylabel('ECP')
plt.savefig('cosmopower-stuff/ecp_nre_likelihood.png')
plt.show()