import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
import pickle
from cmblike.data import get_data
import torch
from scipy.special import logsumexp

def get_info(split):
    for j in range(len(split)):
        if split[j] == 'nets':
            number_nets = int(split[j-1])
        elif split[j] == 'nets/':
            number_nets = int(split[j-1])
        elif split[j] == 'data':
            data_norm = split[j-1]
            if data_norm == 'custom':
                data_norm = 'structured'
                marker = '*'
            else:
                marker = 'x'
        elif split[j] == 'hls' or split[j] == 'hls/':
            hls = int(split[j-1])
            if hls == 1:
                color = 'r'
            elif hls == 2:
                color='b'
    if np.any(split == 'number2') or np.any(split == 'number2/'):
        repeat = True
    else:
        repeat = False
    return number_nets, data_norm, hls, marker, repeat, color

# number2 corresponds to retraining the network on the same training data
# and sampling again with polychord

# could train a bunch of networks with best set up and average the estimates of
# the evidence from polychord...
file_names = ['wmap_planck_fit_with_independent_data_norm_plus_batch_norm_2_nets_2_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_2_nets_number2_2_hls/',
              'wmap_planck_fit_with_custom_data_norm_plus_batch_norm_2_nets_2_hls/',
              'wmap_planck_fit_with_custom_data_norm_plus_batch_norm_2_nets_number2_2_hls/',
              'wmap_planck_fit_with_custom_data_norm_plus_batch_norm_3_nets_2_hls/',
              'wmap_planck_fit_with_custom_data_norm_plus_batch_norm_3_nets_number2_2_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_3_nets_2_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_4_nets_2_hls/',
              'wmap_planck_fit_with_custom_data_norm_plus_batch_norm_2_nets_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_2_nets_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_5_nets_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_5_nets_number2_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_5_nets_number3_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_6_nets_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_8_nets_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_8_nets_number2_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_10_nets_1_hls/',
              'wmap_planck_fit_with_independent_data_norm_plus_batch_norm_12_nets_1_hls/',]

PLOT_EVIDENCES = True
PLOT_LIKELIHOODS = True

if PLOT_EVIDENCES:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    axr = ax[1]
    ax = ax[0]

    planck_chains = read_chains('Planck_fit/test')
    wmap_chains = read_chains('wmap_fit/test')
    sumlogZ = planck_chains.logZ(1000).values + wmap_chains.logZ(1000).values
    ax.axhline(np.mean(sumlogZ), color='k', linestyle='--', label='Planck + WMAP')
    ax.axhspan(np.mean(sumlogZ) - np.std(sumlogZ), np.mean(sumlogZ) + np.std(sumlogZ),
                color='k', alpha=0.3)

    for i in range(len(file_names)):
        chains = read_chains(file_names[i] + '/test')
        split = file_names[i].split('_')
        number_nets, data_norm, hls, marker, repeat, color = get_info(split)
        print(number_nets)
        logZ = chains.logZ(1000).values
        meanlogZ = np.mean(logZ)
        stdlogZ = np.std(logZ)
        ax.errorbar(number_nets, meanlogZ, yerr=stdlogZ, 
                    marker=marker, c=color,
                    #label='{}, {} nets, {} hls'.format(data_norm, number_nets, hls),
                    capsize=5, elinewidth=2)
        axr.errorbar(number_nets, meanlogZ - np.mean(sumlogZ), yerr=stdlogZ, 
                    marker=marker, c=color,
                    #label='{}, {} nets, {} hls'.format(data_norm, number_nets, hls),
                    capsize=5, elinewidth=2)

    axr.axhline(0, color='k', linestyle='--', label='Planck + WMAP')
    axr.set_xlabel('Number of nets')
    axr.set_ylabel('log(R)')
    #axr.legend()

    ax.set_xlabel('Number of nets')
    ax.set_ylabel('logZ')
    fig.suptitle('Parameters all `independent`ly normalized')
    ax.scatter([], [], c='r', label='1 hls')
    ax.scatter([], [], c='b', label='2 hls')
    ax.scatter([], [], marker='x', c='k', label='independent')
    ax.scatter([], [], marker='*', c='k', label='structured')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('nle-evidences.png', dpi=300, bbox_inches='tight')
    plt.show()

if PLOT_LIKELIHOODS:
    wmapraw, lwmap = get_data(base_dir='../tension-networks/cosmology-data/').get_wmap()
    praw, l = get_data(base_dir='../tension-networks/cosmology-data/').get_planck()

    training_data = np.loadtxt('planck-wmap-nle-examples.txt').astype(np.float32)
    training_params = np.loadtxt('planck-wmap-nle-params.txt').astype(np.float32)
    import random
    idx = random.sample(range(len(training_data)), 5000)
    training_data = training_data[idx]
    training_params = training_params[idx]
    training_planck = training_data[:, :len(praw)]
    training_wmap = training_data[:, len(praw):]

    likelihoods, labels = [], []
    nns, dn, hiddenls = [], [], []
    cls = []
    marks = []
    for i in range(len(file_names)):
        pickle_file = file_names[i].split('/')[0] + '.pkl'
        pickle_file = 'planck_wmap_likelihood_' + ''.join(pickle_file[16:])
        with open(pickle_file, 'rb') as f:
            density_estimator = pickle.load(f)
        split = file_names[i].split('_')
        number_nets, data_norm, hls, marker, repeat, colors = get_info(split)

        if data_norm == 'structured' or data_norm == 'custom':
            std_planck = np.std(training_planck)
            std_wmap = np.std(training_wmap)
            stds = np.concatenate([np.ones_like(praw)*1/std_planck, 
                                  np.ones_like(wmapraw)*1/std_wmap])
            correction = np.linalg.slogdet(np.diag(stds))[1]

            norm_praw = (training_planck - np.mean(training_planck)) / std_planck
            norm_wmapraw = (training_wmap - np.mean(training_wmap)) / std_wmap
            data = np.hstack([norm_praw, norm_wmapraw]).astype(np.float32)


        elif data_norm == 'independent':
            std_planck = np.std(training_planck, axis=0)
            std_wmap = np.std(training_wmap, axis=0)
            stds = np.concatenate([1/std_planck, 1/std_wmap])
            correction = np.linalg.slogdet(np.diag(stds))[1]

            # redefine data
            norm_praw = (training_planck - np.mean(training_planck, axis=0)) / std_planck
            norm_wmapraw = (training_wmap - np.mean(training_wmap, axis=0)) / std_wmap
            data = np.hstack([norm_praw, norm_wmapraw]).astype(np.float32)
        elif data_norm == 'covariance':
            covariance = np.cov(training_data.T)
            invL = np.linalg.inv(np.linalg.cholesky(covariance))
            correction = np.linalg.slogdet(invL)[1]
            data = np.dot((training_data - 
                        np.mean(training_data, axis=0)), invL).astype(np.float32)
        
        def likelihood(datai, paramsi):
            return density_estimator.log_prob(torch.tensor([datai]), 
                            torch.tensor([paramsi.astype(np.float32)])).detach().numpy() + correction
        
        like = np.array([likelihood(data[i], training_params[i]) for i in range(len(data))])
        likelihoods.append(np.mean(like))
        nns.append(number_nets)
        dn.append(data_norm)
        hiddenls.append(hls)
        marks.append(marker)
        cls.append(colors)
    
    likelihoods -= np.max(likelihoods)
    print(likelihoods)
    [plt.scatter(nns[i], likelihoods[i], marker=marks[i], c=cls[i],
                #label='{}, {} nets, {} hls'.format(dn[i], nns[i], hiddenls[i])
                ) for i in range(len(nns))]
    #plt.legend(bbox_to_anchor=(0, 1.3))
    plt.scatter([], [], c='r', label='1 hls')
    plt.scatter([], [], c='b', label='2 hls')
    plt.scatter([], [], marker='x', c='k', label='independent')
    plt.scatter([], [], marker='*', c='k', label='structured')
    plt.legend(loc='lower right')
    plt.xlabel('Number of nets')
    plt.ylabel(r'$\frac{1}{N} \sum_{i=1}^N \log p(x_i | \theta_i)$' + 
                r'$- \max_i \frac{1}{N} \sum_{i=1}^N \log p(x_i | \theta_i)$')
    plt.axhline(0, color='k', linestyle='--')
    #plt.yscale('log')
    plt.ylim(-10, 0)
    plt.tight_layout()
    plt.savefig('nle-likelihoods.png', dpi=300, bbox_inches='tight')
    plt.show()
