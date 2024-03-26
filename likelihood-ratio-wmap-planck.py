import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from tensionnet import wmapplanck
from cmblike.data import get_data
from cmblike.cmb import CMB
from tqdm import tqdm

#@tf.function(jit_compile=True)
def _test_step(param, truth):
        
    r"""
    This function is used to calculate the loss value at each epoch and
    adjust the weights and biases of the neural networks via the
    optimizer algorithm.
    """
    prediction = tf.transpose(model(param, training=False))[0]
    prediction = tf.keras.layers.Activation('sigmoid')(prediction)
    truth = tf.convert_to_tensor(truth)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(truth, prediction)
    return loss

#@tf.function(jit_compile=True)
def _train_step(params, truth):

    r"""
    This function is used to calculate the loss value at each epoch and
    adjust the weights and biases of the neural networks via the
    optimizer algorithm.
    """

    with tf.GradientTape() as tape:
        prediction = tf.transpose(model(params, training=True))[0]
        prediction = tf.keras.layers.Activation('sigmoid')(prediction)
        truth = tf.convert_to_tensor(truth)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(truth, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients,
            model.trainable_variables))
    return loss


######## Load the data ########
wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

import os

###### set up... ########
BASE_DIR = 'cosmopower-stuff/'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

load = True
load_trained_model = False
nSamples = 50000
joint = wmapplanck.jointClGenCP(path='/Users/harrybevins/Documents/Software/cosmopower')

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

def prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(len(parameters))]).T

if load:
    planckExamples = np.load(BASE_DIR + 'planckExamplesCP.npy')
    wmapExamples = np.load(BASE_DIR + 'wmapExamplesCP.npy')
    samples = np.load(BASE_DIR + 'samplesCP.npy')
else:
    pe, we, samps = [], [], []
    #for i in tqdm(range(nSamples//100)):
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
    np.save(BASE_DIR + 'planckExamplesCP.npy', planckExamples)
    np.save(BASE_DIR + 'wmapExamplesCP.npy', wmapExamples)
    np.save(BASE_DIR + 'samplesCP.npy', samples)

print(samples)
print(planckExamples.shape, wmapExamples.shape, samples.shape)

if load_trained_model:
    trainWmap = np.loadtxt('cosmopower-stuff/train_wmap.txt')
    testWmap = np.loadtxt('cosmopower-stuff/test_wmap.txt')
    trainPlanck = np.loadtxt('cosmopower-stuff/train_planck.txt')
    testPlanck = np.loadtxt('cosmopower-stuff/test_planck.txt')
    trainParams = np.loadtxt('cosmopower-stuff/train_params.txt')
    testParams = np.loadtxt('cosmopower-stuff/test_params.txt')
else:
    print('Careful if loading a new model because the test data and train data will get shuffled differently')
    splitIdx = np.arange(len(wmapExamples))
    np.random.shuffle(splitIdx)
    trainIdx = splitIdx[:int(len(wmapExamples)*0.8)]
    testIdx = splitIdx[int(len(wmapExamples)*0.8):]

    print('Splitting data...')
    trainWmap = wmapExamples[trainIdx]
    testWmap = wmapExamples[testIdx]
    trainPlanck = planckExamples[trainIdx]
    testPlanck = planckExamples[testIdx]
    trainParams = samples[trainIdx]
    testParams = samples[testIdx]

    np.savetxt('cosmopower-stuff/train_wmap.txt', trainWmap)
    np.savetxt('cosmopower-stuff/test_wmap.txt', testWmap)
    np.savetxt('cosmopower-stuff/train_planck.txt', trainPlanck)
    np.savetxt('cosmopower-stuff/test_planck.txt', testPlanck)
    np.savetxt('cosmopower-stuff/train_params.txt', trainParams)
    np.savetxt('cosmopower-stuff/test_params.txt', testParams)
    np.savetxt('cosmopower-stuff/train_wmap_mean.txt', np.mean(trainWmap, axis=0))
    np.savetxt('cosmopower-stuff/train_wmap_std.txt', np.std(trainWmap, axis=0))
    np.savetxt('cosmopower-stuff/train_planck_mean.txt', np.mean(trainPlanck, axis=0))
    np.savetxt('cosmopower-stuff/train_planck_std.txt', np.std(trainPlanck, axis=0))
    np.savetxt('cosmopower-stuff/train_params_mean.txt', np.mean(trainParams, axis=0))
    np.savetxt('cosmopower-stuff/train_params_std.txt', np.std(trainParams, axis=0))

print('Normalising data...')
normtrainwmapExamples = (trainWmap -np.mean(trainWmap, axis=0))/np.std(trainWmap, axis=0)
normtestwmapExamples = (testWmap -np.mean(trainWmap, axis=0))/np.std(trainWmap, axis=0)
normtrainplanckExamples = (trainPlanck -np.mean(trainPlanck, axis=0))/np.std(trainPlanck, axis=0)
normtestplanckExamples = (testPlanck -np.mean(trainPlanck, axis=0))/np.std(trainPlanck, axis=0)
normtrainParams = (trainParams -np.mean(trainParams, axis=0))/np.std(trainParams, axis=0)
normtestParams = (testParams -np.mean(trainParams, axis=0))/np.std(trainParams, axis=0)

print('Shuffling and stacking training and test data...')
matchedtrainData = np.hstack([normtrainwmapExamples, normtrainParams, normtrainplanckExamples])
matchedtrainLabels = np.ones(len(matchedtrainData))
matchedtestData = np.hstack([normtestwmapExamples, normtestParams, normtestplanckExamples])
matchedtestLabels = np.ones(len(matchedtestData))

idx = np.arange(len(matchedtrainData))
np.random.shuffle(idx)
shuffledtrainPlanck = normtrainplanckExamples[idx]
shuffledtrainData = np.hstack([normtrainwmapExamples, normtrainParams, shuffledtrainPlanck])
shuffledtrainLabels = np.zeros(len(shuffledtrainData))

data_train = np.vstack([matchedtrainData, shuffledtrainData])
labels_train = np.hstack([matchedtrainLabels, shuffledtrainLabels])

idx = np.arange(len(matchedtestData))
np.random.shuffle(idx)
shuffledtestPlanck = normtestplanckExamples[idx]
shuffledtestData = np.hstack([normtestwmapExamples, normtestParams, shuffledtestPlanck])
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


if load_trained_model:
    print('Loading network...')
    model = tf.keras.models.load_model('cosmopower-stuff/cosmopower_joint_likelihood.keras')
else:
    print('Building network...')
    optimizer = tf.keras.optimizers.Adam(
                    learning_rate=1e-4)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(data_train.shape[1], activation='sigmoid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='linear',
                            kernel_initializer=tf.keras.initializers.GlorotNormal()),
    ])

    print('Training network...')
    epochs = 10000
    batch_size = 32000
    patience = 100

    train_dataset = np.hstack([data_train, labels_train[:, np.newaxis]]).astype(np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = np.hstack([data_test, labels_test[:, np.newaxis]]).astype(np.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
    test_dataset = test_dataset.batch(batch_size)

    loss_history = []
    test_loss_history = []
    epoch_loss_avg = tf.keras.metrics.Mean()
    c = 0
    #for i in tqdm(range(epochs)):
    for i in range(epochs):

        loss = [_train_step(x[:, :-1], x[:, -1]) for x in  train_dataset]
        epoch_loss_avg.update_state(loss)
        loss_history.append(epoch_loss_avg.result())

        test_loss = [_test_step(x[:, :-1], x[:, -1]) for x in  test_dataset]
        test_loss_history.append(tf.reduce_mean(test_loss))
        print('Epoch: {} Loss: {:.5f} Test Loss: {:.5f}'.format(
            i, loss_history[-1], test_loss_history[-1]))

        c += 1
        if i == 0:
            minimum_loss = test_loss_history[-1]
            minimum_epoch = i
            minimum_model = model
        else:
            if test_loss_history[-1] < minimum_loss:
                minimum_loss = test_loss_history[-1]
                minimum_epoch = i
                minimum_model = model
                c = 0
        if minimum_model:
            if c == patience:
                print('Early stopped. Epochs used = ' + str(i) +
                        '. Minimum at epoch = ' + str(minimum_epoch))
                model = minimum_model
                break

    plt.plot(loss_history, label='train')
    plt.plot(test_loss_history, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('cosmopower-stuff/liklelihood_ratio_loss_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    model.save('cosmopower-stuff/cosmopower_joint_likelihood.keras')

print('Evaluation test...')
from cmblike.noise import wmap_noise
from cmblike.cmb import CMB

wmap_noise = wmap_noise(lwmap).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')
likelihood = cmbs.get_likelihood(wmapraw, lwmap, noise=wmap_noise, cp=True)

def __call__(planck, wmap, params, nonNormParams):

        r_values = []
        for i in range(len(params)):
            ps = tf.convert_to_tensor(np.array([[*wmap, 
                        *params[i], *planck]]).astype('float32'))
            logr = model(ps).numpy()[0]
            r_values.append(logr + likelihood(nonNormParams[i])[0] + 589.8)

        r_values = np.array(r_values).T[0]
        return r_values


normWmap = (wmapraw - np.mean(trainWmap, axis=0))/np.std(trainWmap, axis=0)
normPlanck = (praw - np.mean(trainPlanck, axis=0))/np.std(trainPlanck, axis=0)

samps = samples[:200]
norm_params = (samps - np.mean(trainParams, axis=0))/np.std(trainParams, axis=0)

# logL - log Z
r_values = __call__(normPlanck, normWmap, norm_params, samps)
prior = np.log(1/(0.04-0.005)*1/(0.21 -0.08)*1/(1.2-0.8)*1/(3.8-2.6)*1/(0.9-0.5))
r_values = r_values + prior
r_values -= np.max(r_values)

cbar = plt.scatter(samps[:, 0], samps[:, 1], c=r_values, s=10, cmap='viridis')
plt.colorbar(cbar)
plt.xlabel('Omega_b h^2')
plt.ylabel('Omega_c h^2')
plt.show()

print('Testing for confusion...')
idx = random.sample(range(len(data_test)), 1000)
data_test = data_test[idx]
labels_test = labels_test[idx]
r_values = []
for i in range(len(data_test)):
    ps = tf.convert_to_tensor(np.array([data_test[i]]).astype('float32'))
    logr = model(ps).numpy()[0]
    rv = tf.keras.layers.Activation('sigmoid')(logr)
    r_values.append(rv)
p = np.array(r_values).T[0]

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

fig, axes = plt.subplots(1, 1)
axes.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(3):
        axes.text(j, i, '{:.2f}'.format(cm[i][j]) + r'$\%$', ha='center', va='center', color='k',
                bbox=dict(facecolor='white', lw=0), fontsize=10)
axes.set_xticks([0, 1, 2], ['Correct', 'Wrong', 'Confused'])
axes.set_yticks([0, 1], [r'$P(D_p(\phi))~P(D_w(\theta), \theta)$', r'$P(D_p(\theta), D_w(\theta), \theta)$'])
axes.set_xlabel('Prediction')
axes.set_ylabel('Truth')
plt.tight_layout()
plt.savefig('cosmopower-stuff/confusion_matrix_joint_likelihood.png', dpi=300, bbox_inches='tight')
plt.show()