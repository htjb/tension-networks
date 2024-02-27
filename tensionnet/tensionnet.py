import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle
import keras
from keras import layers
import pickle
import tqdm


class nre():
    def __init__(self, lr=1e-4):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout
        self.batch_norm = tf.keras.layers.BatchNormalization
        self.lr = lr
        self.compress = False

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)

    def build_model(
            self, input_dim, output_dim, layer_sizes, activation):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # build the model
        a0 = self.Inputs(shape=(self.input_dim,))
        inputs = a0
        for layer_size in self.layer_sizes:
            outputs = self.Dense(layer_size, activation=self.activation,
                kernel_initializer=tf.keras.initializers.GlorotNormal())(a0)
            outputs = self.batch_norm()(outputs)
            a0 = outputs
        outputs = self.Dense(self.output_dim, activation='linear',
            kernel_initializer=tf.keras.initializers.GlorotNormal())(a0)
        self.model = self.Model(inputs, outputs)
    
    def build_compress_model(
            self, input_dimA, input_dimB, output_dim, layer_sizesA,
            layer_sizesB, layer_sizesC, activation):
        
        """
        Builds a more complicated feedforward neural network with the given
        architecture. We basically compress the data (Pcal, PNS, PL, various s11s
        and the frequency) into a few nodes before mixing them in with the
        temperature.

        You will get much better performance with this network!

        input_dimA: int
            The number of inputs in the data vector (e.g. Ps and nu).
        
        input_dimB: int
            The number of inputs in the temperature vector (e.g. T).
        
        output_dim: int
            The number of output features from the network.
        
        layer_sizesA: list
            A list of integers, each integer is the number of neurons in
            that layer of the network that compressess the data.
        
        layer_sizesB: list
            A list of integers, each integer is the number of neurons in
            that layer of the network that combines the temperature and compressed
            data and outputs the probability.
        
        activation: str
            The activation function to use in the hidden layers. Best left to
            'sigmoid'.

        """
        
        self.input_dimA = input_dimA
        self.input_dimB = input_dimB
        self.output_dim = output_dim
        self.layer_sizesA = layer_sizesA
        self.layer_sizesB = layer_sizesB
        self.layer_sizesC = layer_sizesC
        self.activation = activation
        self.output_activation='sigmoid'
        self.compress = True

        # build the network that compresses the data
        a0 = self.Inputs(shape=(self.input_dimA,))
        inputsA = a0
        a0 = self.batch_norm()(a0)
        for layer_sizeA in self.layer_sizesA:
            outputs = self.Dense(layer_sizeA, activation=self.activation)(a0)
            outputs = self.batch_norm()(outputs)
            a0 = outputs

        # build the network that compresses the data
        a1 = self.Inputs(shape=(self.input_dimB,))
        inputsB = a1
        a1 = self.batch_norm()(a1)
        for layer_sizeB in self.layer_sizesB:
            outputs = self.Dense(layer_sizeB, activation=self.activation)(a1)
            outputs = self.batch_norm()(outputs)
            a1 = outputs

        # combine the two networks
        a = tf.keras.layers.Concatenate()([a0, a1])
        for layer_sizeC in self.layer_sizesC:
            outputs = self.Dense(layer_sizeC, activation=self.activation)(a)
            outputs = self.batch_norm()(outputs)
            a = outputs
        outputs = self.Dense(self.output_dim, activation=self.output_activation)(a)
        
        self.model = self.Model(inputs=[inputsA, inputsB], outputs=outputs)

        self.input_dim = [self.input_dimA, self.input_dimB]
        self.layer_sizes = [self.layer_sizesA, self.layer_sizesB, self.layer_sizesC]
    
    def build_simulations(self, simulation_func_A, simulation_func_B,
                           prior_function_A, prior_function_B,
                            shared_prior, n=10000, call_type='train'):
        
        print('Building simulations...')

        self.simulation_func_A = simulation_func_A
        self.simulation_func_B = simulation_func_B
        self.prior_function_A = prior_function_A
        self.prior_function_B = prior_function_B
        self.shared_prior = shared_prior

        thetaA = self.prior_function_A(n)
        thetaB = self.prior_function_B(n)
        thetaShared = self.shared_prior(n)

        # generate lots of simulations 
        simsA, params = [], []
        simsB = []
        for i in tqdm.tqdm(range(n)):
            simsA.append(self.simulation_func_A(thetaA[i], thetaShared[i]))
            simsB.append(self.simulation_func_B(thetaB[i], thetaShared[i]))
            params.append([*thetaA[i], *thetaB[i], *thetaShared[i]])
        simsA = np.array(simsA)
        simsB = np.array(simsB)
        self.params = np.array(params)

        simsA = (simsA - simsA.mean(axis=0)) / simsA.std(axis=0)
        simsB = (simsB - simsB.mean(axis=0)) / simsB.std(axis=0)

        #simsA = (simsA - simsA.min(axis=0)) / (simsA.max(axis=0) - simsA.min(axis=0))
        #simsB = (simsB - simsB.min(axis=0)) / (simsB.max(axis=0) - simsB.min(axis=0))

        idx = np.arange(0, n, 1)
        shuffle(idx)
        mis_labeled_simsB = simsB[idx]

        data = []
        for i in range(n):
            """
            Sigma(log(r)) = 1 results in R >> 1 i.e. data sets are consistent
            sigma(log(r)) = 0 --> R << 1 i.e. data sets are inconsistent
            """
            data.append([*simsA[i], *simsB[i], 1]) 
            if call_type == 'train':
                data.append([*simsA[i], *mis_labeled_simsB[i], 0])
        data = np.array(data)

        idx = np.arange(0, 2*n, 1)
        if call_type == 'train':
            shuffle(idx)
            input_data = data[idx, :-1]
            labels = data[idx, -1]
        elif call_type == 'eval':
            input_data = data[:, :-1]
            labels = data[:, -1]

        if call_type == 'eval':
            return input_data, labels
        elif call_type == 'train':
            self.data = input_data
            self.labels = labels
        
        print('Simulations built.')
        

    def training(self, epochs, early_stop=True, batch_size=32):

        data_train, data_test, labels_train, labels_test = \
                train_test_split(self.data, self.labels, test_size=0.2)
        
        self.data_test = data_test
        self.labels_test = labels_test
        
        train_dataset = np.hstack([data_train, labels_train[:, np.newaxis]]).astype(np.float32)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.batch(batch_size)

        self.loss_history = []
        self.test_loss_history = []
        c = 0
        for i in tqdm.tqdm(range(epochs)):

            epoch_loss_avg = tf.keras.metrics.Mean()

            loss = [self._train_step(x[:, :-1], x[:, -1]) for x in  train_dataset]
            epoch_loss_avg.update_state(loss)
            self.loss_history.append(epoch_loss_avg.result())

            self.test_loss_history.append(self._test_step(data_test, labels_test))

            if early_stop:
                c += 1
                if i == 0:
                    minimum_loss = self.test_loss_history[-1]
                    minimum_epoch = i
                    minimum_model = None
                else:
                    if self.test_loss_history[-1] < minimum_loss:
                        minimum_loss = self.test_loss_history[-1]
                        minimum_epoch = i
                        minimum_model = self.model
                        c = 0
                if minimum_model:
                    if c == round((epochs/100)*2):
                        print('Early stopped. Epochs used = ' + str(i) +
                                '. Minimum at epoch = ' + str(minimum_epoch))
                        return minimum_model, data_test, labels_test
        return self.model, data_test, labels_test

    @tf.function(jit_compile=True)
    def _test_step(self, param, truth):
            
            r"""
            This function is used to calculate the loss value at each epoch and
            adjust the weights and biases of the neural networks via the
            optimizer algorithm.
            """
    
            if self.compress:
                prediction = tf.transpose(self.model([param[:, :self.input_dimA],
                                                    param[:, self.input_dimA:]], training=True))[0]
            else:
                prediction = tf.transpose(self.model(param, training=True))[0]
            prediction = tf.keras.layers.Activation('sigmoid')(prediction)
            truth = tf.convert_to_tensor(truth)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(truth, prediction)
            return loss

    @tf.function(jit_compile=True)
    def _train_step(self, params, truth):

            r"""
            This function is used to calculate the loss value at each epoch and
            adjust the weights and biases of the neural networks via the
            optimizer algorithm.
            """

            with tf.GradientTape() as tape:
                if self.compress:
                    prediction = tf.transpose(self.model([params[:, :self.input_dimA],
                                                          params[:, self.input_dimA:]], training=True))[0]
                else:
                    prediction = tf.transpose(self.model(params, training=True))[0]
                prediction = tf.keras.layers.Activation('sigmoid')(prediction)
                truth = tf.convert_to_tensor(truth)
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(truth, prediction)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients,
                        self.model.trainable_variables))
                return loss
    
    def __call__(self, iters=2000):
        """
        Draw samples from the nre
        
        iters: int or list/array
            If int generate a new set of matching samples to evaluate the nre
            at. If list the list should contain data to evaluate at.
        """

        if isinstance(iters, int):
            data, labels = self.build_simulations(self.simulation_func_A, 
                            self.simulation_func_B,
                            self.prior_function_A, self.prior_function_B,
                            self.shared_prior, n=iters, call_type='eval')
        else:
            data = iters.copy()
        r_values = []
        for i in range(len(data)):
            params = tf.convert_to_tensor(np.array([[*data[i]]]).astype('float32'))
            if self.compress:
                logr = self.model([params[:, :self.input_dimA],
                                params[:, self.input_dimA:]]).numpy()[0]
            else:
                logr = self.model(params).numpy()[0]
            r_values.append(logr)

        self.r_values = np.array(r_values).T[0]

    def save(self, filename):

        """
        Save the network and associated data to a pickle file.

        filename: str
            The name of the file to save the network to.
        """

        w = self.model.get_weights()

        with open(filename, 'wb') as f:
            pickle.dump([w,
                         self.input_dim,
                         self.output_dim,
                         self.layer_sizes,
                         self.activation,
                         self.loss_history,
                         self.test_loss_history,
                         self.data_test,
                         self.labels_test
                         ], f)
    
    @classmethod
    def load(cls, filename,
             simulation_func_A, simulation_func_B,
             prior_function_A, prior_function_B,
             shared_prior):

        """
        Load the network and associated data from a pickle file. Gets a
        bit complicated with the compress model but it definitely works.
        """

        with open(filename, 'rb') as f:
            data = pickle.load(f)

            weights, input_dim, output_dim, layer_sizes, \
                activation, \
                loss_history, test_loss_history, \
                     data_test, labels_test = data
            
            inst = cls()
            # build the model whether compress or not
            if not isinstance(input_dim, int):
                inst.build_compress_model(input_dim[0], input_dim[1], 
                                       output_dim, 
                                       layer_sizes[0], layer_sizes[1],
                                       layer_sizes[2],
                                       activation)
            else:
                inst.build_model(input_dim, output_dim, 
                                 layer_sizes, activation)
            
            # initiallise all the important variables
            inst.model.set_weights(weights)
            inst.loss_history = loss_history
            inst.test_loss_history = test_loss_history
            inst.simulation_func_A = simulation_func_A
            inst.simulation_func_B = simulation_func_B
            inst.prior_function_A = prior_function_A
            inst.prior_function_B = prior_function_B
            inst.shared_prior = shared_prior
            inst.data_test = data_test
            inst.labels_test = labels_test
        
        return inst