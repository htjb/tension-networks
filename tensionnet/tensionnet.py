import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle
import keras
from keras import layers
import pickle
import tqdm
import time


class nre():
    def __init__(self, lr=1e-4, reduction_method='sum',
                 test_size=0.2):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout
        self.batch_norm = tf.keras.layers.BatchNormalization
        self.lr = lr
        self.reduction_method = reduction_method
        self.test_size = test_size

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)
        #self.optimizer = tf.keras.optimizers.legacy.Nadam(learning_rate=self.lr)

    def build_model(
            self, input_dim, layer_sizes, activation, 
            skip_layers=True, kernel_regularizer='l1',
            use_bias=False, dropout=0.0):
        
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.drop = dropout
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.skip_layers = skip_layers
        self.compress = None
        
        # build the model
        a0 = self.Inputs(shape=(self.input_dim,))
        inputs = a0
        for i, layer_size in enumerate(self.layer_sizes):
            outputs = self.Dense(layer_size, 
                                 activation=self.activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 use_bias=self.use_bias,
                                 )(a0)
            outputs = self.Dropout(self.drop)(outputs)
            if self.skip_layers:
                if i % 2 == 0 and i != 0:
                    outputs = self.batch_norm()(outputs)
                    outputs = tf.keras.layers.add([outputs, a0])
                    outputs = tf.keras.layers.Activation(self.activation)(outputs)
            outputs = self.batch_norm()(outputs)
            a0 = outputs
        outputs = self.Dense(1, activation='linear',
                            use_bias=self.use_bias,
                            kernel_regularizer=self.kernel_regularizer)(a0)
        self.model = self.Model(inputs, outputs)

    def build_compress_model(
            self, input_dimA, input_dimB, compress_layer_sizesA, 
            layer_sizes, activation, compress='both', skip_layers=True,
            kernel_regularizer='l1', use_bias=False, dropout=0.0,
            compress_layer_sizesB=None):
        
        if compress_layer_sizesB is None:
            self.compress_layer_sizesB = compress_layer_sizesA
        else:
            self.compress_layer_sizesB = compress_layer_sizesB
        self.input_dimA = input_dimA
        self.input_dimB = input_dimB
        self.input_dim = input_dimA + input_dimB
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.drop = dropout
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.skip_layers = skip_layers
        self.compress = compress
        self.compress_layer_sizesA = compress_layer_sizesA

        
        # build the model
        all_inputs = self.Inputs(shape=(self.input_dim,))
        inputs = all_inputs
        if self.compress == 'A' or self.compress == 'both':
            a0 = all_inputs[:, :self.input_dimA]
            for i, layer_size in enumerate(self.compress_layer_sizesA):
                outputs = self.Dense(layer_size, 
                                    activation=self.activation,
                                    kernel_regularizer=self.kernel_regularizer,
                                    use_bias=self.use_bias,
                                    )(a0)
                outputs = self.Dropout(self.drop)(outputs)
                outputs = self.batch_norm()(outputs)
                a0 = outputs
        else:
            a0 = all_inputs[:, :self.input_dimA]
        if self.compress == 'B' or self.compress == 'both':
            a1 = all_inputs[:, self.input_dimA:]
            for i, layer_size in enumerate(self.compress_layer_sizesB):
                outputs = self.Dense(layer_size, 
                                    activation=self.activation,
                                    kernel_regularizer=self.kernel_regularizer,
                                    use_bias=self.use_bias,
                                    )(a1)
                outputs = self.Dropout(self.drop)(outputs)
                outputs = self.batch_norm()(outputs)
                a1 = outputs
        else:
            a1 = all_inputs[:, self.input_dimA:]
        a0 = tf.keras.layers.concatenate([a0, a1])
        for i, layer_size in enumerate(self.layer_sizes):
            outputs = self.Dense(layer_size, 
                                 activation=self.activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 use_bias=self.use_bias,
                                 )(a0)
            outputs = self.Dropout(self.drop)(outputs)
            if self.skip_layers:
                if i % 2 == 0 and i != 0:
                    outputs = self.batch_norm()(outputs)
                    outputs = tf.keras.layers.add([outputs, a0])
                    outputs = tf.keras.layers.Activation(self.activation)(outputs)
            outputs = self.batch_norm()(outputs)
            a0 = outputs
        outputs = self.Dense(1, activation='linear',
                            use_bias=self.use_bias,
                            kernel_regularizer=self.kernel_regularizer)(a0)
        self.model = self.Model(inputs, outputs)
    
    def build_simulations(self, simulation_func_A, simulation_func_B,
                            shared_prior, n=10000, call_type='train',
                            prior_function_A=None, prior_function_B=None):
        
        print('Building simulations...')

        self.simulation_func_A = simulation_func_A
        self.simulation_func_B = simulation_func_B


        self.shared_prior = shared_prior
        thetaShared = self.shared_prior(n)

        if prior_function_A:
            self.prior_function_A = prior_function_A
            thetaA = self.prior_function_A(n)
            thetaA = np.hstack([thetaA, thetaShared])
        else:
            self.prior_function_A = None
            thetaA = thetaShared.copy()

        if prior_function_B:
            self.prior_function_B = prior_function_B
            thetaB = self.prior_function_B(n)
            thetaB = np.hstack([thetaB, thetaShared])
        else:
            self.prior_function_B = None
            thetaB = thetaShared.copy()
        

        # generate lots of simulations 
        simsB, simsA = [], []
        for i in tqdm.tqdm(range(n)):
            simsA.append(self.simulation_func_A(thetaA[i]))
            simsB.append(self.simulation_func_B(thetaB[i]))
        simsA = np.array(simsA)
        simsB = np.array(simsB)

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
            # need to normalise the data
            dataA = input_data[:, :len(simsA[0])]
            dataB = input_data[:, len(simsA[0]):]
            data_trainA = self.data_train[:, :len(simsA[0])]
            data_trainB = self.data_train[:, len(simsA[0]):]
            dataA = (dataA - data_trainA.mean(axis=0)) / \
                data_trainA.std(axis=0)
            dataB = (dataB - data_trainB.mean(axis=0)) / \
                data_trainB.std(axis=0)
            input_data = np.hstack([dataA, dataB])
            print('Simulations built and normalized.')
            return input_data, labels
        elif call_type == 'train':
            self.data = input_data
            self.labels = labels
        
        print('Simulations built.')
        print('Splitting data and normalizing...')

        data_train, data_test, labels_train, labels_test = \
                train_test_split(self.data, self.labels, 
                                 test_size=self.test_size)
        
        self.labels_test = labels_test
        self.labels_train = labels_train

        data_trainA = data_train[:, :len(simsA[0])]
        data_trainB = data_train[:, len(simsA[0]):]
        data_testA = data_test[:, :len(simsA[0])]
        data_testB = data_test[:, len(simsA[0]):]

        data_testA = (data_testA - data_trainA.mean(axis=0)) / \
            data_trainA.std(axis=0)
        data_testB = (data_testB - data_trainB.mean(axis=0)) / \
            data_trainB.std(axis=0)
        data_trainA = (data_trainA - data_trainA.mean(axis=0)) / \
            data_trainA.std(axis=0)
        data_trainB = (data_trainB - data_trainB.mean(axis=0)) / \
            data_trainB.std(axis=0)

        self.data_train = np.hstack([data_trainA, data_trainB])
        self.data_test = np.hstack([data_testA, data_testB])

        print('Data split and normalized.')
        
    def training(self, epochs, early_stop=True, batch_size=32, patience=None):
        
        
        train_dataset = np.hstack([self.data_train, 
                                   self.labels_train[:, np.newaxis]]
                                   ).astype(np.float32)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.batch(batch_size)

        test_dataset = np.hstack([self.data_test,
                                    self.labels_test[:, np.newaxis]]
                                    ).astype(np.float32)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
        test_dataset = test_dataset.batch(batch_size)

        if patience is None:
            patience = round((epochs/100)*2)
        
        self.loss_history = []
        self.test_loss_history = []
        c = 0
        #for i in tqdm.tqdm(range(epochs)):
        for i in range(epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()

            s = time.time()
            loss = [self._train_step(x[:, :-1], x[:, -1]) 
                    for x in  train_dataset]
            epoch_loss_avg.update_state(loss)
            self.loss_history.append(epoch_loss_avg.result())

            self.test_loss_history.append(np.mean([
                self._test_step(x[:, :-1], x[:, -1]) 
                    for x in  test_dataset]))
            e = time.time()
            print('Epoch: ' + str(i) + ' Loss: ' + str(self.loss_history[-1]) +
                    ' Test Loss: ' + str(self.test_loss_history[-1]), ' Time: ' + str(e-s))

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
                if c == patience:
                    print('Early stopped. Epochs used = ' + str(i) +
                            '. Minimum at epoch = ' + str(minimum_epoch))
                    return minimum_model, self.data_test, self.labels_test
        return self.model, self.data_test, self.labels_test

    @tf.function(jit_compile=True)
    def _test_step(self, param, truth):
            
            r"""
            This function is used to calculate the loss value at each epoch and
            adjust the weights and biases of the neural networks via the
            optimizer algorithm.
            """
    
            prediction = tf.transpose(self.model(param, training=False))[0]
            prediction = tf.keras.layers.Activation('sigmoid')(prediction)
            truth = tf.convert_to_tensor(truth)
            loss = tf.keras.losses.BinaryCrossentropy(
                                from_logits=False,
                                reduction=self.reduction_method
                                )(truth, prediction)
            return loss

    @tf.function(jit_compile=True)
    def _train_step(self, params, truth):

            r"""
            This function is used to calculate the loss value at each epoch and
            adjust the weights and biases of the neural networks via the
            optimizer algorithm.
            """

            with tf.GradientTape() as tape:
                prediction = tf.transpose(self.model(params, 
                                                         training=True))[0]
                prediction = tf.keras.layers.Activation('sigmoid')(prediction)
                truth = tf.convert_to_tensor(truth)
                loss = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False, reduction=self.reduction_method
                    )(truth, prediction)
                gradients = tape.gradient(loss, 
                                          self.model.trainable_variables)
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
                            self.simulation_func_B, self.shared_prior,
                            prior_function_A=self.prior_function_A, 
                            prior_function_B=self.prior_function_B,
                            n=iters, call_type='eval')
        else:
            data = iters.copy()
        
        r_values = []
        for i in range(len(data)):
            params = tf.convert_to_tensor(np.array(
                [[*data[i]]]).astype('float32'))
            logr = self.model(params, training=False).numpy()[0]
            r_values.append(logr)

        self.r_values = np.array(r_values).T[0]
        
        return data

    def save(self, filename):

        """
        Save the network and associated data to a pickle file.

        filename: str
            The name of the file to save the network to.
        """

        w = self.model.get_weights()

        if not self.compress:
            with open(filename, 'wb') as f:
                pickle.dump([w,
                            self.input_dim,
                            self.layer_sizes,
                            self.activation,
                            self.loss_history,
                            self.test_loss_history,
                            self.data_test,
                            self.labels_test,
                            self.data_train,
                            self.labels_train,
                            self.use_bias,
                            self.drop,
                            self.kernel_regularizer,
                            self.skip_layers,
                            ], f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump([w,
                            self.input_dimA,
                            self.input_dimB,
                            self.compress_layer_sizesA,
                            self.compress_layer_sizesB,
                            self.layer_sizes,
                            self.activation,
                            self.loss_history,
                            self.test_loss_history,
                            self.data_test,
                            self.labels_test,
                            self.data_train,
                            self.labels_train,
                            self.use_bias,
                            self.drop,
                            self.kernel_regularizer,
                            self.skip_layers,
                            self.compress,
                            ], f)

    @classmethod
    def load(cls, filename,
             simulation_func_A, simulation_func_B, shared_prior,
             prior_function_A=None, prior_function_B=None):

        """
        Load the network and associated data from a pickle file. Gets a
        bit complicated with the compress model but it definitely works.
        """

        with open(filename, 'rb') as f:
            data = pickle.load(f)

            try:
                weights, input_dim, layer_sizes, \
                    activation, \
                    loss_history, test_loss_history, \
                        data_test, labels_test, \
                            data_train, labels_train, use_bias, \
                                dropout, kernel_regularizer, \
                                    skip_layers = data
                compress = None
            except:
                weights, input_dimA, input_dimB, \
                    compress_layer_sizesA, compress_layer_sizesB, \
                    layer_sizes, activation, \
                    loss_history, test_loss_history, \
                        data_test, labels_test, \
                            data_train, labels_train, use_bias, \
                                dropout, kernel_regularizer, \
                                    skip_layers, compress = data
            
            inst = cls()
            if compress is None:
                inst.build_model(input_dim, 
                                 layer_sizes, activation,
                                 skip_layers=skip_layers, 
                                 kernel_regularizer=kernel_regularizer,
                                 use_bias=use_bias, dropout=dropout)
            else:
                inst.build_compress_model(input_dimA, input_dimB, 
                                    compress_layer_sizesA, 
                                    layer_sizes, activation,
                                    compress=compress, 
                                    skip_layers=skip_layers, 
                                    kernel_regularizer=kernel_regularizer,
                                    use_bias=use_bias, dropout=dropout,
                                    compress_layer_sizesB=compress_layer_sizesB)
            
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
            inst.data_train = data_train
            inst.labels_train = labels_train
        
        return inst