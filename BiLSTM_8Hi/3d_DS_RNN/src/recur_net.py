import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cPickle as pkl
import time
import os.path

import theano
import theano.tensor as T

from model_base.rnn_xy import RNN, BiRNN
from model_base.gru_xy import GRU, BiGRU
from model_base.lstm_xy import LSTM, BiLSTM, VanillaStack_BiLSTM

from utility.optimizers import get_optimizer
from utility.vol_grid_io import vols_serialization, vols_serialization_bbx, vols_serialization_subgrid
dtype = theano.config.floatX

ds_N = 8


__author__ = 'xinyang'


class recur_net_c(object):
    #########################
    # Initialize parameters #
    #########################
    def __init__(self, recur_type='gru', n_input=100, minibatch=2, n_hidden=6, n_layer=1, n_output=100, n_class=2,
                 optimizer='rmsprop', learning_rate=0.001, n_epochs=8, vld_intval=10, earlyErr_T = 1.0/10,
                 info_file=None, model_file=None, err_txt=None):
        print '### initializing recurrent model...'
        self.recur_type = recur_type
        self.n_input = n_input
        self.minibatch = minibatch
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.n_output = n_output
        self.n_class = n_class
        self.optimizer = optimizer
        # self.learning_rate = learning_rate
        self.train_epochs = n_epochs
        self.final_epoch = n_epochs
        self.valid_intval = vld_intval
        self.earlyErr_T = earlyErr_T
        self.model_params = None
        # save file
        self.model_file = model_file
        self.info_file = info_file
        self.err_txt = err_txt

        # basic structure parameter collection, facilitating future loading
        self.basic_info = [self.recur_type, self.n_input, self.minibatch,
                           self.n_hidden, self.n_layer, self.n_output,
                           self.optimizer, learning_rate,
                           self.train_epochs, self.valid_intval, self.earlyErr_T]
        print '# success!'


    ##############################
    # Load Exist Recurrent Model #
    ##############################
    def load_exist_model(self, ex_info_file, ex_model_file):
        if self.recur_type == 'birnn' or self.recur_type == 'bigru' or self.recur_type == 'bilstm':
            print('Loading parameters for xxxxxx models is not supported')
            raise SystemExit
        else:
            print 'loading existing network info. and model parameters...'
            ## network information
            if os.path.isfile(ex_info_file):
                with open(ex_info_file, 'rb') as f:
                    self.basic_info = pkl.load(f)
                    # update
                    [self.recur_type, self.n_input, self.minibatch,
                    self.n_hidden, self.n_layer, self.n_output,
                    self.optimizer, self.learning_rate,
                    self.train_epochs, self.valid_intval, self.earlyErr_T] = self.basic_info
                    print '========= model info. ========='
                    print 'type: %s, input=%d, minibatch=%d, hiddenUnit=%d,\n' \
                          'layer=%d, output=%d, optimizer=%s, learnRate=%.4f,\n' \
                          'epoch=%d, valid_intval=%d, earlyErr_T=%.3f' % \
                          (self.recur_type, self.n_input, self.minibatch, self.n_hidden, self.n_layer,
                           self.n_output, self.optimizer, self.learning_rate, self.train_epochs,
                           self.valid_intval, self.earlyErr_T)
            else:
                print('Unable to load network info. %s!' % self.model_file)

            ## existing network parameters
            if os.path.isfile(ex_model_file):
                with open(ex_model_file, 'rb') as f:
                    self.model_params = pkl.load(f)
                print '# success!'
            else:
                print('Unable to load existing model %s!' % self.model_file)

    #########################
    # Build Recurrent Model #
    #########################
    def build_symbol_model(self, mode='TRAIN'):
        print('### building the model')

        ## check
        if mode != 'TRAIN' and mode != 'TEST':
            print('"mode" can only be TRAIN or TEST.')
            raise SystemExit

        # create input & output symbols
        self.learning_rate = T.fscalar('learn_rate')
        if self.minibatch == 1:
            self.x = T.fmatrix('x')
            # self.y = T.imatrix('y')
            self.y = T.itensor3('y')
        else: # for minibatch
            self.x = T.tensor3('x', dtype=dtype)
            self.y = T.tensor3('y', dtype=dtype)

        # training type
        print 'recurrent type is: < %s >' % self.recur_type
        # single direction
        if self.recur_type == 'RNN':
            self.model = RNN(input=self.x, input_dim=self.n_input, minibatch=self.minibatch,
                             hidden_dim=self.n_hidden, output_dim=self.n_output, params=self.model_params)
        elif self.recur_type == 'GRU':
            self.model = GRU(input=self.x, input_dim=self.n_input, hidden_dim=self.n_hidden,
                             output_dim=self.n_output, params=self.model_params)
        elif self.recur_type == 'LSTM':
            self.model = LSTM(input=self.x, input_dim=self.n_input, minibatch=self.minibatch,
                              hidden_dim=self.n_hidden, output_dim=self.n_output, params=self.model_params)
        # bi-direction
        elif self.recur_type == 'BiRNN':
            self.model = BiRNN(input=self.x, input_dim=self.n_input, minibatch=self.minibatch,
                               hidden_dim=self.n_hidden, output_dim=self.n_output, params=self.model_params)
        elif self.recur_type == 'BiGRU':
            self.model = BiGRU(input=self.x, input_dim=self.n_input, hidden_dim=self.n_hidden,
                               output_dim=self.n_output, params=self.model_params)
        elif self.recur_type == 'BiLSTM':
            self.model = BiLSTM(input=self.x, input_dim=self.n_input, minibatch=self.minibatch,
                                hidden_dim=self.n_hidden, output_dim=self.n_output, params=self.model_params)
        elif self.recur_type == 'VanStackBiLSTM':
            self.model = VanillaStack_BiLSTM(input=self.x, input_dim=self.n_input, minibatch=self.minibatch,
                                             hidden_dim=self.n_hidden, output_dim=self.n_output, params=self.model_params)
        else:
            print('Only supports : RNN, GRU, LSTM, BiRNN, BiGRU, BiLSTM \n')
            raise SystemExit

        # calculate cost
        # cost = self.model.meansquare_err(self.y)
        # cost = self.model.mse2consist_err(self.y)
        cost = self.model.cross_entropy_err(self.y)

        if mode == 'TRAIN':
            # update parameters
            updates = get_optimizer(self.optimizer, cost, self.model.params, self.learning_rate)

            # learning model function
            self.recur_learn_fun = theano.function(inputs=[self.x, self.y, self.learning_rate], outputs=cost,
                                                   updates=updates, allow_input_downcast=True)

        # model prediction function
        self.predict_fun = theano.function(inputs=[self.x], outputs=self.model.y_label, allow_input_downcast=True)
        self.validate_fun = theano.function(inputs=[self.x, self.y], outputs=cost, allow_input_downcast=True)

        # self.contextMat = T.jacobian(T.sum(self.model.y_t[:,0,:], axis=2), T.sum(self.x[:,0,:], axis=2))
        # self.context_fun = theano.function(inputs=[self.x], outputs=self.contextMat, allow_input_downcast=True)
        print '# success!'

    ############################################
    # Train Recurrent Model #
    ############################################
    def train_recur_model(self, train_set_x, train_set_y):
        print('### training model -- %s ' % self.model_file)

        seq_n = train_set_x.shape[1] # seq_len, seq_n, seq_width
        d_range = np.arange(seq_n)
        # train cost
        self.costs = [None]*self.train_epochs
        min_train_err = np.inf

        ''' training begin '''
        for ep in range(self.train_epochs):
            print 'epoch: %d' % ep,
            train_s = time.time()
            ep_error = 0
            # randomize sample index
            np.random.shuffle(d_range)
            n_batch = seq_n / self.minibatch
            for b in xrange(n_batch):
                p_idx = d_range[b*self.minibatch:(b+1)*self.minibatch]

                batch_x = train_set_x[:, p_idx, :]
                batch_y = train_set_y[:, p_idx, :]
                # core call #
                train_cost = self.recur_learn_fun(batch_x, batch_y)
                ep_error += train_cost

            ep_error = ep_error/n_batch
            self.costs[ep] = ep_error
            train_e = time.time()
            print ', time: %f, error: %.5f' % (train_e - train_s, ep_error)

            # save the current best model
            if ep_error < min_train_err:
                min_train_err = ep_error
                # save model and info.
                self.save_recur_model()

            # early stopping #
            err_R = self.costs[ep] / self.costs[0]
            if err_R < self.earlyErr_T:
                print 'early stopping is triggered.'
                break

        # plot training curve
        self.plot_error_curve()

    ##############################################
    # Train Recurrent Model with online sampling #
    ##############################################
    def train_recur_model_online_sample(self, dict_s, s):

        print('### training model -- %s ' % self.model_file)
        # train cost
        self.costs = [None] * self.train_epochs
        min_train_err = np.inf

        seq_n = dict_s['img_n']
        d_range = np.arange(seq_n)
        ''' training begin '''
        for ep in range(self.train_epochs):
            print 'epoch: %d' % ep,
            train_s = time.time()
            ep_error = 0
            # randomize sample index
            np.random.shuffle(d_range)
            n_batch = seq_n / self.minibatch
            print '\n'
            for b in xrange(n_batch):
                print 'batch No. %d...' % b
                # p_idx = d_range[b * self.minibatch:(b + 1) * self.minibatch]
                p_idx = d_range[b]
                # load data on-the-fly
                # x, y = vols_serialization(dict_s, p_idx)
                x, y = vols_serialization_bbx(dict_s, p_idx)
                # core call #
                train_cost = self.recur_learn_fun(x, y)
                ep_error += train_cost

            ep_error = ep_error / n_batch
            self.costs[ep] = ep_error
            train_e = time.time()
            print ', time: %f, error: %.5f' % (train_e - train_s, ep_error)

            # save the current best model
            if ep_error < min_train_err:
                min_train_err = ep_error
                # save model and info.
                self.save_recur_model()

            # early stopping #
            err_R = self.costs[ep] / self.costs[0]
            if err_R < self.earlyErr_T:
                print 'early stopping is triggered.'
                break

        # plot training curve
        self.plot_error_curve()

    ##############################################
    # Train Recurrent Model with online sampling #
    ##############################################
    def train_recur_model_online_subgrid_sample(self, dict_s, s):

        print('### training model -- %s ' % self.model_file)
        # train cost
        self.costs = [0.0] * self.train_epochs
        min_train_err = np.inf
        learning_rate = dict_s['learning_rate']

        seq_n = dict_s['img_n']
        d_range = np.arange(seq_n)
        ''' training begin '''
        for ep in range(self.train_epochs):
            # decrease learning rate with epoch
            if ep!=0 and np.mod(ep, dict_s['lr_interval']) ==0:
                learning_rate = learning_rate / dict_s['lr_ita']

            print '\nepoch: %d, learning rate: %.6f' % (ep, learning_rate)
            train_s = time.time()
            ep_error = 0
            # randomize sample index
            np.random.shuffle(d_range)
            n_batch = seq_n / self.minibatch
            for b in xrange(n_batch):
                print 'batch No. %d...' % b
                # p_idx = d_range[b * self.minibatch:(b + 1) * self.minibatch]
                p_idx = d_range[b]
                # load data on-the-fly
                # x, y = vols_serialization(dict_s, p_idx)
                x, y = vols_serialization_subgrid(dict_s, p_idx)
                # core call #
                train_cost = self.recur_learn_fun(x, y, learning_rate)
                ep_error += train_cost

            ep_error = ep_error / n_batch
            self.costs[ep] = ep_error
            train_e = time.time()
            print ', time: %f, error: %.5f, ds_err: %.5f' % (train_e - train_s, ep_error, ep_error/ds_N)

            # record error
            with open(self.err_txt, 'wb') as err_fid:
                np.savetxt(err_fid, np.asarray(self.costs) / ds_N, fmt='%s')

            # save model and info.
            if ep!=0 and np.mod(ep, 3) == 0:
                self.save_recur_model()

        # plot training curve
        self.plot_error_curve()


    ########################
    # Save Recurrent Model #
    ########################
    def save_recur_model(self):
        # save basic network info.
        with open(self.info_file, 'wb') as f:
            pkl.dump(self.basic_info, f, pkl.HIGHEST_PROTOCOL)
        # save trained network parameters
        with open(self.model_file, 'wb') as f:
            pkl.dump(self.model.params, f, pkl.HIGHEST_PROTOCOL)

    #############################
    # Plot Training Error Curve #
    #############################
    def plot_error_curve(self):
        plt.title('%s' % self.model_file)
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.plot(np.arange(self.train_epochs), np.asarray(self.costs) / ds_N, 'b-')
        plt.grid()
        plt.savefig('error-plot.png')
        plt.show()
        plt.close()

    ########################
    # Test Recurrent Model #
    ########################
    def test_recur_model(self, feat_mat):
        y_label = self.predict_fun(feat_mat)
        return y_label

