import theano
import theano.tensor as T
import numpy as np

from utility.initial import get


__author__ = 'xinyang'


class LSTM(object):
    def __init__(self, input, input_dim, minibatch, hidden_dim, output_dim, init='uniform',
                 inner_init='orthonormal', gate_act=T.nnet.sigmoid,
                 tanh_act=T.tanh, params=None):
        self.input = input
        self.gate_act = gate_act
        self.activation = tanh_act

        # create tuning parameters or use existing ones
        if params is None:
            # input gate
            self.W_i = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)), name='W_i', borrow=True)
            self.U_i = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)), name='U_i', borrow=True)
            self.b_i = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='b_i', borrow=True)
            # forget gate
            self.W_f = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)), name='W_f', borrow=True)
            self.U_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)), name='U_f', borrow=True)
            self.b_f = theano.shared(value=get(identifier='one', shape=(hidden_dim, )), name='b_f', borrow=True)
            # output gate
            self.W_o = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)), name='W_o', borrow=True)
            self.U_o = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)), name='U_o', borrow=True)
            self.b_o = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='b_o', borrow=True)
            # memory
            self.W_c = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)), name='W_c', borrow=True)
            self.U_c = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)), name='U_c', borrow=True)
            self.b_c = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='b_c', borrow=True)
            # weights to output neuron
            self.V_0 = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)), name='V_0', borrow=True)
            self.b_y_0 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='b_y_0', borrow=True)
            self.V_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)), name='V_1', borrow=True)
            self.b_y_1 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='b_y_1', borrow=True)
            self.V_2 = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)), name='V_2', borrow=True)
            self.b_y_2 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='b_y_2', borrow=True)
            self.V_3 = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)), name='V_3', borrow=True)
            self.b_y_3 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='b_y_3', borrow=True)

        elif params is not None:
                [self.W_i, self.U_i, self.b_i,
                 self.W_f, self.U_f, self.b_f,
                 self.W_o, self.U_o, self.b_o,
                 self.W_c, self.U_c, self.b_c,
                 self.V_0, self.b_y_0,
                 self.V_1, self.b_y_1,
                 self.V_2, self.b_y_2,
                 self.V_3, self.b_y_3] = params

        # parameter list
        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o,
                       self.W_c, self.U_c, self.b_c,
                       self.V_0, self.b_y_0,
                       self.V_1, self.b_y_1,
                       self.V_2, self.b_y_2,
                       self.V_3, self.b_y_3]

        # initialize internal and hidden state
        if minibatch == 1:
            self.c0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='c0', borrow=True)
            self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        else:
            self.c0 = theano.shared(value=get(identifier='zero', shape=(minibatch, hidden_dim)), name='c0', borrow=True)
            self.h0 = theano.shared(value=get(identifier='zero', shape=(minibatch, hidden_dim)), name='h0', borrow=True)

        def recurrence(x_t, c_tm_prev, h_tm_prev):
            i_t = gate_act(T.dot(x_t, self.W_i) + T.dot(h_tm_prev, self.U_i) + self.b_i)
            f_t = gate_act(T.dot(x_t, self.W_f) + T.dot(h_tm_prev, self.U_f) + self.b_f)
            o_t = gate_act(T.dot(x_t, self.W_o) + T.dot(h_tm_prev, self.U_o) + self.b_o)
            # internal memory
            x_c = T.dot(x_t, self.W_c) + self.b_c
            c_t = f_t * c_tm_prev + i_t * tanh_act(x_c + T.dot(h_tm_prev, self.U_c))
            # hidden state
            h_t = o_t * tanh_act(c_t)
            # output
            # y_t_0 = T.nnet.sigmoid(T.dot(h_t, self.V_0) + self.b_y_0)
            # y_t_1 = T.nnet.sigmoid(T.dot(h_t, self.V_1) + self.b_y_1)
            # y_t_2 = T.nnet.sigmoid(T.dot(h_t, self.V_2) + self.b_y_2)
            # y_t_3 = T.nnet.sigmoid(T.dot(h_t, self.V_3) + self.b_y_3)

            y_t_0 = T.dot(h_t, self.V_0) + self.b_y_0
            y_t_1 = T.dot(h_t, self.V_1) + self.b_y_1
            y_t_2 = T.dot(h_t, self.V_2) + self.b_y_2
            y_t_3 = T.dot(h_t, self.V_3) + self.b_y_3

            y_t = T.stack([y_t_0, y_t_1, y_t_2, y_t_3], axis=1)
            # softmax
            y_t = T.nnet.softmax(y_t)
            # class label with maximum probability
            y_label = T.argmax(y_t, axis=1)

            return c_t, h_t, y_t, y_label

        [self.c_t, self.h_t, self.y_t, self.y_label], _ = theano.scan(recurrence, sequences=self.input, outputs_info=[self.c0, self.h0, None, None])

    # cost measure
    def meansquare_err(self, y):
        print '=== using mse error. ==='
        mse = T.mean(T.pow(y - self.y_t, 2))
        # mse = T.mean(T.pow(self.y_t, 2))
        return mse

    def mse2consist_err(self, y):
        print '=== using mse2consist error. ==='
        # mean square error
        mse = T.mean(T.pow(y - self.y_t, 2))
        # consistency error
        cst_err = T.mean(T.pow(self.y_t - T.roll(self.y_t, shift=1, axis=0), 2))

        hybrid_err = 0.9 * mse + 0.1 * cst_err
        return hybrid_err

    def cross_entropy_err(self, y):
        print '=== using cross entropy. ==='
        # deeply supervised error
        T_len = y.shape[0]
        T_step = T_len/4

        crs_mat = T.log(self.y_t) * y

        err_0 = -T.mean(crs_mat[0:T_step, :, :])
        err_1 = -T.mean(crs_mat[0:2*T_step, :, :])
        err_2 = -T.mean(crs_mat[0:3*T_step, :, :])
        err_3 = -T.mean(crs_mat)
        err = err_0 + err_1 + err_2 + err_3

        # common error
        # err = -T.mean(T.log(self.y_t)*y)
        return err


class BiLSTM(object):
    def __init__(self, input, input_dim, minibatch, hidden_dim, output_dim, params=None):
        self.in_fwd = input
        self.in_bwd = input[::-1]

        ## create tuning parameters or use existing ones
        if params is None:
            self.fwd_lstm = LSTM(input=self.in_fwd, input_dim=input_dim, minibatch=minibatch,
                                 hidden_dim=hidden_dim, output_dim=output_dim)
            self.bwd_lstm = LSTM(input=self.in_bwd, input_dim=input_dim, minibatch=minibatch,
                                 hidden_dim=hidden_dim, output_dim=output_dim)
            # self.V_f = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_f', borrow=True)
            # self.V_b = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_b', borrow=True)
            # self.by = theano.shared(value=get('zero', shape=(output_dim, )), name='by', borrow=True)

            self.V_f_0 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_f_0', borrow=True)
            self.V_f_1 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_f_1', borrow=True)
            self.V_f_2 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_f_2', borrow=True)
            self.V_f_3 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_f_3', borrow=True)

            self.V_b_0 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_b_0', borrow=True)
            self.V_b_1 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_b_1', borrow=True)
            self.V_b_2 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_b_2', borrow=True)
            self.V_b_3 = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)), name='V_b_3', borrow=True)

            self.by_0 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='by_0', borrow=True)
            self.by_1 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='by_1', borrow=True)
            self.by_2 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='by_2', borrow=True)
            self.by_3 = theano.shared(value=get(identifier='zero', shape=(output_dim, )), name='by_3', borrow=True)

        elif params is not None:
            [fwd_lstm_Wi, fwd_lstm_Ui, fwd_lstm_bi,  fwd_lstm_Wf, fwd_lstm_Uf, fwd_lstm_bf,
             fwd_lstm_Wo, fwd_lstm_Uo, fwd_lstm_bo,  fwd_lstm_Wc, fwd_lstm_Uc, fwd_lstm_bc,
             bwd_lstm_Wi, bwd_lstm_Ui, bwd_lstm_bi,  bwd_lstm_Wf, bwd_lstm_Uf, bwd_lstm_bf,
             bwd_lstm_Wo, bwd_lstm_Uo, bwd_lstm_bo,  bwd_lstm_Wc, bwd_lstm_Uc, bwd_lstm_bc,
             V_f_0, V_f_1, V_f_2, V_f_3,
             V_b_0, V_b_1, V_b_2, V_b_3,
             by_0, by_1, by_2, by_3] = params

            void_M = theano.shared(value=np.zeros(1))

            fwd_param = [fwd_lstm_Wi, fwd_lstm_Ui, fwd_lstm_bi,  fwd_lstm_Wf, fwd_lstm_Uf, fwd_lstm_bf,
                         fwd_lstm_Wo, fwd_lstm_Uo, fwd_lstm_bo,  fwd_lstm_Wc, fwd_lstm_Uc, fwd_lstm_bc,
                         void_M, void_M, void_M, void_M, void_M, void_M, void_M, void_M]
            bwd_param = [bwd_lstm_Wi, bwd_lstm_Ui, bwd_lstm_bi,  bwd_lstm_Wf, bwd_lstm_Uf, bwd_lstm_bf,
                         bwd_lstm_Wo, bwd_lstm_Uo, bwd_lstm_bo,  bwd_lstm_Wc, bwd_lstm_Uc, bwd_lstm_bc,
                         void_M, void_M, void_M, void_M, void_M, void_M, void_M, void_M]

            self.fwd_lstm = LSTM(input=self.in_fwd, input_dim=input_dim, minibatch=minibatch,hidden_dim=hidden_dim,
                                 output_dim=output_dim, params=fwd_param)
            self.bwd_lstm = LSTM(input=self.in_bwd, input_dim=input_dim, minibatch=minibatch,hidden_dim=hidden_dim,
                                 output_dim=output_dim, params=bwd_param)
            self.V_f_0 = V_f_0
            self.V_f_1 = V_f_1
            self.V_f_2 = V_f_2
            self.V_f_3 = V_f_3

            self.V_b_0 = V_b_0
            self.V_b_1 = V_b_1
            self.V_b_2 = V_b_2
            self.V_b_3 = V_b_3

            self.by_0 = by_0
            self.by_1 = by_1
            self.by_2 = by_2
            self.by_3 = by_3

        # parameter list
        self.params = [self.fwd_lstm.W_i, self.fwd_lstm.U_i, self.fwd_lstm.b_i,
                       self.fwd_lstm.W_f, self.fwd_lstm.U_f, self.fwd_lstm.b_f,
                       self.fwd_lstm.W_o, self.fwd_lstm.U_o, self.fwd_lstm.b_o,
                       self.fwd_lstm.W_c, self.fwd_lstm.U_c, self.fwd_lstm.b_c,

                       self.bwd_lstm.W_i, self.bwd_lstm.U_i, self.bwd_lstm.b_i,
                       self.bwd_lstm.W_f, self.bwd_lstm.U_f, self.bwd_lstm.b_f,
                       self.bwd_lstm.W_o, self.bwd_lstm.U_o, self.bwd_lstm.b_o,
                       self.bwd_lstm.W_c, self.bwd_lstm.U_c, self.bwd_lstm.b_c,

                       self.V_f_0, self.V_f_1, self.V_f_2, self.V_f_3,
                       self.V_b_0, self.V_b_1, self.V_b_2, self.V_b_3,
                       self.by_0, self.by_1, self.by_2, self.by_3]

        self.bwd_lstm.h_t = self.bwd_lstm.h_t[::-1]
        # weighted sum of forward & backward
        # self.y_t = T.nnet.sigmoid(T.dot(self.fwd_lstm.h_t, self.V_f) + T.dot(self.bwd_lstm.h_t, self.V_b) + self.by)

        # self.y_t_0 = T.nnet.sigmoid(T.dot(self.fwd_lstm.h_t, self.V_f_0) + T.dot(self.bwd_lstm.h_t, self.V_b_0) + self.by_0)
        # self.y_t_1 = T.nnet.sigmoid(T.dot(self.fwd_lstm.h_t, self.V_f_1) + T.dot(self.bwd_lstm.h_t, self.V_b_1) + self.by_1)
        # self.y_t_2 = T.nnet.sigmoid(T.dot(self.fwd_lstm.h_t, self.V_f_2) + T.dot(self.bwd_lstm.h_t, self.V_b_2) + self.by_2)
        # self.y_t_3 = T.nnet.sigmoid(T.dot(self.fwd_lstm.h_t, self.V_f_3) + T.dot(self.bwd_lstm.h_t, self.V_b_3) + self.by_3)

        self.y_t_0 = T.dot(self.fwd_lstm.h_t, self.V_f_0) + T.dot(self.bwd_lstm.h_t, self.V_b_0) + self.by_0
        self.y_t_1 = T.dot(self.fwd_lstm.h_t, self.V_f_1) + T.dot(self.bwd_lstm.h_t, self.V_b_1) + self.by_1
        self.y_t_2 = T.dot(self.fwd_lstm.h_t, self.V_f_2) + T.dot(self.bwd_lstm.h_t, self.V_b_2) + self.by_2
        self.y_t_3 = T.dot(self.fwd_lstm.h_t, self.V_f_3) + T.dot(self.bwd_lstm.h_t, self.V_b_3) + self.by_3

        self.y_temp = T.stack([self.y_t_0, self.y_t_1, self.y_t_2, self.y_t_3], axis=2)
        self.y_t = T.reshape(self.y_temp, [self.y_temp.shape[0]*self.y_temp.shape[1], self.y_temp.shape[2]])
        # softmax
        self.y_t = T.nnet.softmax(self.y_t)
        # class label with maximum probability
        self.y_label = T.argmax(self.y_t, axis=1)

        self.y_t = T.reshape(self.y_t, [self.y_temp.shape[0], self.y_temp.shape[1], self.y_temp.shape[2]])
        self.y_label = T.reshape(self.y_label, [self.y_temp.shape[0], self.y_temp.shape[1]])

    # cost measure
    def meansquare_err(self, y):
        print '=== using mse error. ==='
        mse = T.mean(T.pow(y - self.y_t, 2))
        return mse

    def mse2consist_err(self, y):
        print '=== using mse2consist error. ==='
        # mean square error
        mse = T.mean(T.pow(y - self.y_t, 2))
        # consistency error
        cst_err = T.mean(T.pow(self.y_t - T.roll(self.y_t, shift=1, axis=0), 2))

        hybrid_err = 0.9*mse + 0.1*cst_err
        return hybrid_err

    def cross_entropy_err(self, y):
        print '=== using cross entropy. ==='
        T_len = y.shape[0]
        T_step = T_len/8

        crs_mat = T.log(self.y_t) * y

        err_0 = -T.mean(crs_mat[0:T_step, :, :])
        err_1 = -T.mean(crs_mat[0:2*T_step, :, :])
        err_2 = -T.mean(crs_mat[0:3*T_step, :, :])
        err_3 = -T.mean(crs_mat[0:4*T_step, :, :])
        err_4 = -T.mean(crs_mat[0:5*T_step, :, :])
        err_5 = -T.mean(crs_mat[0:6*T_step, :, :])
        err_6 = -T.mean(crs_mat[0:7 * T_step, :, :])
        err_7 = -T.mean(crs_mat)
        err = err_0 + err_1 + err_2 + err_3 + err_4 + err_5 + err_6 + err_7

        # deeply supervised error
        # T_len = y.shape[0]
        # T_step = T_len/4
        #
        # crs_mat = T.log(self.y_t) * y
        #
        # err_0 = -T.mean(crs_mat[0:T_step, :, :])
        # err_1 = -T.mean(crs_mat[0:2*T_step, :, :])
        # err_2 = -T.mean(crs_mat[0:3*T_step, :, :])
        # err_3 = -T.mean(crs_mat)
        # err = err_0 + err_1 + err_2 + err_3

        # simple supervision
        # err = -T.mean(T.log(self.y_t)*y)

        return err


##########################
# Vanilla Stacked BiLSTM #
##########################
class VanillaStack_BiLSTM(object):

    def __init__(self, input, input_dim, minibatch, hidden_dim, output_dim, params=None):

        if params is None:
            # layer 1
            self.input_L1 = input
            self.blstm_L1 = BiLSTM(input=self.input_L1, input_dim=input_dim, minibatch=minibatch,
                                hidden_dim=hidden_dim, output_dim=output_dim, params=None)
            # layer 2
            self.input_L2 = self.blstm_L1.y_t
            self.blstm_L2 = BiLSTM(input=self.input_L2, input_dim=input_dim, minibatch=minibatch,
                                hidden_dim=hidden_dim, output_dim=output_dim, params=None)
            # layer 3
            self.input_L3 = self.blstm_L2.y_t
            self.blstm_L3 = BiLSTM(input=self.input_L3, input_dim=input_dim, minibatch=minibatch,
                                hidden_dim=hidden_dim, output_dim=output_dim, params=None)
            # output
            self.y_t =self.blstm_L3.y_t

        elif params is not None:
            [L1_fwd_W_i, L1_fwd_U_i, L1_fwd_b_i,  L1_fwd_W_f, L1_fwd_U_f, L1_fwd_b_f,
             L1_fwd_W_o, L1_fwd_U_o, L1_fwd_b_o,  L1_fwd_W_c, L1_fwd_U_c, L1_fwd_b_c,
             L1_bwd_W_i, L1_bwd_U_i, L1_bwd_b_i,  L1_bwd_W_f, L1_bwd_U_f, L1_bwd_b_f,
             L1_bwd_W_o, L1_bwd_U_o, L1_bwd_b_o,  L1_bwd_W_c, L1_bwd_U_c, L1_bwd_b_c,
             L1_V_f, L1_V_b, L1_by,
             # layer 2
             L2_fwd_W_i, L2_fwd_U_i, L2_fwd_b_i,  L2_fwd_W_f, L2_fwd_U_f, L2_fwd_b_f,
             L2_fwd_W_o, L2_fwd_U_o, L2_fwd_b_o,  L2_fwd_W_c, L2_fwd_U_c, L2_fwd_b_c,
             L2_bwd_W_i, L2_bwd_U_i, L2_bwd_b_i,  L2_bwd_W_f, L2_bwd_U_f, L2_bwd_b_f,
             L2_bwd_W_o, L2_bwd_U_o, L2_bwd_b_o,  L2_bwd_W_c, L2_bwd_U_c, L2_bwd_b_c,
             L2_V_f, L2_V_b, L2_by,
             # layer 3
             L3_fwd_W_i, L3_fwd_U_i, L3_fwd_b_i,  L3_fwd_W_f, L3_fwd_U_f, L3_fwd_b_f,
             L3_fwd_W_o, L3_fwd_U_o, L3_fwd_b_o,  L3_fwd_W_c, L3_fwd_U_c, L3_fwd_b_c,
             L3_bwd_W_i, L3_bwd_U_i, L3_bwd_b_i,  L3_bwd_W_f, L3_bwd_U_f, L3_bwd_b_f,
             L3_bwd_W_o, L3_bwd_U_o, L3_bwd_b_o,  L3_bwd_W_c, L3_bwd_U_c, L3_bwd_b_c,
             L3_V_f, L3_V_b, L3_by] = params

            L1_param = [L1_fwd_W_i, L1_fwd_U_i, L1_fwd_b_i,  L1_fwd_W_f, L1_fwd_U_f, L1_fwd_b_f,
                        L1_fwd_W_o, L1_fwd_U_o, L1_fwd_b_o,  L1_fwd_W_c, L1_fwd_U_c, L1_fwd_b_c,
                        L1_bwd_W_i, L1_bwd_U_i, L1_bwd_b_i,  L1_bwd_W_f, L1_bwd_U_f, L1_bwd_b_f,
                        L1_bwd_W_o, L1_bwd_U_o, L1_bwd_b_o,  L1_bwd_W_c, L1_bwd_U_c, L1_bwd_b_c,
                        L1_V_f, L1_V_b, L1_by]

            L2_param = [L2_fwd_W_i, L2_fwd_U_i, L2_fwd_b_i,  L2_fwd_W_f, L2_fwd_U_f, L2_fwd_b_f,
                        L2_fwd_W_o, L2_fwd_U_o, L2_fwd_b_o,  L2_fwd_W_c, L2_fwd_U_c, L2_fwd_b_c,
                        L2_bwd_W_i, L2_bwd_U_i, L2_bwd_b_i,  L2_bwd_W_f, L2_bwd_U_f, L2_bwd_b_f,
                        L2_bwd_W_o, L2_bwd_U_o, L2_bwd_b_o,  L2_bwd_W_c, L2_bwd_U_c, L2_bwd_b_c,
                        L2_V_f, L2_V_b, L2_by]

            L3_param = [L3_fwd_W_i, L3_fwd_U_i, L3_fwd_b_i,  L3_fwd_W_f, L3_fwd_U_f, L3_fwd_b_f,
                        L3_fwd_W_o, L3_fwd_U_o, L3_fwd_b_o,  L3_fwd_W_c, L3_fwd_U_c, L3_fwd_b_c,
                        L3_bwd_W_i, L3_bwd_U_i, L3_bwd_b_i,  L3_bwd_W_f, L3_bwd_U_f, L3_bwd_b_f,
                        L3_bwd_W_o, L3_bwd_U_o, L3_bwd_b_o,  L3_bwd_W_c, L3_bwd_U_c, L3_bwd_b_c,
                        L3_V_f, L3_V_b, L3_by]

            # layer 1
            self.input_L1 = input
            self.blstm_L1 = BiLSTM(input=self.input_L1, input_dim=input_dim, minibatch=minibatch,
                                hidden_dim=hidden_dim, output_dim=output_dim, params=L1_param)
            # layer 2
            self.input_L2 = self.blstm_L1.y_t
            self.blstm_L2 = BiLSTM(input=self.input_L2, input_dim=input_dim, minibatch=minibatch,
                                hidden_dim=hidden_dim, output_dim=output_dim, params=L2_param)
            # layer 3
            self.input_L3 = self.blstm_L2.y_t
            self.blstm_L3 = BiLSTM(input=self.input_L3, input_dim=input_dim, minibatch=minibatch,
                                hidden_dim=hidden_dim, output_dim=output_dim, params=L3_param)
            # output
            self.y_t =self.blstm_L3.y_t

        # parameter list
        self.params = [self.blstm_L1.fwd_lstm.W_i, self.blstm_L1.fwd_lstm.U_i, self.blstm_L1.fwd_lstm.b_i,
                       self.blstm_L1.fwd_lstm.W_f, self.blstm_L1.fwd_lstm.U_f, self.blstm_L1.fwd_lstm.b_f,
                       self.blstm_L1.fwd_lstm.W_o, self.blstm_L1.fwd_lstm.U_o, self.blstm_L1.fwd_lstm.b_o,
                       self.blstm_L1.fwd_lstm.W_c, self.blstm_L1.fwd_lstm.U_c, self.blstm_L1.fwd_lstm.b_c,

                       self.blstm_L1.bwd_lstm.W_i, self.blstm_L1.bwd_lstm.U_i, self.blstm_L1.bwd_lstm.b_i,
                       self.blstm_L1.bwd_lstm.W_f, self.blstm_L1.bwd_lstm.U_f, self.blstm_L1.bwd_lstm.b_f,
                       self.blstm_L1.bwd_lstm.W_o, self.blstm_L1.bwd_lstm.U_o, self.blstm_L1.bwd_lstm.b_o,
                       self.blstm_L1.bwd_lstm.W_c, self.blstm_L1.bwd_lstm.U_c, self.blstm_L1.bwd_lstm.b_c,

                       self.blstm_L1.V_f, self.blstm_L1.V_b, self.blstm_L1.by,
                       # layer 2
                       self.blstm_L2.fwd_lstm.W_i, self.blstm_L2.fwd_lstm.U_i, self.blstm_L2.fwd_lstm.b_i,
                       self.blstm_L2.fwd_lstm.W_f, self.blstm_L2.fwd_lstm.U_f, self.blstm_L2.fwd_lstm.b_f,
                       self.blstm_L2.fwd_lstm.W_o, self.blstm_L2.fwd_lstm.U_o, self.blstm_L2.fwd_lstm.b_o,
                       self.blstm_L2.fwd_lstm.W_c, self.blstm_L2.fwd_lstm.U_c, self.blstm_L2.fwd_lstm.b_c,

                       self.blstm_L2.bwd_lstm.W_i, self.blstm_L2.bwd_lstm.U_i, self.blstm_L2.bwd_lstm.b_i,
                       self.blstm_L2.bwd_lstm.W_f, self.blstm_L2.bwd_lstm.U_f, self.blstm_L2.bwd_lstm.b_f,
                       self.blstm_L2.bwd_lstm.W_o, self.blstm_L2.bwd_lstm.U_o, self.blstm_L2.bwd_lstm.b_o,
                       self.blstm_L2.bwd_lstm.W_c, self.blstm_L2.bwd_lstm.U_c, self.blstm_L2.bwd_lstm.b_c,

                       self.blstm_L2.V_f, self.blstm_L2.V_b, self.blstm_L2.by,
                       # layer 3
                       self.blstm_L3.fwd_lstm.W_i, self.blstm_L3.fwd_lstm.U_i, self.blstm_L3.fwd_lstm.b_i,
                       self.blstm_L3.fwd_lstm.W_f, self.blstm_L3.fwd_lstm.U_f, self.blstm_L3.fwd_lstm.b_f,
                       self.blstm_L3.fwd_lstm.W_o, self.blstm_L3.fwd_lstm.U_o, self.blstm_L3.fwd_lstm.b_o,
                       self.blstm_L3.fwd_lstm.W_c, self.blstm_L3.fwd_lstm.U_c, self.blstm_L3.fwd_lstm.b_c,

                       self.blstm_L3.bwd_lstm.W_i, self.blstm_L3.bwd_lstm.U_i, self.blstm_L3.bwd_lstm.b_i,
                       self.blstm_L3.bwd_lstm.W_f, self.blstm_L3.bwd_lstm.U_f, self.blstm_L3.bwd_lstm.b_f,
                       self.blstm_L3.bwd_lstm.W_o, self.blstm_L3.bwd_lstm.U_o, self.blstm_L3.bwd_lstm.b_o,
                       self.blstm_L3.bwd_lstm.W_c, self.blstm_L3.bwd_lstm.U_c, self.blstm_L3.bwd_lstm.b_c,

                       self.blstm_L3.V_f, self.blstm_L3.V_b, self.blstm_L3.by]

    # cost measure
    def meansquare_err(self, y):
        mse = T.mean(T.pow(y - self.y_t, 2))
        return mse