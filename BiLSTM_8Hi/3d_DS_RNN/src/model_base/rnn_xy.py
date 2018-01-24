import theano
import theano.tensor as T
import numpy as np

from utility.initial import get


__author__ = 'xinyang'


# RNN class
class RNN(object):

    def __init__(self, input, input_dim, minibatch, hidden_dim, output_dim,
                 init='uniform', inner_init='orthonormal', params=None):
        self.input = input
        self.hidden_dim = hidden_dim

        # create tuning parameters or use existing ones
        if params is None:
            self.W = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                   name='W', borrow=True)
            self.U = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                   name='U', borrow=True)
            self.bh = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                    name='bh', borrow=True)

            self.V = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)),
                                   name='V', borrow=True)
            self.by = theano.shared(value=get(identifier='zero', shape=(output_dim, )),
                                    name='by', borrow=True)
        elif params is not None:
            self.W, self.U, self.bh, self.V, self.by = params

        # parameter list
        self.params = [self.W, self.U, self.bh, self.V, self.by]

        # initialize hidden state
        if minibatch == 1:
            self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        else:
            self.h0 = theano.shared(value=get(identifier='zero', shape=(minibatch, hidden_dim)), name='h0', borrow=True)

        def recurrence(x_t, h_tm_prev):
            # hidden state
            h_t = T.tanh(T.dot(x_t, self.W) + T.dot(h_tm_prev, self.U) + self.bh)
            # output
            y_t = T.nnet.sigmoid(T.dot(h_t, self.V) + self.by)
            return h_t, y_t

        ### recurrent propagation ###
        [self.h_t, self.y_t], _ = theano.scan(recurrence, sequences=input, outputs_info=[self.h0, None])

    # cost measure
    def meansquare_err(self, y):
        mse = T.mean(T.pow(y - self.y_t, 2))
        return mse

### Bidirectional RNN class ###
class BiRNN(object):

    def __init__(self, input, input_dim, minibatch, hidden_dim, output_dim, params=None):
        self.in_fwd = input
        self.in_bwd = input[::-1]
        self.hidden_dim = hidden_dim

        # create tuning parameters or use existing ones
        if params is None:
            self.fwd_rnn = RNN(input=self.in_fwd, input_dim=input_dim, minibatch=minibatch,
                               hidden_dim=hidden_dim, output_dim=output_dim)
            self.bwd_rnn = RNN(input=self.in_bwd, input_dim=input_dim, minibatch=minibatch,
                               hidden_dim=hidden_dim, output_dim=output_dim)
            self.V_fwd = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                                       name='V_f', borrow=True)
            self.V_bwd = theano.shared(value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                                       name='V_b', borrow=True)
            self.by = theano.shared(value=get('zero', shape=(output_dim, )),
                                    name='by', borrow=True)
        elif params is not None:
            [fwd_rnn_W, fwd_rnn_U, fwd_rnn_bh, bwd_rnn_W, bwd_rnn_U, bwd_rnn_bh, V_fwd, V_bwd, by] = params
            void_M = theano.shared(value=np.zeros(1))

            fwd_param = [fwd_rnn_W, fwd_rnn_U, fwd_rnn_bh, void_M, void_M]
            bwd_param = [bwd_rnn_W, bwd_rnn_U, bwd_rnn_bh, void_M, void_M]

            self.fwd_rnn = RNN(input=self.in_fwd, input_dim=input_dim, minibatch=minibatch,
                               hidden_dim=hidden_dim, output_dim=output_dim, params=fwd_param)
            self.bwd_rnn = RNN(input=self.in_bwd, input_dim=input_dim, minibatch=minibatch,
                               hidden_dim=hidden_dim, output_dim=output_dim, params=bwd_param)
            self.V_fwd = V_fwd
            self.V_bwd = V_bwd
            self.by = by

        # print '#########################'
        # print np.asarray(self.fwd_rnn.W.eval())

        # parameter list
        self.params = [self.fwd_rnn.W, self.fwd_rnn.U, self.fwd_rnn.bh,
                       self.bwd_rnn.W, self.bwd_rnn.U, self.bwd_rnn.bh,
                       self.V_fwd, self.V_bwd, self.by]

        self.bwd_rnn.h_t = self.bwd_rnn.h_t[::-1]
        # weighted sum of forward & backward
        self.y_t = T.nnet.sigmoid(T.dot(self.fwd_rnn.h_t, self.V_fwd) + T.dot(self.bwd_rnn.h_t, self.V_bwd) + self.by)

    # cost measure
    def meansquare_err(self, y):
        mse = T.mean(T.pow(y - self.y_t, 2))
        return mse