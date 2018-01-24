import theano
import theano.tensor as T

from utility.initial import get

__author__ = 'uyaseen'


class GRU(object):
    def __init__(self, input, input_dim, hidden_dim, output_dim, init='uniform',
                 inner_init='orthonormal', inner_activation=T.nnet.hard_sigmoid,
                 activation=T.tanh, params=None):
        self.input = input
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.inner_activation = inner_activation
        if params is None:
            # update gate
            self.W_z = theano.shared(
                value=get(identifier=init, shape=(input_dim, hidden_dim)),
                name='W_z',
                borrow=True)
            self.U_z = theano.shared(
                value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                name='U_z',
                borrow=True)
            self.b_z = theano.shared(
                value=get(identifier='zero', shape=(hidden_dim,)),
                name='b_z',
                borrow=True)
            # reset gate
            self.W_r = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                     name='W_r',
                                     borrow=True)
            self.U_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_r',
                                     borrow=True)
            self.b_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_r',
                                     borrow=True)
            # weights pertaining to input, hidden & output neurons (externally)
            self.W = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                   name='W',
                                   borrow=True)
            self.U = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                   name='U',
                                   borrow=True)
            self.V = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)),
                                   name='V',
                                   borrow=True)
            self.b_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_h',
                                     borrow=True)
            self.b_y = theano.shared(value=get(identifier='zero', shape=(output_dim,)),
                                     name='b_y',
                                     borrow=True)
        else:
            self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, \
                self.W, self.U, self.V, self.b_h, self.b_y = params

        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.params = [self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W, self.U, self.V,
                       self.b_h, self.b_y]

        def recurrence(x_t, h_tm_prev):
            x_z = T.dot(x_t, self.W_z) + self.b_z
            x_r = T.dot(x_t, self.W_r) + self.b_r
            x_h = T.dot(x_t, self.W) + self.b_h

            z_t = inner_activation(x_z + T.dot(h_tm_prev, self.U_z))
            r_t = inner_activation(x_r + T.dot(h_tm_prev, self.U_r))
            hh_t = activation(x_h + T.dot(r_t * h_tm_prev, self.U))
            h_t = (T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm_prev

            y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b_y)

            return h_t, y_t[0]

        [self.h_t, self.y_t], _ = theano.scan(
            recurrence,
            sequences=self.input,
            outputs_info=[self.h0, None]
        )

        self.y = T.argmax(self.y_t, axis=1)

    def cross_entropy(self, y):
        return T.sum(T.nnet.categorical_crossentropy(self.y_t, y))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_t)[:, y])

    def errors(self, y):
        return T.mean(T.neq(self.y, y))

    # TODO: Find a better way of sampling.
    def generative_sampling(self, seed, emb_data, sample_length):
        fruit = theano.shared(value=seed)

        def step(h_tm, y_tm):

            x_z = T.dot(emb_data[y_tm], self.W_z) + self.b_z
            x_r = T.dot(emb_data[y_tm], self.W_r) + self.b_r
            x_h = T.dot(emb_data[y_tm], self.W) + self.b_h

            z_t = self.inner_activation(x_z + T.dot(h_tm, self.U_z))
            r_t = self.inner_activation(x_r + T.dot(h_tm, self.U_r))
            hh_t = self.activation(x_h + T.dot(r_t * h_tm, self.U))
            h_t = (T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm

            y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b_y)
            y = T.argmax(y_t, axis=1)

            return h_t, y[0]

        [_, samples], _ = theano.scan(fn=step,
                                      outputs_info=[self.h0, fruit],
                                      n_steps=sample_length)

        get_samples = theano.function(inputs=[],
                                      outputs=samples)

        return get_samples()


class BiGRU(object):
    def __init__(self, input, input_dim, hidden_dim, output_dim,
                 params=None):
        self.input_f = input
        self.input_b = input[::-1]
        if params is None:
            self.fwd_gru = GRU(input=self.input_f, input_dim=input_dim, hidden_dim=hidden_dim,
                               output_dim=output_dim)
            self.bwd_gru = GRU(input=self.input_b, input_dim=input_dim, hidden_dim=hidden_dim,
                               output_dim=output_dim)
            self.V_f = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='V_f',
                borrow=True
            )
            self.V_b = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='V_b',
                borrow=True
            )
            self.by = theano.shared(
                value=get('zero', shape=(output_dim,)),
                name='by',
                borrow=True)

        else:
            # To support loading from persistent storage, the current implementation of Gru() will require a
            # change and is therefore not supported.
            # An elegant way would be to implement BiGru() without using Gru() [is a trivial thing to do].
            raise NotImplementedError

        # since now bigru is doing the actual classification ; we don't need 'Gru().V & Gru().by' as they
        # are not part of computational graph (separate logistic-regression unit/layer is probably the best way to
        # handle this). Here's the ugly workaround -_-
        self.params = [self.fwd_gru.W_z, self.fwd_gru.U_z, self.fwd_gru.b_z,
                       self.fwd_gru.W_r, self.fwd_gru.U_r, self.fwd_gru.b_r,
                       self.fwd_gru.W, self.fwd_gru.U, self.fwd_gru.b_h,

                       self.bwd_gru.W_z, self.bwd_gru.U_z, self.bwd_gru.b_z,
                       self.bwd_gru.W_r, self.bwd_gru.U_r, self.bwd_gru.b_r,
                       self.bwd_gru.W, self.bwd_gru.U, self.bwd_gru.b_h,

                       self.V_f, self.V_b, self.by]

        self.bwd_gru.h_t = self.bwd_gru.h_t[::-1]
        # Take the weighted sum of forward & backward gru's hidden representation
        self.h_t = T.dot(self.fwd_gru.h_t, self.V_f) + T.dot(self.bwd_gru.h_t, self.V_b)

        self.y_t = T.nnet.softmax(self.h_t + self.by)
        self.y = T.argmax(self.y_t, axis=1)

    def cross_entropy(self, y):
        return T.sum(T.nnet.categorical_crossentropy(self.y_t, y))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_t)[:, y])

    def errors(self, y):
        return T.mean(T.neq(self.y, y))

    # TODO: Find a way of sampling (running forward + backward gru manually is really ugly and therefore, avoided).
    def generative_sampling(self, seed, emb_data, sample_length):
        return NotImplementedError
