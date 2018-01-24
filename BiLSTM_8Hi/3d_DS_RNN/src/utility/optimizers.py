import theano
import theano.tensor as T

import numpy as np

__author__ = 'uyaseen'

grad_clip = 5


# source: https://gist.github.com/Newmu/acb738767acb4788bac3
def adam(cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.dtype(theano.config.floatX).type(1))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        g = T.clip(g, -grad_clip, grad_clip)
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        g = T.clip(g, -grad_clip, grad_clip)
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def sgd_vanilla(cost, params, lr=0.001):
    grads = T.grad(cost, params)
    updates = [(param, param - lr * T.clip(grad, -grad_clip, grad_clip))
               for param, grad in zip(params, grads)
               ]
    return updates


def get_optimizer(identifier, cost, params, learning_rate):
    if identifier == 'adam':
        return adam(cost, params, lr=learning_rate)
    elif identifier == 'rmsprop':
        return rmsprop(cost, params, lr=learning_rate)
    elif identifier == 'sgd_vanilla':
        return sgd_vanilla(cost, params, lr=learning_rate)
    else:
        return NotImplementedError
