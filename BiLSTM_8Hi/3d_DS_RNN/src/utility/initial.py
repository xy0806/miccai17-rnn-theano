import theano
import numpy as np

__author__ = 'uyaseen'


def uniform(shape, scale=0.08):
    return np.asarray(np.random.uniform(
                    low=-scale, high=scale,
                    size=shape),
                    dtype=theano.config.floatX)


def zero(shape):
    return np.asarray(np.zeros(shape), dtype=theano.config.floatX)


def one(shape):
    return np.asarray(np.ones(shape), dtype=theano.config.floatX)


def eye(rows):
    return np.asarray(np.eye(rows), dtype=theano.config.floatX)


def orthogonal(shape, scale=1.1, name=None):
    # From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]


def get(identifier, shape):
    # seed = 36
    # np.random.seed(seed)
    if identifier == 'uniform':
        return uniform(shape)
    elif identifier == 'orthonormal':
        return np.asarray(orthogonal(shape), dtype=theano.config.floatX)
    elif identifier == 'zero':
        return zero(shape)
    elif identifier == 'one':
        return one(shape)
    elif identifier == 'eye':
        return eye(shape[0])
    else:
        raise NotImplementedError