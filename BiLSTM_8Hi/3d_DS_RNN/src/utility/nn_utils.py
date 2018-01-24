import theano.tensor as T


__author__ = 'uyaseen'


def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])
