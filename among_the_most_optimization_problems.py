import numpy as np
from scipy import optimize as op


def fun(*args):

    x, y = args[0]

    # log_likelihood =
    # return  - log_likelihood
    return x**2 + y*x


print(op.minimize(fun=fun, x0=np.array([10**4, 10**3]), bounds=[(-np.inf, np.inf), ] * 2))

