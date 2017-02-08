import numpy as np
from scipy import optimize as op


def fun(x):

    return x**2 + 19*x


print(op.minimize(fun=fun, x0=np.array([10**2]), bounds=[(-10**6, 10**6)]))
