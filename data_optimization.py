import numpy as np
from data_manager import import_data
from scipy import optimize as op


def fun(x):

    # log_likelihood =
    # return  - log_likelihood
    return x**2

data_for_subject_one = import_data()[0]


print(op.minimize(fun=fun, x0=np.array([10000000])))
