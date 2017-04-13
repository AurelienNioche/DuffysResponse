from hyperopt import fmin, tpe, hp

# # ----- WITH SCIPY ---- #
# import numpy as np
# from scipy import optimize as op
# def fun(*args):
#
#     x, y = args[0]
#
#     # log_likelihood =
#     # return  - log_likelihood
#     return x**2 + y*x
#
#
# print(op.minimize(fun=fun, x0=np.array([0, 0]), bounds=[(-10, 10), ] * 2))


# ----- WITH HYPEROPT ---- #

def fun(args):

    x, y = args

    return x**2 + y*x

best = fmin(
    fn=fun,
    space=[hp.uniform('x', -10, 10), hp.uniform('y', -10, 10)],
    algo=tpe.suggest,
    max_evals=1000
)

print(best)

# With integers only
best = fmin(
    fn=fun,
    space=[hp.quniform('x', -10, 10, 1), hp.quniform('y', -10, 10, 1)],
    algo=tpe.suggest,
    max_evals=1000
)

print(best)

