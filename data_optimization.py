import numpy as np
from scipy import optimize as op
from data_optimization2 import PerformanceComputer


def main():

    storing_costs = np.asarray([1, 4, 9])
    u = 100

    pc = PerformanceComputer(subject_idx=0, raw_storing_costs=storing_costs,
                             raw_u=u)

    first_alpha = 0.5
    alpha_limits = (0., 1.)

    first_temp = 0.5
    temp_limits = (0.01, 1.)  # Lower limit can not be zero, motherfucker!

    first_gamma = 0.5
    gamma_limits = (0., 1.)

    q_values = np.random.random((12, 2))
    q_values_limits = [(0, 1), ] * q_values.size

    x0 = np.array(
        [first_alpha, first_temp, first_gamma]
        + list(np.ravel(q_values))
    )

    bounds = np.array(
        [alpha_limits, temp_limits, gamma_limits] + q_values_limits
    )

    print(op.minimize(fun=pc.run, x0=x0, method="L-BFGS-B", bounds=bounds))


if __name__ == "__main__":

    main()
