import numpy as np


def get_roles(n_goods):
    roles = np.zeros((n_goods, 2), dtype=int)

    for i in range(n_goods):
        roles[i] = (i + 1) % n_goods, i

    return roles