import numpy as np


class ModelA(object):
    roles = np.array([
        [1, 0, 2],
        [2, 1, 0],
        [0, 2, 1]
    ], dtype=int)

    name = "ModelA"


class ModelB(object):
    roles = np.array([
        [2, 0, 1],
        [0, 1, 2],
        [1, 2, 0]
    ], dtype=int)

    name = "ModelB"
