from os import path
import numpy as np
from Economy import launch
from RL2 import RL2Agent
from KWModels import ModelA
from analysis import represent_results


if __name__ == "__main__":

    parameters = {
        "t_max": 500,
        "agent_parameters": {"alpha_plus": 0.4, "alpha_minus": 0.05, "temp": 0.01},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([0.1, 0.25, 0.4]),
        "kw_model": ModelA,
        "agent_model": RL2Agent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)
