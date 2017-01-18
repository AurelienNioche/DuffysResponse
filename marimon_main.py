import numpy as np
from os import path
from KWModels import ModelA
from Economy import launch
from analysis import represent_results
from MarimonAgent import MarimonAgent


if __name__ == "__main__":

    parameters = {
        "t_max": 10000,
        "agent_parameters": {"u": 100, "b11": 0.025, "b12": 0.025, "b21": 0.25, "b22": 0.25, "initial_strength": 0},
        "role_repartition": np.array([50, 50, 50]),
        "storing_costs": np.array([0.1, 1., 20]),
        "kw_model": ModelA,
        "agent_model": MarimonAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters, fig_name=path.expanduser("~/Desktop/KW_Marimon_Agents.pdf"))