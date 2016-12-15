from os import path
import numpy as np
from Economy import launch
from RL import RLAgent
from KWModels import ModelA
from analysis import represent_results


if __name__ == "__main__":

    parameters = {
        "t_max": 200,
        "agent_parameters": {"alpha": 0.1, "temp": 0.01},
        "role_repartition": np.array([100, 100, 100]),
        "storing_costs": np.array([0.1, 0.25, 0.265]),
        "kw_model": ModelA,
        "agent_model": RLAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters, fig_name=path.expanduser("~/Desktop/KW.pdf"))
