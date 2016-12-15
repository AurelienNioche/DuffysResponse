import numpy as np
from os import path
from KWModels import ModelA
from Economy import launch
from analysis import represent_results
from DuffyAgent import DuffyAgent


if __name__ == "__main__":

    parameters = {
        "t_max": 500,
        "agent_parameters": {"beta": 0.9, "u": 1},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([0.1, 0.25, 0.5]),
        "kw_model": ModelA,
        "agent_model": DuffyAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters, fig_name=path.expanduser("~/Desktop/KW_Duffy_Agents.pdf"))
