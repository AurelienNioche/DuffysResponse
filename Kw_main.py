from os import path
import numpy as np
from Economy import launch
from KwAgent import KwAgent
from KWModels import ModelA
from analysis import represent_results


if __name__ == "__main__":

    parameters = {
        "t_max": 100,
        "agent_parameters": {"alpha": 0.3, "temp": 0.01},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([0.01, 0.04, 0.09]),
        "kw_model": ModelA,
        "agent_model": KwAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters, fig_name=path.expanduser("~/Desktop/KW_Kw_Agents.pdf"))
