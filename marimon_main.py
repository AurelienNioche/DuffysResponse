import numpy as np
from os import path
from KWModels import ModelA
from Economy import launch
from analysis import represent_results
from MarimonAgent import MarimonAgent


if __name__ == "__main__":

    parameters = {
        "t_max": 1000,
        "agent_parameters": {"u": 100, "b11": 0.025, "b12": 0.025, "b21": 0.25, "b22": 0.25, "initial_strength": 0},
        "role_repartition": np.array([50, 50, 50]),
        "storing_costs": np.array([0.1, 1., 20.]),
        "kw_model": ModelA,
        "agent_model": MarimonAgent,
    }

    backup = \
        launch(
            **parameters
        )
    fig_name = path.expanduser("~/Desktop/KW_Marimon_Agents.pdf")
    init_fig_name = fig_name.split(".")[0]
    i = 2
    while path.exists(fig_name):
        fig_name = init_fig_name + "{}.pdf".format(i)
        i += 1
    represent_results(backup=backup, parameters=parameters, fig_name=fig_name)
