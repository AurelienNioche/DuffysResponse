import numpy as np
from os import path
from pylab import plt
from KWModels import ModelA
from Economy import launch
from analysis import represent_results
from MarimonAgent import MarimonAgent


def plot_proportions(proportions, fig_name):

    # Container for proportions of agents having this or that in hand according to their type
    #  - rows: type of agent
    # - columns: type of good

    fig = plt.figure(figsize=(25, 12))
    fig.patch.set_facecolor('white')

    n_lines = 3
    n_columns = 1

    x = np.arange(len(proportions))

    for agent_type in range(3):

        # First subplot
        ax = plt.subplot(n_lines, n_columns, agent_type + 1)
        ax.set_title("\nProportion of agents of type {} having good 1, 2, 3 in hand\n".format(agent_type + 1))

        y0 = []
        y1 = []
        y2 = []
        for proportions_at_t in proportions:
            y0.append(proportions_at_t[agent_type, 0])
            y1.append(proportions_at_t[agent_type, 1])
            y2.append(proportions_at_t[agent_type, 2])

        ax.set_ylim([-0.02, 1.02])

        ax.plot(x, y0, label="Good 1", linewidth=2)
        ax.plot(x, y1, label="Good 2", linewidth=2)
        ax.plot(x, y2, label="Good 3", linewidth=2)
        ax.legend()

    plt.tight_layout()

    plt.savefig(filename=fig_name)


if __name__ == "__main__":

    parameters = {
        "t_max": 500,
        "agent_parameters": {"u": 500, "b11": 0.025, "b12": 0.025, "b21": 0.25, "b22": 0.25, "initial_strength": 0},
        "role_repartition": np.array([50, 50, 50]),
        "storing_costs": np.array([0.1, 1., 20.]),
        "kw_model": ModelA,
        "agent_model": MarimonAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)
    plot_proportions(proportions=backup["proportions"])
