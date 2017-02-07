import numpy as np
from AbstractAgent import Agent
from KWModels import ModelA
from Economy import launch
from analysis import represent_results


class KwAgent(Agent):
    name = "Kw"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = self.agent_parameters["beta"]
        self.u = self.agent_parameters["u"]

        assert self.kw_model == ModelA, "Must be 'Model A'."
        assert 0 < self.storing_costs[0] < self.storing_costs[1] < self.storing_costs[2], "Must be 'Economy A'."

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, proposed_object, type_of_other_agent, proportions):

        if type_of_other_agent == self.type:
            return 0
        elif proposed_object == self.P:
            return 0
        elif proposed_object == self.C:
            return 1

        elif proposed_object == self.T:
            if self.type == 1:
                return 1
            elif self.type == 2:
                return 0
            else:
                # P 300 of Duffy's Learning to Speculate
                cond = (self.storing_costs[2] - self.storing_costs[1]) > \
                    (proportions[2, 0] - (1-proportions[1, 2])) / 3 * self.beta * self.u
                if cond:
                    print("ijdeijidneirn f")
                    return 0
                else:
                    return 1


def main():

    parameters = {
        "t_max": 200,
        "agent_parameters": {"beta": 0.9, "u": 100},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([0.01, 0.04, 0.09]),
        "kw_model": ModelA,
        "agent_model": KwAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)



if __name__ == "__main__":

    main()
