import numpy as np
from AbstractAgent import Agent
from KWModels import ModelA
from Economy import launch
from analysis import represent_results


class KwAgent(Agent):
    name = "Kw"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        assert self.kw_model == ModelA, "Must be 'Model A'."
        assert 0 < self.storing_costs[0] < self.storing_costs[1] < self.storing_costs[2], "Must be 'Economy A'."

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, partner_good, type_of_other_agent, proportions):

        if partner_good == self.C:
            return 1

        elif type_of_other_agent == self.C or partner_good == self.P:  # Type is defined by what an agent consumes
            return 0

        elif partner_good == self.T:

            if self.C == 1:
                return 1

            elif self.C == 2:
                return 0

            else:
                # P 300 of Duffy's Learning to Speculate
                cond = (self.storing_costs[2] - self.storing_costs[1]) > \
                    (proportions[2, 0] - (1-proportions[1, 2])) / 3 * self.beta * self.u
                if cond:
                    return 0
                else:
                    return 1
                
    # -------------- FITTING ------------------------- #
    
    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        if partner_good == self.C:
            return subject_response == 1

        elif partner_type == self.C or partner_good == self.P:  # Type is defined by what an agent consumes
            return subject_response == 0

        elif partner_good == self.T:

            if self.C == 1:
                return subject_response == 1

            elif self.C == 2:
                return subject_response == 0

            else:
                # P 300 of Duffy's Learning to Speculate
                cond = (self.storing_costs[2] - self.storing_costs[1]) > \
                    (proportions[2] - (1-proportions[1])) / 3 * self.beta * self.u
                if cond:
                    return subject_response == 0
                else:
                    return subject_response == 1


def main():

    parameters = {
        "t_max": 200,
        "u": 100,
        "beta": 0.9,
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
