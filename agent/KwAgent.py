import numpy as np

from agent.stupid_agent import StupidAgent
from environment.Economy import launch
from graph.graph import represent_results


class KwAgent(StupidAgent):
    name = "Kw"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        assert len(self.storing_costs) == 3, "KW Agent can not handle only 3 goods."

        self.T = [i for i in range(3) if i != self.P and i != self.C][0]

        assert 0 < self.storing_costs[0] < self.storing_costs[1] < self.storing_costs[2], "Must be 'Economy A'."

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        if partner_good == self.C:
            return 1

        elif partner_type == self.C or partner_good == self.P:  # Type is defined by what an agent consumes
            return 0

        elif partner_good == self.T:

            if self.C == 1:
                return 1

            elif self.C == 2:
                return 0

            else:
                # P 300 of Duffy's Learning to Speculate
                cond = (self.storing_costs[2] - self.storing_costs[1]) < \
                    (proportions[2, 0] - (1-proportions[1, 2])) / 3 * self.beta * self.u
                return int(cond)
                
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
                cond = (self.storing_costs[2] - self.storing_costs[1]) < \
                    (proportions[2] - (1-proportions[1])) / 3 * self.beta * self.u
                return subject_response == int(cond)


def main():

    parameters = {
        "t_max": 100,
        "u": 1,
        "beta": 0.9,
        "repartition_of_roles": np.array([500, 500, 500]),
        "storing_costs": np.array([0.01, 0.04, 0.09]),
        "agent_model": KwAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":

    main()
