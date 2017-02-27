import numpy as np
from Economy import launch
from graph import represent_results
from stupid_agent import StupidAgent


class DuffyAgent(StupidAgent):

    name = "Duffy"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        assert len(self.storing_costs) == 3, "Duffy Agent can not handle only 3 goods."

        self.T = [i for i in range(3) if i != self.P and i != self.C][0]

        # Let values[0] be the v_{i+1} and values[1] be v_{i+2}
        self.values = np.zeros(2)

        self.u, self.storing_costs = self.define_u_and_storing_costs(self.u, self.storing_costs)

        # Let gamma[0] be gamma_{i+1} and gamma[1] be gamma_{i+2}
        self.gamma = np.array([
            - self.storing_costs[self.P] + self.beta * self.u,
            - self.storing_costs[self.T] + self.beta * self.u
        ])

        self.H_at_the_beginning_of_the_round = self.P

    @staticmethod
    def define_u_and_storing_costs(u, storing_costs):

        new_storing_costs = np.zeros(3)
        new_storing_costs[:] = storing_costs[:] / u

        new_u = 1

        return new_u, new_storing_costs

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        self.H_at_the_beginning_of_the_round = self.H

        if partner_good == self.C:
            accept = 1

        elif self.H == self.P and partner_good == self.T:

            x = self.values[0] - self.values[1]
            p_refusing = np.exp(x) / (1 + np.exp(x))
            accept = np.random.choice([0, 1], p=[p_refusing, 1 - p_refusing])

        else:
            accept = 0

        return accept

    def consume(self):

        super().consume()

        self.learn()

    def learn(self):

        if self.H_at_the_beginning_of_the_round == self.P:

            self.values[0] += self.consumption * self.gamma[0] - (1-self.consumption) * self.gamma[1]

        elif self.H_at_the_beginning_of_the_round == self.T:

            self.values[1] += self.consumption * self.gamma[1] - (1-self.consumption) * self.gamma[0]

            # ----------  FOR OPTIMIZATION PART ---------- #

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        self.H_at_the_beginning_of_the_round = self.H

        if partner_good == self.C:

            p_values = [0, 1]  # Accept for sure

        elif self.H == self.P and partner_good == self.T:

            x = self.values[0] - self.values[1]
            p_refusing = np.exp(x) / (1 + np.exp(x))
            p_values = [p_refusing, 1 - p_refusing]

        else:

            p_values = [1, 0]

        return p_values[subject_response]

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        if subject_choice and partner_choice:
            self.H = partner_good

        self.consume()

        self.learn()


def main():

    parameters = {
        "t_max": 500,
        "beta": 0.9,
        "u": 100,
        "repartition_of_roles": [500, 500, 500],
        "storing_costs": [1, 4, 9],
        "agent_model": DuffyAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)
if __name__ == "__main__":

    main()
