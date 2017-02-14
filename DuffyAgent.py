import numpy as np
from KWModels import ModelA
from Economy import launch
from analysis import represent_results
from AbstractAgent import Agent


class DuffyAgent(Agent):

    name = "Duffy"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Let values[0] be the v_{i+1} and values[1] be v_{i+2}
        self.values = np.zeros(2)

        self.u, self.storing_costs = self.define_u_and_storing_costs(self.u, self.storing_costs)

        # Let gamma[0] be gamma_{i+1} and gamma[1] be gamma_{i+2}
        self.gamma = np.array([
            - self.storing_costs[self.P] + self.beta * self.u,
            - self.storing_costs[self.T] + self.beta * self.u
        ])

        self.in_hand_at_the_beginning_of_the_round = self.P

    @staticmethod
    def define_u_and_storing_costs(u, storing_costs):

        new_storing_costs = np.zeros(3)
        new_storing_costs[:] = storing_costs[:] / u

        new_u = 1

        return new_u, new_storing_costs

    def are_you_satisfied(self, proposed_object, type_of_other_agent, proportions):

        self.in_hand_at_the_beginning_of_the_round = self.in_hand

        if proposed_object == self.C:
            accept = 1

        elif self.in_hand == self.P and proposed_object == self.T:

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

        if self.in_hand_at_the_beginning_of_the_round == self.P:

            self.values[0] += self.consumption * self.gamma[0] - (1-self.consumption) * self.gamma[1]

        elif self.in_hand_at_the_beginning_of_the_round == self.T:

            self.values[1] += self.consumption * self.gamma[1] - (1-self.consumption) * self.gamma[0]

            # ----------  FOR OPTIMIZATION PART ---------- #

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        self.in_hand_at_the_beginning_of_the_round = self.in_hand

        if partner_good == self.C:
            print("C")
            print("s", subject_response)
            p_values = [0, 1]  # Accept for sure

        elif self.in_hand == self.P and partner_good == self.T:

            x = self.values[0] - self.values[1]
            p_refusing = np.exp(x) / (1 + np.exp(x))
            p_values = [p_refusing, 1 - p_refusing]

        else:
            print("other")
            print("s", subject_response)
            p_values = [1, 0]

        return p_values[subject_response]

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        if subject_choice and partner_choice:
            self.in_hand = partner_good

        self.consume()

        self.learn()


def main():

    parameters = {
        "t_max": 500,
        "beta": 0.9,
        "u": 100,
        "role_repartition": [500, 500, 500],
        "storing_costs": [1, 4, 9],
        "kw_model": ModelA,
        "agent_model": DuffyAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)
if __name__ == "__main__":

    main()
