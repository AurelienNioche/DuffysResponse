import numpy as np
from AbstractClasses import Agent


class DuffyAgent(Agent):

    name = "Duffy"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Let values[0] be the v_{i+1} and values[1] be v_{i+2}
        self.values = np.zeros(2)

        # Let gamma[0] be gamma_{i+1} and gamma[1] be gamma_{i+2}
        self.gamma = np.array([
            - self.storing_costs[self.P] + self.agent_parameters["beta"] * self.agent_parameters["u"],
            - self.storing_costs[self.T] + self.agent_parameters["beta"] * self.agent_parameters["u"]
        ])

        self.in_hand_at_the_beginning_of_the_round = self.P

    def are_you_satisfied(self, proposed_object, proportions):

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
