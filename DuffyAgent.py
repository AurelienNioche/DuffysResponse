import numpy as np
from AbstractClasses import Agent


class DuffyAgent(Agent):

    def __init__(self, prod, cons, third, agent_type, idx, storing_costs, beta=0.9, u=1):

        super().__init__(prod=prod, cons=cons, third=third, agent_type=agent_type, idx=idx)

        self.storing_costs = storing_costs

        self.values = np.zeros(2)

        # Let gamma[0] be gamma_{i+1} and gamma[1] be gamma_{i+2}
        self.gamma = np.array([
            - self.storing_costs[self.P] + beta * u,
            - self.storing_costs[self.T] + beta * u
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



