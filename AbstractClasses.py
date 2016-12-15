import numpy as np


class Agent(object):

    """
    Abstract class for agents
    """

    def __init__(self, prod, cons, third, agent_type, idx):

        # Production object (integer in [0, 1, 2])
        self.P = prod

        # Consumption object (integer in [0, 1, 2])
        self.C = cons

        # Other object (integer in [0, 1, 2])
        self.T = third

        # Type of agent (integer in [0, 1, 2])
        self.type = agent_type

        # Index of agent (more or less his name ; integer in [0, ..., n] with n : total number of agent)
        self.idx = idx

        # Keep a trace for time t if the agent consumed or not.
        self.consumption = 0

        # Object an agent
        self.in_hand = self.P

    def are_you_satisfied(self, proposed_object, proportions):

        return np.random.choice([True, False])

    def consume(self):

        self.consumption = self.in_hand == self.C

        if self.consumption:
            self.in_hand = self.P
