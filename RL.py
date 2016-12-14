import numpy as np
from module.useful_functions import softmax


class Agent(object):

    def __init__(self, alpha, temp, prod, cons, third, storing_costs, agent_type, model, idx):

        self.P = prod
        self.C = cons
        self.T = third

        self.model = model

        self.storing_costs = storing_costs

        self.in_hand = self.P

        self.agent_type = agent_type
        self.idx = idx

        # ------- STRATEGIES ------- #

        # Dimension 0: strategies,
        # Dimension 1: object in hand (i, k),
        # Dimension 2: proposed object (i, j, k),
        # We suppose that :
        # - An agent always accepts his consumption good
        # - An agent always refuse the exchange if the proposed object is the same that the one he has in hand
        # - Strategies therefore contrast by attitude of the agent towards the third good if he has his production
        #    good in hand, and the production good if he has his third good in hand
        self.strategies = np.array([
            [[0, 1, 0],
             [np.nan, np.nan, np.nan],
             [0, 1, 0]],
            [[0, 1, 1],
             [np.nan, np.nan, np.nan],
             [0, 1, 0]],
            [[0, 1, 0],
             [np.nan, np.nan, np.nan],
             [1, 1, 0]],
            [[0, 1, 1],
             [np.nan, np.nan, np.nan],
             [1, 1, 0]],
        ])
        self.strategies_values = np.random.random(len(self.strategies))
        self.n_strategies = len(self.strategies)
        self.followed_strategy = 0

        self.utility = 0
        self.consumption = 0

        self.abs_to_relative = np.array([
            [
                np.where(self.model.roles[0] == 0)[0][0],
                np.where(self.model.roles[0] == 1)[0][0],
                np.where(self.model.roles[0] == 2)[0][0]
            ],
            [
                np.where(self.model.roles[1] == 0)[0][0],
                np.where(self.model.roles[1] == 1)[0][0],
                np.where(self.model.roles[1] == 2)[0][0]
            ],
            [
                np.where(self.model.roles[2] == 0)[0][0],
                np.where(self.model.roles[2] == 1)[0][0],
                np.where(self.model.roles[2] == 2)[0][0]
            ]
        ], dtype=int)

        # Take object with absolute reference to give object relating to agent
        self.int_to_ijk = self.abs_to_relative[agent_type]

        # ----- RL PARAMETERS ---- #

        self.alpha = alpha
        self.temp = temp

    def are_you_satisfied(self, proposed_object):
        return self.strategies[self.followed_strategy, self.int_to_ijk[self.in_hand], self.int_to_ijk[proposed_object]]

    def consume(self):
        self.consumption = self.in_hand == self.C

        if self.consumption:
            self.in_hand = self.P

        self.utility = \
            0.5 + self.consumption / 2 - self.storing_costs[self.in_hand]

        assert 0 <= self.utility <= 1

    # ------------------------ RL PART ------------------------------------------------------ #

    def learn(self):
        self.strategies_values[self.followed_strategy] += \
            self.alpha * (self.utility - self.strategies_values[self.followed_strategy])

    def select_strategy(self):
        p_values = softmax(self.strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(self.strategies_values)), p=p_values)