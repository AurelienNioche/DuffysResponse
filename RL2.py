import numpy as np
from AbstractClasses import Agent
from module.useful_functions import softmax


'''
Same as 'RL' but with different learning rates for positive and negative outcomes.
RL with reinforcement of strategies understood as Game Theory does
 (a strategy is a set of action plans for every possible situation)
'''


class RL2Agent(Agent):
    name = "RL2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ------- STRATEGIES ------- #

        # Dimension 0: strategies,
        # Dimension 1: object in hand with relative idx (0: production good, 1: consumption good, 2: third good),
        # Dimension 2: proposed object with relative idx (0: production good, 1: consumption good, 2: third good),
        # We suppose that :
        # - An agent can never has his consumption good in hand
        #                   -> he directly consumes it (that is why we have 'nan' for Not A Number)
        # - An agent always accepts his consumption good
        # - An agent always refuse the exchange if the proposed object is the same that the one he has in hand
        # - Strategies therefore contrast by attitude of the agent towards the third good if he has his production
        #    good in hand, and the production good if he has his third good in hand
        self.strategies = np.array([
            # Strategy '0'
            [[0, 1, 0],
             [np.nan, np.nan, np.nan],
             [0, 1, 0]],
            # Strategy '1'
            [[0, 1, 1],
             [np.nan, np.nan, np.nan],
             [0, 1, 0]],
            # Strategy '2'
            [[0, 1, 0],
             [np.nan, np.nan, np.nan],
             [1, 1, 0]],
            # Strategy '3'
            [[0, 1, 1],
             [np.nan, np.nan, np.nan],
             [1, 1, 0]],
        ])

        self.strategies_values = np.random.random(len(self.strategies))

        # It will be an integer between 0 and 3
        self.followed_strategy = None

        self.utility = None

        absolute_to_relative_3_types = np.array([
            [
                np.where(self.kw_model.roles[0] == 0)[0][0],
                np.where(self.kw_model.roles[0] == 1)[0][0],
                np.where(self.kw_model.roles[0] == 2)[0][0]
            ],
            [
                np.where(self.kw_model.roles[1] == 0)[0][0],
                np.where(self.kw_model.roles[1] == 1)[0][0],
                np.where(self.kw_model.roles[1] == 2)[0][0]
            ],
            [
                np.where(self.kw_model.roles[2] == 0)[0][0],
                np.where(self.kw_model.roles[2] == 1)[0][0],
                np.where(self.kw_model.roles[2] == 2)[0][0]
            ]
        ], dtype=int)

        # Take object with absolute reference to give object relating to agent
        #    (with 0: production good, 1: consumption good, 2: third object)

        self.absolute_to_relative = absolute_to_relative_3_types[self.type]

        # ----- RL2 PARAMETERS ---- #

        self.alpha_plus = self.agent_parameters["alpha_plus"]
        self.alpha_minus = self.agent_parameters["alpha_minus"]
        self.temp = self.agent_parameters["temp"]

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, proposed_object, type_of_other_agent, propositions):
        self.select_strategy()
        agreeing = self.strategies[self.followed_strategy,
                                   self.absolute_to_relative[self.in_hand],
                                   self.absolute_to_relative[proposed_object]]

        return agreeing

    def consume(self):
        # Call 'consume' method from parent (that is 'Agent')
        super().consume()

        # In more, compute utility and learn something from results
        self.compute_utility()
        self.learn()

    # ------------------------ RL2 PART ------------------------------------------------------ #

    def compute_utility(self):
        self.utility = \
            0.5 + self.consumption / 2 - self.storing_costs[self.in_hand]

        # Be sure that utility lies between 0 and 1
        assert 0 <= self.utility <= 1

    def learn(self):
        # Double Learning Rate RL rule
        if self.utility - self.strategies_values[self.followed_strategy] >= 0:
            self.strategies_values[self.followed_strategy] += \
                self.alpha_plus * (self.utility - self.strategies_values[self.followed_strategy])

        if self.utility - self.strategies_values[self.followed_strategy] < 0:
            self.strategies_values[self.followed_strategy] += \
                self.alpha_minus * (self.utility - self.strategies_values[self.followed_strategy])

    def select_strategy(self):
        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(self.strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(self.strategies_values)), p=p_values)
