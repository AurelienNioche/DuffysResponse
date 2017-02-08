import numpy as np
from KWModels import ModelA
from Economy import launch
from analysis import represent_results
from AbstractAgent import Agent
from module.useful_functions import softmax


class BayesianAgent(Agent):
    name = "BayesianAgent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- RL2 PARAMETERS ---- #

        self.alpha = self.agent_parameters["alpha"]
        self.gamma = self.agent_parameters["gamma"]
        self.temp = self.agent_parameters["temp"]

        # Memory of the matching
        self.matching_triplet = (-1, -1, -1)

        # ------- STRATEGIES ------- #
        self.strategies = self.generate_strategies()

        self.followed_strategy = None

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, proposed_object, type_of_other_agent, proportions):

        self.learn(proposed_object, type_of_other_agent)

        self.select_strategy(proposed_object=proposed_object, type_of_other_agent=type_of_other_agent)

        return self.followed_strategy  # 1 for agreeing, 0 otherwise

    # ------------------------ RL2 PART ------------------------------------------------------ #

    def generate_strategies(self):

        strategies = {}

        # For each object the agent could have in hand
        for i in [self.P, self.T]:
            # for every type of agent he could be matched with
            for j in range(3):
                # For each object this 'partner' could have in hand (production, third good)
                for k in [self.kw_model.roles[j][0], self.kw_model.roles[j][2]]:

                    # Key is composed by good in hand, partner type, good in partner's hand
                    strategies[(i, j, k)] = np.zeros(2)

        # For the first round
        strategies[self.matching_triplet] = np.zeros(2)

        return strategies

    def compute_utility(self):

        # You find that strange? We don't care. (F*** you if you do).
        utility = \
            self.consumption - self.storing_costs[self.in_hand]

        # Be sure that utility lies between 0 and 1
        # assert 0 <= utility <= 1

        return utility

    def learn(self, proposed_object, type_of_other_agent):

        # Matching triplet is the matching triplet of t - 1

        delta = self.compute_utility() - self.strategies[self.matching_triplet][self.followed_strategy]

        if not self.consumption:
            forward_value = max(self.strategies[(self.in_hand, type_of_other_agent, proposed_object)])
        else:
            forward_value = 0

        self.strategies[self.matching_triplet][self.followed_strategy] += \
            self.alpha * (
                delta
                + self.gamma * forward_value
            )

    def select_strategy(self, proposed_object, type_of_other_agent):

        relevant_strategies_values = self.strategies[(self.in_hand,  type_of_other_agent, proposed_object)]
        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(relevant_strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(relevant_strategies_values)), p=p_values)

        # Memory for learning
        self.matching_triplet = self.in_hand, type_of_other_agent, proposed_object


if __name__ == "__main__":

    parameters = {
        "t_max": 200,
        "agent_parameters": {"alpha": 0.25, "temp": 0.01, "gamma": 0.5},
        "role_repartition": np.array([5000, 5000, 5000]),
        "storing_costs": np.array([0.01, 0.04, 0.09]),
        "kw_model": ModelA,
        "agent_model": BayesianAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)
