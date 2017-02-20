import numpy as np
from KWModels import ModelA
from Economy import Economy
from analysis import represent_results
from AbstractAgent import Agent
from module.useful_functions import softmax
from save import save


class RLForwardAgent2(Agent):
    name = "RLForward"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- RL2 PARAMETERS ---- #

        self.alpha = self.agent_parameters["alpha"]
        self.gamma = self.agent_parameters["gamma"]
        self.temp = self.agent_parameters["temp"]

        # Memory of the matching
        self.previous_matching_pair = (-1, -1)
        self.matching_pair = None

        # ------- STRATEGIES ------- #
        self.strategies = self.generate_strategies(self.agent_parameters["q_values"].copy())

        self.u, self.storing_costs = self.define_u_and_storing_costs(self.u, self.storing_costs)

        self.previous_followed_strategy = 0
        self.followed_strategy = None

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        self.matching_pair = (self.in_hand, partner_good)
        self.select_strategy()

        return self.followed_strategy  # 1 for agreeing, 0 otherwise

    # ------------------------ RL2 PART ------------------------------------------------------ #

    def generate_strategies(self, initial_values):

        idx = 0

        strategies = {}

        # For each object the agent could have in hand
        for i in [self.P, self.T]:
            # for every object that could be proposed # --- !!!! MAIN DIFFERENCE WITH FIRST VERSION ---- !!!!! 
            for j in range(3):

                    # Key is composed by good in hand, partner type, good in partner's hand
                strategies[(i, j)] = initial_values[idx, :].copy()

                idx += 1

        # For the first round
        strategies[self.previous_matching_pair] = np.zeros(2)

        return strategies

    @staticmethod
    def define_u_and_storing_costs(u, storing_costs):

        # To be sure that q values will be remained between 0 and 1.
        amplitude = u - min(storing_costs) + max(storing_costs)

        new_storing_costs = np.zeros(3)
        new_storing_costs[:] = storing_costs[:] / amplitude

        new_u = u / amplitude

        return new_u, new_storing_costs

    def compute_utility(self):

        # Be sure that utility can not be over 1 or under 1.

        # Anchorage is at the maximum of the storing costs so the worst option leads to a utility of 0.
        utility = \
            max(self.storing_costs) + self.u * self.consumption - self.storing_costs[self.in_hand]

        # Be sure that utility lies between 0 and 1
        assert 0 <= utility <= 1

        return utility

    def learn(self):

        utility = self.compute_utility()

        # Matching triplet is the matching triplet of t - 1

        for learning_rate, pair, strategy in zip(
                [self.alpha, self.gamma],
                [self.previous_matching_pair, self.matching_pair],
                [self.previous_followed_strategy, self.followed_strategy]):

            delta = utility - self.strategies[pair][strategy]

            self.strategies[pair][strategy] += \
                learning_rate * delta

    def select_strategy(self):

        relevant_strategies_values = self.strategies[self.matching_pair]
        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(relevant_strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(relevant_strategies_values)), p=p_values)

    def consume(self):

        super().consume()
        self.learn()

        self.previous_matching_pair = self.matching_pair
        self.previous_followed_strategy = self.followed_strategy

    # ----------  FOR OPTIMIZATION PART ---------- #

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        relevant_strategies_values = self.strategies[(self.in_hand, partner_good)]
        p_values = softmax(relevant_strategies_values, self.temp)

        # Assume there is only 2 p-values, return the one corresponding to the choice of the subject
        return p_values[subject_response]

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        # Memory for learning
        self.matching_pair = self.in_hand, partner_good

        self.followed_strategy = subject_choice

        if subject_choice and partner_choice:
            self.in_hand = partner_good

        self.consume()

        self.learn()


def main():

    storing_costs = np.array([0.01, 0.04, 0.09]) * 5
    u = 1

    parameters = {
        "t_max": 500,
        "agent_parameters": {"alpha": 0.5, "temp": 0.1, "gamma": 0.5,
                             "q_values": np.ones((6, 2))},
        "role_repartition": np.array([50, 50, 50]),
        "storing_costs": storing_costs,
        "u": u,
        "kw_model": ModelA,
        "agent_model": RLForwardAgent2,
    }

    e = Economy(**parameters)

    backup = e.run()

    # backup["last_strategies"] = [agent.strategies for agent in e.agents]

    # save(backup)

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":
    main()
