import numpy as np
from KWModels import ModelA
from Economy import Economy
from analysis import represent_results
from AbstractAgent import Agent
from module.useful_functions import softmax


'''
Same as 'RL' but with different learning rates for positive and negative outcomes.
RL with reinforcement of strategies understood as Game Theory does
 (a strategy is a set of action plans for every possible situation)
'''


class RLForwardAgent(Agent):
    name = "RLForward"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- RL2 PARAMETERS ---- #

        self.alpha = self.agent_parameters["alpha"]
        self.gamma = self.agent_parameters["gamma"]
        self.temp = self.agent_parameters["temp"]

        # Memory of the matching
        self.matching_triplet = (-1, -1, -1)

        # ------- STRATEGIES ------- #
        self.strategies = self.generate_strategies(self.agent_parameters["q_values"].copy())

        self.u, self.storing_costs = self.define_u_and_storing_costs(self.u, self.storing_costs)

        self.followed_strategy = None

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        self.learn(partner_good, partner_type)

        self.select_strategy(partner_good=partner_good, partner_type=partner_type)

        return self.followed_strategy  # 1 for agreeing, 0 otherwise

    # ------------------------ RL2 PART ------------------------------------------------------ #

    def generate_strategies(self, intitial_values):

        idx = 0

        strategies = {}

        # For each object the agent could have in hand
        for i in [self.P, self.T]:
            # for every type of agent he could be matched with
            for j in range(3):
                # For each object this 'partner' could have in hand (production, third good)
                for k in [self.kw_model.roles[j][0], self.kw_model.roles[j][2]]:

                    # Key is composed by good in hand, partner type, good in partner's hand
                    strategies[(i, j, k)] = intitial_values[idx][:]

                    idx += 1

        # For the first round
        strategies[self.matching_triplet] = np.zeros(2)

        return strategies

    @staticmethod
    def define_u_and_storing_costs(u, storing_costs):

        # To be sure that q values will be remained between 0 and 1.
        amplitude = u - min(storing_costs) + max(storing_costs)

        new_storing_costs = np.zeros(3)
        new_storing_costs[:] = storing_costs[:] / amplitude

        new_u = u/amplitude

        return new_u, new_storing_costs

    def compute_utility(self):

        # Be sure that utility can not be over 1 or under 1.

        # Anchorage is at the maximum of the storing costs so the worst option leads to a utility of 0.
        utility = \
            max(self.storing_costs) + self.u * self.consumption - self.storing_costs[self.in_hand]

        # Be sure that utility lies between 0 and 1
        assert 0 <= utility <= 1

        return utility

    def learn(self, partner_good, partner_type):

        # Matching triplet is the matching triplet of t - 1
        delta = self.compute_utility() - self.strategies[self.matching_triplet][self.followed_strategy]

        self.strategies[self.matching_triplet][self.followed_strategy] += \
            self.alpha * delta

        if not self.consumption:

            forward_value = max(self.strategies[(self.in_hand, partner_type, partner_good)]) \
                - self.strategies[self.matching_triplet][self.followed_strategy]

            self.strategies[self.matching_triplet][self.followed_strategy] += \
                self.gamma * forward_value

    def select_strategy(self, partner_good, partner_type):

        relevant_strategies_values = self.strategies[(self.in_hand,  partner_type, partner_good)]
        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(relevant_strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(relevant_strategies_values)), p=p_values)

        # Memory for learning
        self.matching_triplet = self.in_hand, partner_type, partner_good

    # ----------  FOR OPTIMIZATION PART ---------- #
    
    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        relevant_strategies_values = self.strategies[(self.in_hand, partner_type, partner_good)]
        p_values = softmax(relevant_strategies_values, self.temp)
        
        # Assume there is only 2 p-values, return the one corresponding to the choice of the subject
        return p_values[subject_response]
    
    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        # Memory for learning
        self.matching_triplet = self.in_hand, partner_type, partner_good

        self.followed_strategy = subject_choice
        
        if subject_choice and partner_choice:
            
            self.in_hand = partner_good

        self.consume()


def main():

    storing_costs = np.array([0.01, 0.04, 0.09])
    u = 1

    parameters = {
        "t_max": 500,
        "agent_parameters": {"alpha": 0.005, "temp": 0.1, "gamma": 0.9,
                             "q_values": np.random.random((12, 2))},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": storing_costs,
        "u": u,
        "kw_model": ModelA,
        "agent_model": RLForwardAgent,
    }

    e = Economy(**parameters)

    backup = e.play()

    for i, agent in enumerate(e.agents):

        print("Agent {}".format(i))
        for key, value in agent.strategies.items():
            print("{} {:.2f}, {:.2f}".format(key, value[0], value[1]))

        print()

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":

    main()
