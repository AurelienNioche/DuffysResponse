import numpy as np
from Economy import Economy
from graph import represent_results
from stupid_agent import StupidAgent
from module.useful_functions import softmax
from save import save


'''
Same as 'RL' but with different learning rates for positive and negative outcomes.
RL with reinforcement of strategies understood as Game Theory does
 (a strategy is a set of action plans for every possible situation)
'''


class ForwardRLAgent(StupidAgent):
    name = "ForwardRL"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.storing_costs) == 3, "RLForward Agent can not handle only 3 goods."

        self.T = [i for i in range(3) if i != self.P and i != self.C][0]

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

    def generate_strategies(self, initial_values):

        idx = 0

        strategies = {}

        # For each object the agent could have in hand
        for i in [self.P, self.T]:
            # for every type of agent he could be matched with
            for j in range(3):
                # For each object this 'partner' could have in hand (production, third good)
                for k in [i for i in range(3) if i != j]:
                    if initial_values is not None:
                        # Key is composed by good in hand, partner type, good in partner's hand
                        strategies[(i, j, k)] = initial_values[idx, :].copy()
                    else:
                        strategies[(i, j, k)] = np.zeros(2)

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
            max(self.storing_costs) + self.u * self.consumption - self.storing_costs[self.H]

        # Be sure that utility lies between 0 and 1
        assert 0 <= utility <= 1

        return utility

    def learn(self, partner_good, partner_type):

        # Matching triplet is the matching triplet of t - 1
        delta = self.compute_utility() - self.strategies[self.matching_triplet][self.followed_strategy]

        self.strategies[self.matching_triplet][self.followed_strategy] += \
            self.alpha * delta

        if not self.consumption:

            forward_value = max(self.strategies[(self.H, partner_type, partner_good)]) \
                - self.strategies[self.matching_triplet][self.followed_strategy]

            self.strategies[self.matching_triplet][self.followed_strategy] += \
                self.gamma * forward_value

    def select_strategy(self, partner_good, partner_type):

        relevant_strategies_values = self.strategies[(self.H,  partner_type, partner_good)]
        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(relevant_strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(relevant_strategies_values)), p=p_values)

        # Memory for learning
        self.matching_triplet = self.H, partner_type, partner_good

    # ----------  FOR OPTIMIZATION PART ---------- #
    
    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        self.learn(partner_good=partner_good, partner_type=partner_type)

        relevant_strategies_values = self.strategies[(self.H, partner_type, partner_good)]
        p_values = softmax(relevant_strategies_values, self.temp)
        
        # Assume there is only 2 p-values, return the one corresponding to the choice of the subject
        return p_values[subject_response]
    
    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        # Memory for learning
        self.matching_triplet = self.H, partner_type, partner_good

        self.followed_strategy = subject_choice
        
        if subject_choice and partner_choice:
            
            self.H = partner_good

        self.consume()


def main():

    storing_costs = np.array([0.01, 0.04, 0.09])  # 5
    u = 1

    parameters = {
        "t_max": 500,
        "agent_parameters": {"alpha": 0.2, "temp": 0.01, "gamma": 0.2,
                             "q_values": np.ones((12, 2))},
        "repartition_of_roles": np.array([500, 500, 500]),
        "storing_costs": storing_costs,
        "u": u,
        "agent_model": ForwardRLAgent,
    }

    e = Economy(**parameters)

    backup = e.run()

    # backup["last_strategies"] = [agent.strategies for agent in e.agents]

    # save(backup)

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":

    main()
