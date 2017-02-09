import numpy as np
from AbstractAgent import Agent
from KWModels import ModelA
from module.useful_functions import softmax
from Economy import launch
from analysis import represent_results


'''
RL with reinforcement of strategies understood as Game Theory does
 (a strategy is a set of action plans for every possible situation)
'''


class RLAgent(Agent):

    name = "RL"

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

        self.absolute_to_relative = absolute_to_relative_3_types[self.C]

        # ----- RL PARAMETERS ---- #

        self.alpha = self.agent_parameters["alpha"]
        self.temp = self.agent_parameters["temp"]

        self.strategies_values = self.agent_parameters["strategy_values"]

        self.u, self.storing_costs = self.define_u_and_storing_costs(self.u, self.storing_costs)

    @staticmethod
    def define_u_and_storing_costs(u, storing_costs):

        # To be sure that q values will be remained between 0 and 1.
        amplitude = u - min(storing_costs) + max(storing_costs)

        new_storing_costs = np.zeros(3)
        new_storing_costs[:] = storing_costs[:] / amplitude

        new_u = u/amplitude

        return new_u, new_storing_costs

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

    # ------------------------ RL PART ------------------------------------------------------ #

    def compute_utility(self):

        self.utility = \
            self.consumption - self.storing_costs[self.in_hand]

        # Maybe we want that to be bounded between 0 and 1 with something like 0.5 + consumption/2 - storing cost
        # Be sure that utility lies between 0 and 1
        # assert 0 <= self.utility <= 1

    def learn(self):

        # 'Classic' RL rule
        self.strategies_values[self.followed_strategy] += \
            self.alpha * (self.utility - self.strategies_values[self.followed_strategy])

    def select_strategy(self):

        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(self.strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(self.strategies_values)), p=p_values)

    # ---------- OPTIMIZATION PART ---------- #

    def probability_of_responding(self, subject_response, partner_good):

        compatible = self.strategies[:,
                self.absolute_to_relative[self.in_hand],
                self.absolute_to_relative[partner_good]
            ] == subject_response

        p_values = softmax(self.strategies_values, self.temp)
        return sum(p_values[compatible])

    def do_the_encounter(self, subject_choice, partner_choice, partner_good):

        self.followed_strategy = subject_choice

        if subject_choice and partner_choice:
            self.in_hand = partner_good

        self.consume()  # Include learning in this model


def test_agent():

    a = RLAgent(
        prod=1,
        cons=0,
        third=2,
        agent_parameters={
            "alpha": 0.5,
            "temp": 0.5,
            "strategy_values": np.zeros(4)
        },
        storing_costs=np.array([1, 4, 9]),
        u=100
    )

    a.probability_of_responding(subject_response=1, partner_good=0)


def main():

    parameters = {
        "t_max": 500,
        "agent_parameters": {"alpha": 0.5, "temp": 0.01},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([1, 4, 9]),
        "u": 100,
        "kw_model": ModelA,
        "agent_model": RLAgent
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":

    test_agent()

