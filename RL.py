import numpy as np
from stupid_agent import StupidAgent
from module.useful_functions import softmax
from Economy import launch
from graph import represent_results
from get_roles import get_roles

'''
RL with reinforcement of strategies understood as Game Theory does
 (a strategy is a set of action plans for every possible situation)
'''


class RLAgent(StupidAgent):

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

        self.roles = get_roles(len(self.storing_costs))

        # Take object with absolute reference to give object relating to agent
        #    (with 0: production good, 1: consumption good, 2: third object)

        self.absolute_to_relative = self.get_absolute_to_relative()

        # ----- RL PARAMETERS ---- #

        self.alpha = self.agent_parameters["alpha"]
        self.temp = self.agent_parameters["temp"]

        self.strategies_values = self.agent_parameters["strategy_values"].copy()

        self.u, self.storing_costs = self.define_u_and_storing_costs(self.u, self.storing_costs)

    def get_absolute_to_relative(self):

        to_return = np.zeros(3, dtype=int)
        third = [i for i in range(3) if i != self.P and i != self.C][0]

        to_return[self.P] = 0
        to_return[self.C] = 1
        to_return[third] = 2

        return to_return

    @staticmethod
    def define_u_and_storing_costs(u, storing_costs):

        # To be sure that q values will be remained between 0 and 1.
        amplitude = u - min(storing_costs) + max(storing_costs)

        new_storing_costs = np.zeros(3)
        new_storing_costs[:] = storing_costs[:] / amplitude

        new_u = u/amplitude

        return new_u, new_storing_costs

    # ------------------------ SURCHARGED METHODS ------------------------------------------------------ #

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        self.select_strategy()
        agreeing = self.strategies[self.followed_strategy, 
                                   self.absolute_to_relative[self.H], 
                                   self.absolute_to_relative[partner_good]]

        return agreeing

    def consume(self):

        # Call 'consume' method from parent (that is 'Agent')
        super().consume()

        self.learn()

    # ------------------------ RL PART ------------------------------------------------------ #

    def compute_utility(self):

        utility = \
            max(self.storing_costs) + self.consumption * self.u - self.storing_costs[self.H]

        # Maybe we want that to be bounded between 0 and 1 with something like 0.5 + consumption/2 - storing cost
        # Be sure that utility lies between 0 and 1
        assert 0 <= utility <= 1

        return utility

    def learn(self):

        # 'Classic' RL rule
        self.strategies_values[self.followed_strategy] += \
            self.alpha * (self.compute_utility() - self.strategies_values[self.followed_strategy])

    def select_strategy(self):

        # Obtain probability of using this or that strategy by a softmax,
        # and then select a strategy according to these probabilities
        p_values = softmax(self.strategies_values, self.temp)
        self.followed_strategy = np.random.choice(np.arange(len(self.strategies_values)), p=p_values)

    # ---------- OPTIMIZATION PART ---------- #

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        compatible = \
            self.strategies[
                :,
                self.absolute_to_relative[self.H],
                self.absolute_to_relative[partner_good]
            ] == subject_response

        p_values = softmax(self.strategies_values, self.temp)
        return sum(p_values[compatible])

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        self.followed_strategy = subject_choice

        if subject_choice and partner_choice:
            self.H = partner_good

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

    a.probability_of_responding(subject_response=1, partner_good=0, partner_type=None, proportions=None)


def main():

    parameters = {
        "t_max": 500,
        "agent_parameters": {"alpha": 0.2, "temp": 0.01,
                             "strategy_values": np.ones(4)},
        "repartition_of_roles": [500, 500, 500],
        "storing_costs": [0.01, 0.03, 0.09],
        "u": 1,
        "agent_model": RLAgent
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":

    main()

