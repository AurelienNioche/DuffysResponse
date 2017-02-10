import numpy as np
from Economy import launch
from analysis import represent_results
from KWModels import ModelA


class Agent(object):

    """
    Abstract class for agents
    """
    name = "Stupid agent"

    def __init__(self, prod, cons, third, storing_costs, u,  beta=0.9,
                 agent_parameters=None, kw_model=ModelA, idx=None):

        # Production object (integer in [0, 1, 2])
        self.P = prod

        # Consumption object (integer in [0, 1, 2])
        self.C = cons

        # Other object (integer in [0, 1, 2])
        self.T = third

        # Index of agent (more or less his name ; integer in [0, ..., n] with n : total number of agent)
        self.idx = idx

        # Parameters for agent that could be different in nature depending on the agent model in use (Python dictionary)
        self.agent_parameters = agent_parameters

        # Storing costs (numpy array of size 3) and utility derived from consumption
        self.storing_costs = np.asarray(storing_costs)
        self.u = u
        self.beta = beta

        # Keep a trace for time t if the agent consumed or not.
        self.consumption = 0

        # Keep a trace whether the agent proceed to an exchange
        self.exchange = None

        # Object an agent has in hand
        self.in_hand = self.P

        self.kw_model = kw_model

    def are_you_satisfied(self, proposed_object, type_of_other_agent, proportions):

        if proposed_object == self.C:
            return True
        else:
            return np.random.choice([True, False])

    def consume(self):

        self.consumption = self.in_hand == self.C

        if self.consumption:
            self.in_hand = self.P

    def proceed_to_exchange(self, new_object):

        if new_object is not None:
            self.exchange = True
            self.in_hand = new_object

        else:
            self.exchange = False

        # -------------- FITTING ------------------------- #

    def match_departure_good(self, subject_good):

        self.in_hand = subject_good

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        if partner_good == self.C:
            return subject_response == 1
        else:
            return 0.5

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        if subject_choice and partner_choice:

            self.in_hand = partner_good

            if self.in_hand == self.C:
                    self.in_hand = self.P


def main():

    parameters = {
        "t_max": 500,
        "agent_parameters": {"beta": 0.9, "u": 0.2},
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([0.01, 0.04, 0.09]),
        "kw_model": ModelA,
        "agent_model": Agent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)

if __name__ == "__main__":

    main()





