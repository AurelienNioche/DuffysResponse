import numpy as np
from Economy import launch
from graph import represent_results


class StupidAgent(object):

    """
    Abstract class for agents
    """
    name = "Stupid agent"

    def __init__(self, prod, cons, storing_costs, u=1,  beta=0.9,
                 agent_parameters=None, idx=None):

        # Production object (integer in [0, 1, 2])
        self.P = prod

        # Consumption object (integer in [0, 1, 2])
        self.C = cons

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
        self.H = self.P

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        if partner_good == self.C:
            return True
        else:
            return np.random.choice([True, False])

    def consume(self):

        self.consumption = self.H == self.C

        if self.consumption:
            self.H = self.P

    def proceed_to_exchange(self, new_object):

        if new_object is not None:
            self.exchange = True
            self.H = new_object

        else:
            self.exchange = False

        # -------------- FITTING ------------------------- #

    def match_departure_good(self, subject_good):

        self.H = subject_good

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        if partner_good == self.C:
            return subject_response == 1
        else:
            return 0.5

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        if subject_choice and partner_choice:

            self.H = partner_good

            if self.H == self.C:
                    self.H = self.P


def main():

    parameters = {
        "t_max": 500,
        "agent_parameters": {"beta": 0.9, "u": 0.2},
        "repartition_of_roles": np.array([500, 500, 500]),
        "storing_costs": np.array([0.01, 0.04, 0.09]),
        "agent_model": StupidAgent,
    }

    backup = \
        launch(
            **parameters
        )

    represent_results(backup=backup, parameters=parameters)

if __name__ == "__main__":

    main()





