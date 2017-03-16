import numpy as np
from module.useful_functions import softmax
from stupid_agent import StupidAgent
from Economy import Economy
from graph import represent_results
import itertools as it
from compute_equilibrium import compute_equilibrium


DEBUG = 0
IDX_FOR_DEBUG = 0


class FrequentistAgent(StupidAgent):

    name = "Frequentist Agent"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.n_goods = len(self.storing_costs)

        self.memory_span = {
            "encounter": self.agent_parameters["encounter_memory_span"],
            "acceptance": self.agent_parameters["acceptance_memory_span"]
        }
        self.temp = self.agent_parameters["temp"]

        self.probabilities = {
            "encounter": dict([(i, 1/self.n_goods) for i in range(self.n_goods)]),
            "acceptance": self.get_acceptance_dic(n_goods=self.n_goods)
        }

        self.memory = {
            "encounter": dict([(i, []) for i in range(self.n_goods)]),
            "acceptance": self.get_memory_dic(n_goods=self.n_goods)
        }

        self.in_hand_partner_good_pair = None
        self.accept = None

        if {"encounter_probabilities", "acceptance_probabilities"}.issubset(self.agent_parameters.keys()):

            self.set_initial_probabilities(
                self.agent_parameters["encounter_probabilities"],
                self.agent_parameters["acceptance_probabilities"]
            )

        self.u, self.storing_costs = 1, self.storing_costs/self.u

    @staticmethod
    def get_acceptance_dic(n_goods):

        to_return = dict()

        for i in it.permutations(range(n_goods), r=2):
            to_return[i] = 1.

        return to_return

    @staticmethod
    def get_memory_dic(n_goods):

        memory = dict()
        for i in it.permutations(range(n_goods), r=2):
            memory[i] = []

        return memory

    def set_initial_probabilities(self, initial_encounter_probabilities, initial_acceptance_probabilities):

        for i, key in enumerate(sorted(self.probabilities["encounter"].keys())):

            self.probabilities["encounter"][key] = initial_encounter_probabilities[i]
            self.probabilities["acceptance"][key] = initial_acceptance_probabilities[i]

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        self.in_hand_partner_good_pair = self.H, partner_good

        p_values = self.get_p_values(partner_good)

        self.accept = np.random.choice([0, 1], p=p_values) if p_values[1] not in [0, 1] else p_values[1]

        self.learn_from_encounter()

        return self.accept

    def get_p_values(self, partner_good):

        if self.idx == IDX_FOR_DEBUG and DEBUG:
            print()
            print("type", self.C)
            print('H', self.H)
            print("partner_good", partner_good)

        if partner_good == self.C:
            p_values = [0, 1]

        elif partner_good == self.P or partner_good == self.H:
            p_values = [1, 0]

        else:

            p_values = self.accept_a_medium(partner_good)

        return p_values

    def accept_a_medium(self, partner_good):

        v = np.zeros(2)

        # If refuses
        probability_direct_exchange = \
            self.probabilities["acceptance"][(self.P, self.C)] * self.probabilities["encounter"][self.C]

        # DEBUG ONLY
        if self.idx == IDX_FOR_DEBUG and DEBUG: print("p direct exchange", probability_direct_exchange)

        if probability_direct_exchange > 0:
            v[0] = \
                max(0,
                    self.u - self.storing_costs[self.P] /
                    probability_direct_exchange
                    )
        else:
            v[0] = 0

        # If accepts
        probability_indirect_exchange = \
            self.probabilities["acceptance"][(partner_good, self.C)] * self.probabilities["encounter"][self.C]

        # DEBUG ONLY
        if self.idx == IDX_FOR_DEBUG and DEBUG: print("p indirect exchange", probability_indirect_exchange)

        if probability_indirect_exchange > 0:
            v[1] = \
                max(0,
                    self.u - self.storing_costs[partner_good] /
                    probability_indirect_exchange
                    )
        else:
            v[1] = 0

        if v[0] == v[1] == 0:

            p_accept = int(self.storing_costs[partner_good] < self.storing_costs[self.P])
            p_values = [1-p_accept, p_accept]

        else:

            p_values = softmax(v, temp=self.temp)

            if self.idx == IDX_FOR_DEBUG and DEBUG:
                print("v", v)
                print("p", p_values)

        return p_values

    def consume(self):

        self.consumption = self.H == self.C

        if self.consumption:
            self.H = self.P

        self.learn_from_result()

    def learn_from_encounter(self):

        for k in self.probabilities["encounter"].keys():
            cond = int(k == self.in_hand_partner_good_pair[1])
            self.memory["encounter"][k].append(cond)
            if len(self.memory["encounter"][k]) > self.memory_span["encounter"]:
                self.memory["encounter"][k] = self.memory["encounter"][k][1:]
            self.probabilities["encounter"][k] = np.mean(self.memory["encounter"][k])

        if self.idx == IDX_FOR_DEBUG and DEBUG:
            print("encounter memory:", self.memory["encounter"])
            print("encounter probabilities:", self.probabilities["encounter"])

    def learn_from_result(self):

        if self.idx == IDX_FOR_DEBUG and DEBUG:
            print("ACCEPT", self.accept)

        if self.accept and self.in_hand_partner_good_pair in self.memory["acceptance"]:

            successful = int(self.H != self.in_hand_partner_good_pair[0])
            if self.idx == IDX_FOR_DEBUG and DEBUG: print("Success", successful)
            self.memory["acceptance"][self.in_hand_partner_good_pair].append(successful)
            if len(self.memory["acceptance"][self.in_hand_partner_good_pair]) > self.memory_span["acceptance"]:
                self.memory["acceptance"][self.in_hand_partner_good_pair] = \
                    self.memory["acceptance"][self.in_hand_partner_good_pair][1:]
            self.probabilities["acceptance"][self.in_hand_partner_good_pair] = \
                np.mean(self.memory["acceptance"][self.in_hand_partner_good_pair])

            if self.idx == IDX_FOR_DEBUG and DEBUG:

                print("acceptance memory: ", self.memory["acceptance"])
                print("acceptance probabilities:", self.probabilities["acceptance"])

    # -------------- FITTING ------------------------- #

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):

        self.in_hand_partner_good_pair = self.H, partner_good

        self.accept = subject_response

        p_values = self.get_p_values(partner_good)

        self.learn_from_encounter()

        return p_values[subject_response]

    def do_the_encounter(self, subject_choice, partner_choice, partner_good, partner_type):

        self.learn_from_result()

        super().do_the_encounter(subject_choice, partner_choice, partner_good, partner_type)


def main():

    storing_costs = [0.1, 0.20, 0.22]
    u = 1
    beta = 0.9

    agent_parameters = {
        "acceptance_memory_span": 1000,
        "encounter_memory_span": 1000,
        "temp": 0.1,
    }

    parameters = {
        "t_max": 1000,
        "agent_parameters": agent_parameters,
        "repartition_of_roles": np.array([50, 50, 50]),
        "storing_costs": storing_costs,
        "u": u,
        "beta": beta,
        "agent_model": FrequentistAgent
    }

    expected_equilibrium = compute_equilibrium(storing_costs=storing_costs, beta=beta, u=u)
    print("Expected equilibrium is: {}".format(expected_equilibrium))

    e = Economy(**parameters)

    backup = e.run()

    represent_results(backup=backup, parameters=parameters)

if __name__ == "__main__":
    main()
