import numpy as np
from module.useful_functions import softmax
from stupid_agent import StupidAgent
from Economy import Economy
from graph import represent_results
import itertools as it


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
            "encounter": self.get_acceptance_or_encounter_dic(n_goods=self.n_goods),
            "acceptance": self.get_acceptance_or_encounter_dic(n_goods=self.n_goods)
        }

        self.memory = {
            "encounter": self.get_memory_dic(n_goods=self.n_goods),
            "acceptance": self.get_memory_dic(n_goods=self.n_goods)
        }

        self.in_hand_partner_good_pair = None
        self.accept = None

        if {"encounter_probabilities", "acceptance_probabilities"}.issubset(self.agent_parameters.keys()):

            self.set_initial_probabilities(
                self.agent_parameters["encounter_probabilities"],
                self.agent_parameters["acceptance_probabilities"]
            )

    @staticmethod
    def get_acceptance_or_encounter_dic(n_goods):

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

        self.in_hand_partner_good_pair = self.P, partner_good

        if partner_good == self.C:
            self.accept = 1

        elif partner_good == self.P:
            self.accept = 0

        else:

            self.accept = self.accept_a_medium(partner_good)

        self.learn_from_encounter()

        return self.accept

    def accept_a_medium(self, partner_good):

        v = np.zeros(2)

        # If refuses
        probability_direct_exchange = \
            self.probabilities["acceptance"][(self.P, self.C)] * \
            self.probabilities["encounter"][(self.P, self.C)]

        if probability_direct_exchange > 0:
            v[0] = \
                self.u - self.storing_costs[self.P] / \
                probability_direct_exchange
        else:
            v[0] = 0

        # If accepts
        probability_indirect_exchange = \
            self.probabilities["acceptance"][(partner_good, self.C)] * \
            self.probabilities["encounter"][(partner_good, self.C)]

        if probability_indirect_exchange > 0:
            v[1] = \
                self.u - self.storing_costs[partner_good] / \
                probability_indirect_exchange
        else:
            v[1] = 0

        v = np.tanh(v) * 2 - 1

        p = softmax(v, temp=self.temp)

        return np.random.choice(np.array([0, 1]), p=p)

    def consume(self):

        self.consumption = self.H == self.C

        if self.consumption:
            self.H = self.P

        self.learn_from_result()

    def learn_from_encounter(self):

        for k in self.probabilities["encounter"].keys():
            cond = int(k == self.in_hand_partner_good_pair)
            self.memory["encounter"][k].append(cond)
            if len(self.memory["encounter"][k]) > self.memory_span:
                self.memory["encounter"][k] = self.memory["encounter"][k][1:]
            self.probabilities["encounter"][k] = np.mean(self.memory["encounter"][k])

    def learn_from_result(self):

        if self.accept:

            successful = int(self.H != self.in_hand_partner_good_pair[0])
            self.memory["acceptance"][self.in_hand_partner_good_pair].append(successful)
            if len(self.memory["acceptance"][self.in_hand_partner_good_pair]) > self.memory_span:
                self.memory["acceptance"][self.in_hand_partner_good_pair] = \
                    self.memory["acceptance"][self.in_hand_partner_good_pair][1:]
            self.probabilities["acceptance"][self.in_hand_partner_good_pair] = \
                np.mean(self.memory["acceptance"][self.in_hand_partner_good_pair])



def main():

    storing_costs = [0.01, 0.04, 0.09]
    u = 1
    beta = 0.9

    agent_parameters = {
        "acceptance_memory_span": 250,
        "encounter_memory_span": 250,
        "temp": 0.01,
    }

    parameters = {
        "t_max": 100,
        "agent_parameters": agent_parameters,
        "repartition_of_roles": np.array([500, 500, 500]),
        "storing_costs": storing_costs,
        "u": u,
        "beta": beta,
        "agent_model": FrequentistAgent
    }

    e = Economy(**parameters)

    backup = e.run()

    represent_results(backup=backup, parameters=parameters)

if __name__ == "__main__":
    main()