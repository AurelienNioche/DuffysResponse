import numpy as np
from module.useful_functions import softmax
from AbstractAgent import Agent
from Economy import Economy
from analysis import represent_results

from KWModels import ModelA


class SuperAgent(Agent):

    name = "SuperAgent"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.alpha_encounter = 0.05
        self.alpha_acceptance = 0.05
        self.temp = 0.01

        self.encounter = {
            (self.P, self.C): 1,
            (self.T, self.C): 1
        }

        self.acceptance = {
            (self.P, self.C): 1,
            (self.T, self.C): 1
        }

        self.in_hand_partner_good_pair = None
        self.accept = None

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        self.in_hand_partner_good_pair = self.in_hand, partner_good

        if partner_good == self.C:
            self.accept = 1

        elif partner_good == self.P:
            self.accept = 0

        else:

            raw_v_accept = self.storing_costs[self.T] / \
                   (self.acceptance[(self.T, self.C)] * self.encounter[(self.T, self.C)])
            raw_v_refuse = self.storing_costs[self.P] / \
                   (self.acceptance[(self.P, self.C)] * self.encounter[(self.P, self.C)])

            # Normalise values for softmax
            v_accept = min(1, raw_v_accept/raw_v_refuse)
            v_refuse = min(1, raw_v_refuse/raw_v_accept)
            p = softmax(np.array([v_refuse, v_accept]), temp=self.temp)

            self.accept = np.random.choice([0, 1], p=p)

        for k in self.encounter.keys():
            cond = int(k == self.in_hand_partner_good_pair)
            self.encounter[k] += self.alpha_encounter * (cond - self.encounter[k])

        self.alpha_encounter **= 1.01
        return self.accept

    def consume(self):

        super().consume()

        if self.in_hand_partner_good_pair in self.acceptance.keys() and self.accept:

            successful = int(self.in_hand != self.in_hand_partner_good_pair[0])
            self.acceptance[self.in_hand_partner_good_pair] += \
                self.alpha_acceptance * (successful - self.acceptance[self.in_hand_partner_good_pair])

        self.alpha_acceptance **= 1.01


def compute_equilibrium(storing_costs, u, beta):

    if (storing_costs[2] - storing_costs[1]) < (2**0.5 - 1) * (beta/3) * u:
        return "speculative", storing_costs[2] - storing_costs[1], (2**0.5 - 1) * (beta/3) * u

    elif (storing_costs[2] - storing_costs[1]) > 0.5 * (beta/3) * u:

        return 'fundamental'

    else:
        return "no equilibrium"


def main():

    storing_costs = np.array([0.01, 0.04, 0.09])
    u = 1
    beta = 0.9

    print(compute_equilibrium(storing_costs, u, beta))

    parameters = {
        "t_max": 250,
        "agent_parameters": {},
        "role_repartition": np.array([1000, 1000, 1000]),
        "storing_costs": storing_costs,
        "u": u,
        "beta": beta,
        "agent_model": SuperAgent,
        "kw_model": ModelA,
    }

    e = Economy(**parameters)

    backup = e.run()

    represent_results(backup=backup, parameters=parameters)

if __name__ == "__main__":
    main()