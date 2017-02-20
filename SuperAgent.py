import numpy as np
from module.useful_functions import softmax
from AbstractAgent import Agent
from Economy import Economy
from analysis import represent_results

from KWModels import ModelA, ModelB


class SuperAgent(Agent):

    name = "SuperAgent"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.alpha_encounter = 0.01
        self.alpha_acceptance = 0.01
        self.temp = 0.1

        self.encounter = {
            (self.P, self.C): 0.5,
            (self.T, self.C): 0.5
        }

        self.acceptance = {
            (self.P, self.C): 0.5,
            (self.T, self.C): 0.5
        }

        self.in_hand_partner_good_pair = None
        self.accept = None

        # if self.idx == 0:
        #     print("prod", self.P)
        #     print("cons", self.C)
        #     print("third", self.T)

    def are_you_satisfied(self, partner_good, partner_type, proportions, verbose=False):

        self.in_hand_partner_good_pair = self.in_hand, partner_good
        if self.idx == 0 and verbose:
            print("exchange proposed", self.in_hand_partner_good_pair)

        if partner_good == self.C:
            self.accept = 1

        elif partner_good == self.P:
            self.accept = 0

        else:

            raw_v_accept = \
                (self.acceptance[(self.T, self.C)] * self.encounter[(self.T, self.C)]) / \
                self.storing_costs[self.T]

            raw_v_refuse = \
                (self.acceptance[(self.P, self.C)] * self.encounter[(self.P, self.C)]) / \
                self.storing_costs[self.P]

            # Normalise values for softmax
            v_accept = min(1, raw_v_accept/raw_v_refuse)
            v_refuse = min(1, raw_v_refuse/raw_v_accept)

            p = softmax(np.array([v_refuse, v_accept]), temp=self.temp)

            if self.idx == 0 and verbose:
                print("v_a", v_accept, "v_r", v_refuse)
                print("p", p)

            self.accept = np.random.choice(np.array([0, 1]), p=p)

        for k in self.encounter.keys():
            cond = int(k == self.in_hand_partner_good_pair)
            self.encounter[k] += self.alpha_encounter * (cond - self.encounter[k])

        # self.alpha_encounter **= 1.01
        if self.idx == 0 and verbose:

            print("accept", self.accept)
            print(self.encounter)
            print(self.acceptance)
            print()

        return self.accept

    def consume(self):

        super().consume()

        if self.in_hand_partner_good_pair in self.acceptance.keys() and self.accept:
            # print("Hey")
            successful = int(self.in_hand != self.in_hand_partner_good_pair[0])
            # print("successful", successful)
            self.acceptance[self.in_hand_partner_good_pair] += \
                self.alpha_acceptance * (successful - self.acceptance[self.in_hand_partner_good_pair])

        # self.alpha_acceptance **= 1.01


def main():

    storing_costs = np.array([0.01, 0.085, 0.09])
    u = 1
    beta = 0.9

    parameters = {
        "t_max": 100,
        "agent_parameters": {},
        "role_repartition": np.array([500, 500, 500]),
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