import numpy as np
import itertools as it
import pickle
from os import path
from datetime import datetime
from tqdm import tqdm

from environment.Economy import EconomyWithoutBackUp
from environment.compute_equilibrium import compute_equilibrium
from agent.FrequentistAgent import FrequentistAgent


class EconomyForOptimizing(EconomyWithoutBackUp):

    """ Economy class for optimizing parameters"""

    def __init__(self, **parameters):

        super().__init__(**parameters)

        self.good_accepted_as_medium_at_t = np.zeros(self.n_goods)
        self.proposition_of_medium_at_t = np.zeros(self.n_goods)

        self.good_accepted_as_medium_average = np.zeros((self.t_max, self.n_goods))

        self.t = 0

    def run(self):

        self.agents = self.create_agents()

        for t in range(self.t_max):
            self.t = t
            self.time_step()

        return self.give_feed_back()

    def time_step(self):

        """
         Overrided method allowing for backup
        :return: None
        """

        self.reinitialize_backup_containers()

        super().time_step()

        self.make_a_backup_for_t()

    def give_feed_back(self):

        to_return = np.array([
            np.mean(self.good_accepted_as_medium_average[-200:, 0]),
            np.mean(self.good_accepted_as_medium_average[-200:, 1]),
            np.mean(self.good_accepted_as_medium_average[-200:, 2]),
            self.storing_costs[2] - self.storing_costs[1]
        ])

        return to_return

    def make_encounter(self, i, j):

        """
         Overrided method allowing for backup
        :return: None
        """

        i_agreeing, j_agreeing = self.seek_agreement(i=i, j=j, proportions=None)
        self.make_stats_about_medium_of_exchange(i=i, j=j, i_agreeing=i_agreeing, j_agreeing=j_agreeing)
        self.proceed_to_exchange(i=i, j=j, i_agreeing=i_agreeing, j_agreeing=j_agreeing)

    def make_stats_about_medium_of_exchange(self, i, j, i_agreeing, j_agreeing):

        i_H, j_H = self.agents[i].H, self.agents[j].H
        i_P, j_P = self.agents[i].P, self.agents[j].P
        i_C, j_C = self.agents[i].C, self.agents[j].C

        # Consider particular case of offering third object
        i_facing_M = j_H != i_C and i_H == i_P
        j_facing_M = i_H != j_C and j_H == j_P

        if i_facing_M:
            self.proposition_of_medium_at_t[j_H] += 1  # Consider as key the good that is proposed as a medium of ex
            if i_agreeing:
                self.good_accepted_as_medium_at_t[j_H] += 1

        if j_facing_M:
            self.proposition_of_medium_at_t[i_H] += 1
            if j_agreeing:
                self.good_accepted_as_medium_at_t[i_H] += 1

    def reinitialize_backup_containers(self):

        self.good_accepted_as_medium_at_t[:] = 0
        self.proposition_of_medium_at_t[:] = 0

    def make_a_backup_for_t(self):

        for i in range(self.n_goods):

            # Avoid division by zero
            if self.proposition_of_medium_at_t[i] > 0:
                self.good_accepted_as_medium_at_t[i] = \
                    self.good_accepted_as_medium_at_t[i] / self.proposition_of_medium_at_t[i]

            else:
                self.good_accepted_as_medium_at_t[i] = 0

        assert 0 <= self.good_accepted_as_medium_at_t.all() <= 1

        # For back up
        self.good_accepted_as_medium_average[self.t][:] = self.good_accepted_as_medium_at_t


def fun_3_goods(storing_costs):

    t_max = 500
    u = 1
    beta = 0.9
    repartition_of_roles = np.array([50, 50, 50])
    storing_costs = np.asarray(storing_costs) / 100

    agent_parameters = {
        "acceptance_memory_span": 1000,
        "encounter_memory_span": 1000,
        "temp": 0.1,
    }

    parameters = {
        "t_max": t_max,
        "agent_parameters": agent_parameters,
        "repartition_of_roles": repartition_of_roles,
        "storing_costs": storing_costs,
        "u": u,
        "beta": beta,
        "agent_model": FrequentistAgent
    }

    e = EconomyForOptimizing(**parameters)

    return e.run()


def timestamp():

    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")


def optimize_3_goods():

    data_file_name = path.expanduser("~/Desktop/exp_parameters_optimization_by_hand_data.p")
    comb_file_name = path.expanduser("~/Desktop/exp_parameters_optimization_by_hand_comb.p")

    if path.exists(data_file_name):
        with open(data_file_name, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    if path.exists(comb_file_name):
        with open(comb_file_name, 'rb') as f:
            comb = pickle.load(f)
    else:
        comb = list(it.combinations(np.arange(1, 101), r=3))

    initial_len_comb = len(comb)

    try:
        for i in tqdm(range(initial_len_comb)):
            rand = np.random.randint(len(comb))
            c = comb[rand]

            if compute_equilibrium([c[0]/100, c[1]/100, c[2]/100], 0.9, 1) != "speculative":

                data[c] = fun_3_goods(c)

            else:
                data[c] = "non-speculative"

            print("{}: {}".format(c, data[c]))
            print()
            comb.remove(c)

    except KeyboardInterrupt:
        with open(data_file_name, "wb") as file:
            pickle.dump(data, file=file)

        with open(comb_file_name, "wb") as file:
            pickle.dump(comb, file=file)


if __name__ == "__main__":

    optimize_3_goods()

