import numpy as np
import itertools as it
import pickle
from os import path
from datetime import datetime
from tqdm import tqdm

from environment.Economy import EconomyWithoutBackUp
from environment.compute_equilibrium import compute_equilibrium
from agent.FrequentistAgent import FrequentistAgent

from multiprocessing import Process, Queue, Event, cpu_count


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


class Computer(Process):

    def __init__(self, int_name, input_queue, output_queue, shutdown):

        super().__init__()
        self.int_name = int_name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown = shutdown

    def run(self):

        while not self.shutdown.is_set():

            raw_storing_costs = self.input_queue.get()
            storing_costs = np.asarray(raw_storing_costs) / 100
            if compute_equilibrium(storing_costs, 0.9, 1) == "speculative":
                res = self.fun_3_goods(storing_costs)
            else:
                res = "non-speculative"
            self.output_queue.put((self.int_name, raw_storing_costs, res))

    @staticmethod
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


class Optimizer(object):

    data_file_name = path.expanduser(
        "~/Desktop/exp_parameters_optimization_by_hand_data.p")
    comb_file_name = path.expanduser(
        "~/Desktop/exp_parameters_optimization_by_hand_comb.p")

    def __init__(self):

        self.shutdown = Event()
        self.queue = Queue()
        self.data, self.comb = self.load()
        self.processes, self.processes_queues = self.create_processes()

    def load(self):

        if path.exists(self.data_file_name):
            with open(self.data_file_name, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}

        if path.exists(self.comb_file_name):
            with open(self.comb_file_name, 'rb') as f:
                comb = pickle.load(f)
        else:
            comb = list(it.combinations(np.arange(1, 101), r=3))

        return data, comb

    def create_processes(self):

        processes = []
        queues = []
        for i in range(cpu_count()):

            queue = Queue()
            process = Computer(input_queue=queue, output_queue=self.queue, int_name=i, shutdown=self.shutdown)

            processes.append(process)
            queues.append(queue)

        return processes, queues

    def start_processes(self):

        for i in range(cpu_count()):

            if len(self.comb):

                self.processes[i].start()

                num = np.random.randint(len(self.comb))
                self.processes_queues[i].put(self.comb[num])

            else:
                self.shutdown.set()
                break

    def run(self):

        p_bar = tqdm(total=len(self.comb))

        self.start_processes()

        while not self.shutdown.is_set():

            process_name, process_comb, process_result = self.queue.get()
            self.data[process_comb] = process_result

            if not self.shutdown.is_set() and len(self.comb) > 1:
                if type(process_result) is not str and process_result[2] > 0.01:

                    num = self.comb.index(process_comb)

                    if num == len(self.comb) - 1:
                        num -= 1
                else:
                    num = np.random.randint(len(self.comb)-1)

                self.comb.remove(process_comb)
                new_comb = self.comb[num]
                self.processes_queues[process_name].put(new_comb)

            else:
                self.comb.remove(process_comb)

            p_bar.update(1)

            if not len(self.comb):
                break
        self.finish()

    def finish(self):

        with open(self.data_file_name, "wb") as file:
            pickle.dump(self.data, file=file)

        with open(self.comb_file_name, "wb") as file:
            pickle.dump(self.comb, file=file)


def optimize_3_goods():

    op = Optimizer()

    try:
        op.run()

    except KeyboardInterrupt:
        op.shutdown.set()
        op.finish()


if __name__ == "__main__":

    optimize_3_goods()

