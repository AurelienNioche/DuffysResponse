import numpy as np
from hyperopt import fmin, tpe, hp
from data_manager import import_data
from RLForward import RLForwardAgent


class PerformanceComputer(object):

    def __init__(self, subject_idx, raw_storing_costs, raw_u):

        self.data = import_data()[subject_idx]  # This is for subject one

        self.raw_storing_costs = raw_storing_costs
        self.raw_u = raw_u

        self.t_max = len(self.data["subject_good"])

    def run(self, *args):

        alpha, temp, gamma = args[0][:3]
        q_values = np.asarray(args[0][3:]).reshape((12, 2))

        m = RLForwardAgent(
            prod=self.data["subject_good"][0],
            cons=(self.data["subject_good"][0] - 1) % 3,  # Suppose we are in the KW's Model A
            third=(self.data["subject_good"][0] - 2) % 3,
            agent_type=(self.data["subject_good"][0] - 1) % 3,
            storing_costs=self.raw_storing_costs,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "gamma": gamma,
                "q_values": q_values,
                "u": self.raw_u
            }
        )

        log_likelihood_list = []

        for i in range(self.t_max):
            m.learn(partner_good=self.data["partner_good"][i], partner_type=self.data["partner_type"][i])

            likelihood = m.probability_of_responding(subject_response=self.data["subject_choice"][i],
                                                     partner_good=self.data["partner_good"][i],
                                                     partner_type=self.data["partner_type"][i])
            log_likelihood_list.append(np.log(likelihood))

            m.do_the_encounter(partner_choice=self.data["partner_choice"][i],
                               partner_type=self.data["partner_type"][i],
                               partner_good=self.data["partner_good"][i],
                               subject_choice=self.data["subject_choice"][i])

        result = - sum(log_likelihood_list)
        return result


def optimize(ind):

    storing_costs = np.asarray([1, 4, 9])
    u = 100

    pc = PerformanceComputer(subject_idx=ind, raw_storing_costs=storing_costs, raw_u=u)

    print()
    print("Begin optimization for ind {}...".format(ind))

    search_space = [
        hp.uniform("alpha", 0., 1.),
        hp.uniform("temp", 0.01, 1.),  # Minimum can not be 0!!!
        hp.uniform("gamma", 0., 1.)
    ]

    n_situations = 12

    for i in range(n_situations):
        for j in range(2):
            search_space.append(
                hp.uniform("q_{:02d}_{}".format(i, j), 0., 1.)
            )

    best = fmin(fn=pc.run,
                space=search_space,
                algo=tpe.suggest,
                max_evals=100)

    print("Optimization done!")
    print()

    return best


def main():

    n_situations = 12

    backup = dict()

    backup["alpha"] = []
    backup["temp"] = []
    backup["gamma"] = []

    for i in range(n_situations):
        for j in range(2):
            backup["q_{:02d}_{}".format(i, j)] = []

    for i in range(30):

        best = optimize(i)
        for key in best.keys():
            backup[key].append(best[key])

    for key in sorted(backup.keys()):

        r = "{}: {} +/- {}".format(key, np.mean(backup[key]), np.std(backup[key]))
        print(r)

if __name__ == "__main__":
    main()
