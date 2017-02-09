import numpy as np
import hyperopt as op
from data_manager import import_data
from RLForward import RLForwardAgent
from RL import RLAgent as RLStrategicAgent


class PerformanceComputer(object):

    def __init__(self, individual_data, raw_storing_costs, raw_u, model):

        self.data = individual_data

        self.raw_storing_costs = raw_storing_costs
        self.raw_u = raw_u

        self.t_max = len(self.data["subject_good"])

        self.model = model

        self.func = {
            "RLForward": self.run_forward_RL,
            "RLStrategic": self.run_strategic_RL,
            "Stupid": self.run_stupid
        }

    def run(self, *args):

        return self.func[self.model](*args)

    def run_stupid(self, *args):

        pass

    def run_forward_RL(self, *args):

        alpha, temp, gamma = args[0][:3]
        q_values = np.asarray(args[0][3:]).reshape((12, 2))

        m = RLForwardAgent(
            prod=self.data["subject_good"][0],
            cons=(self.data["subject_good"][0] - 1) % 3,  # Suppose we are in the KW's Model A
            third=(self.data["subject_good"][0] - 2) % 3,
            u=self.raw_u,
            storing_costs=self.raw_storing_costs,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "gamma": gamma,
                "q_values": q_values
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

    def run_strategic_RL(self, *args):

        alpha, temp = args[0][:2]
        strategy_values = np.asarray(args[0][2:])

        m = RLStrategicAgent(
            prod=self.data["subject_good"][0],
            cons=(self.data["subject_good"][0] - 1) % 3,  # Suppose we are in the KW's Model A
            third=(self.data["subject_good"][0] - 2) % 3,
            storing_costs=self.raw_storing_costs,
            u=self.raw_u,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "strategy_values": strategy_values,
            }
        )

        log_likelihood_list = []

        for i in range(self.t_max):

            likelihood = m.probability_of_responding(subject_response=self.data["subject_choice"][i],
                                                     partner_good=self.data["partner_good"][i])

            if likelihood > 0:
                perf = np.log(likelihood)
            else:
                perf = np.log(0.001)
            log_likelihood_list.append(perf)

            m.do_the_encounter(partner_choice=self.data["partner_choice"][i],
                               partner_good=self.data["partner_good"][i],
                               subject_choice=self.data["subject_choice"][i])

        result = - sum(log_likelihood_list)
        return result


class Optimizer(object):

    def __init__(self):

        self.storing_costs = np.asarray([1, 4, 9])
        self.u = 100

        self.data = import_data()

    def optimize(self, ind, model, search_space):

        print("Begin optimization for ind {} with {}...".format(ind, model))

        pc = PerformanceComputer(
            individual_data=self.data[ind],
            raw_storing_costs=self.storing_costs,
            raw_u=self.u,
            model=model)

        alg = op.partial(op.tpe.suggest,  # bayesian optimization
                          # tpe.rand, #random optimization
                          n_startup_jobs=100)
        best = op.fmin(
            fn=pc.run,
            space=search_space,
            algo=alg,
            max_evals=500)

        print("Optimization done!")
        print()

        return best

    def optimize_with_RLForward(self, ind):

        model = "RLForward"

        search_space = [
            op.hp.uniform("alpha", 0., 1.),
            op.hp.uniform("temp", 0.01, 1.),  # Minimum can not be 0!!!
            op.hp.uniform("gamma", 0., 1.)
        ]

        n_situations = 12

        for i in range(n_situations):
            for j in range(2):
                search_space.append(
                    op.hp.uniform("q_{:02d}_{}".format(i, j), 0., 1.)
                )

        best = self.optimize(ind=ind, model=model, search_space=search_space)

        return best

    def optimize_for_n_agents_with_RLForward(self):

        backup = dict()

        backup["alpha"] = []
        backup["temp"] = []
        backup["gamma"] = []

        n_situations = 12

        for i in range(n_situations):
            for j in range(2):
                backup["q_{:02d}_{}".format(i, j)] = []

        for i in range(30):

            best = self.optimize_with_RLForward(i)
            for key in best.keys():
                backup[key].append(best[key])

        for key in sorted(backup.keys()):
            r = "{}: {} +/- {}".format(key, np.mean(backup[key]), np.std(backup[key]))
            print(r)

    def optimize_with_RLStrategic(self, ind):

        model = "RLStrategic"

        search_space = [
            op.hp.uniform("alpha", 0., 1.),
            op.hp.uniform("temp", 0.01, 1.),  # Minimum can not be 0!!!
        ]

        n_strategies = 4

        for i in range(n_strategies):
            search_space.append(
                op.hp.uniform("s_{:02d}".format(i), 0., 1.)
            )

        best = self.optimize(ind=ind, model=model, search_space=search_space)

        return best

    def optimize_for_n_agents_with_RLStrategic(self):

        backup = dict()

        backup["alpha"] = []
        backup["temp"] = []

        n_strategies = 4

        for i in range(n_strategies):
            backup["s_{:02d}".format(i)] = []

        for i in range(30):

            best = self.optimize_with_RLStrategic(i)
            for key in best.keys():
                backup[key].append(best[key])

        for key in sorted(backup.keys()):
            r = "{}: {} +/- {}".format(key, np.mean(backup[key]), np.std(backup[key]))
            print(r)


def main():

    optimizer = Optimizer()
    optimizer.optimize_for_n_agents_with_RLStrategic()
    # optimizer.optimize_for_n_agents_with_RLForward()

if __name__ == "__main__":
    main()
