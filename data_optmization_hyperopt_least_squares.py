import numpy as np
import hyperopt as op
import csv
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from data_manager import import_data
from RLForward import RLForwardAgent
from RLForward2 import RLForwardAgent2
from RL import RLAgent as RLStrategicAgent
from stupidy_is_better import TotalGogol, StupidAgent
from DuffyAgent import DuffyAgent
from KwAgent import KwAgent


class PerformanceComputer(object):

    def __init__(self, individual_data, model):

        self.data = individual_data

        self.raw_storing_costs = self.data["storing_costs"]
        self.raw_u = self.data["u"]
        self.beta = self.data["beta"]

        self.prod = self.data["subject_good"][0]
        self.cons = (self.data["subject_good"][0] - 1) % 3
        self.third = (self.data["subject_good"][0] - 2) % 3

        self.t_max = len(self.data["subject_good"])

        self.model = model

        self.func = {
            "RLForward": self.get_FRL_model,
            "RLForward2": self.get_FRL2_model,
            "RLStrategic": self.get_SRL_model,
            "NonParametrizedAgent": self.get_non_parametrized_model,
        }

        self.non_parametrized_model = {
            "TotalGogol": TotalGogol, "StupidAgent": StupidAgent,
            "Duffy": DuffyAgent, "KW": KwAgent
        }

    def run(self, *args):

        model = self.func[self.model](*args)
        squares_sum = self.compute_sum_errors_squares(model)
        return squares_sum

    def get_FRL_model(self, *args):

        alpha, temp, gamma = args[0][:3]
        q_values = np.asarray(args[0][3:]).reshape((12, 2))

        model = RLForwardAgent(
            prod=self.prod,
            cons=self.cons,  # Suppose we are in the KW's Model A
            third=self.third,
            u=self.raw_u,
            beta=self.beta,
            storing_costs=self.raw_storing_costs,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "gamma": gamma,
                "q_values": q_values
            }
        )

        return model

    def get_FRL2_model(self, *args):

        alpha, temp, gamma = args[0][:3]
        q_values = np.asarray(args[0][3:]).reshape((6, 2))

        model = RLForwardAgent2(
            prod=self.prod,
            cons=self.cons,  # Suppose we are in the KW's Model A
            third=self.third,
            u=self.raw_u,
            beta=self.beta,
            storing_costs=self.raw_storing_costs,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "gamma": gamma,
                "q_values": q_values
            }
        )

        return model

    def get_SRL_model(self, *args):

        try:

            alpha, temp = args[0][:2]
            strategy_values = np.asarray(args[0][2:])
        except:
            print(args)
            raise Exception

        model = RLStrategicAgent(
            prod=self.prod,
            cons=self.cons,  # Suppose we are in the KW's Model A
            third=self.third,
            storing_costs=self.raw_storing_costs,
            u=self.raw_u,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "strategy_values": strategy_values,
            }
        )

        return model

    def get_non_parametrized_model(self, args):

        model_name = args[0][0]

        model = self.non_parametrized_model[model_name](
                prod=self.prod,
                cons=self.cons,  # Suppose we are in the KW's Model A
                third=self.third,
                storing_costs=self.raw_storing_costs,
                u=self.raw_u,
                beta=self.beta
            )

        return model

    def compute_sum_errors_squares(self, model):

        squares_list = []

        for t in range(self.t_max):

            model.match_departure_good(subject_good=self.data["subject_good"][t])

            likelihood = model.probability_of_responding(
                subject_response=self.data["subject_choice"][t],
                partner_good=self.data["partner_good"][t],
                partner_type=self.data["partner_type"][t],
                proportions=self.data["prop"][t])

            error = 1 - likelihood
            squares_list.append(error ** 2)

            model.do_the_encounter(
                partner_choice=self.data["partner_choice"][t],
                partner_type=self.data["partner_type"][t],
                partner_good=self.data["partner_good"][t],
                subject_choice=self.data["subject_choice"][t])

        result = sum(squares_list)
        return result

    def evaluate(self, *args):

        squares_sum = self.run(args)

        if "NonParametrizedAgent" in args:
            degrees_of_freedom = 0
        else:
            degrees_of_freedom = len(args)

        return squares_sum, self.bic_formula(
            squares_sum=squares_sum,
            n_trials=self.t_max, degrees_of_freedom=degrees_of_freedom)

    @staticmethod
    def bic_formula(squares_sum, n_trials, degrees_of_freedom):

        return n_trials * np.log(squares_sum / n_trials) + np.log(n_trials) * degrees_of_freedom


class Optimizer(object):

    def __init__(self, data, subjects_idx=None):

        self.data = data

        if subjects_idx is not None:
            self.subjects_idx = subjects_idx
        else:
            self.subjects_idx = np.arange(len(self.data))

        # ------ Optimization parameters ------- #

        self.random_evaluations = 25
        self.max_evaluations = 100

        self.n_processes = cpu_count() * 4

        self._create_search_space = {
            "RLForward": self._create_search_space_for_RLForward,
            "RLForward2": self._create_search_space_for_RLForward2,
            "RLStrategic": self._create_search_space_for_RLStrategic,
        }

        self.models = self._create_search_space.keys()

    # ----- Let's say 'generic' functions -------- #

    def run(self, model):

        print("Optimizing with {}...".format(model))
        print()

        # Create a progression bar
        p_bar = tqdm(total=len(self.subjects_idx))

        # Do a list of arguments for threads that will be used for computation
        compute_args = []

        for i in self.subjects_idx:
            compute_args.append(
                {
                    "i": i,
                    "model": model,
                    "p_bar": p_bar
                }

            )

        # Optimize for selected individuals using multi threading
        pool = ThreadPool(processes=self.n_processes)
        backup = pool.map(self._compute, compute_args)

        # Close progression bar
        p_bar.close()

        print()
        print("Optimization done!")
        print()

        return backup

    def run_for_a_single_agent(self, model, idx):

        # Create search space depending of the model
        search_space, parameters = self._create_search_space[model]()

        results = self._compute(args={
            "i": idx,
            "model": model
        })

        return results

    def _compute(self, args):

        # Create search space depending of the model
        search_space, parameters = self._create_search_space[args["model"]](args["i"])

        best = self._optimize(ind=args["i"], model=args["model"], search_space=search_space)

        squares_sum, bic_value = \
            self._evaluate_performance(ind=args["i"], model=args["model"], args=[best[i] for i in parameters])

        # Put results in a dictionary
        results = dict()
        results["squares_sum"] = squares_sum
        results["bic"] = bic_value
        results["best"] = best

        # Update the progression bar
        args["p_bar"].update()

        return results

    def _optimize(self, ind, model, search_space):

        pc = PerformanceComputer(
            individual_data=self.data[ind],
            model=model)

        alg = op.partial(
            op.tpe.suggest,  # bayesian optimization # tpe.rand, #random optimization
            n_startup_jobs=self.random_evaluations)

        best = op.fmin(
            fn=pc.run,
            space=search_space,
            algo=alg,
            max_evals=self.max_evaluations)

        return best

    def _evaluate_performance(self, ind, model, args):

        pc = PerformanceComputer(
            individual_data=self.data[ind],
            model=model)
        return pc.evaluate(*args)

    # --------------------- SEARCH SPACE ------------------------------ #

    def _create_search_space_for_RLForward(self, ind):

        parameters = [
            "alpha",
            "temp",
            "gamma"
        ]

        # For each object the agent could have in hand
        for i in [self.data[ind]["subject_good"][0], (self.data[ind]["subject_good"][0] - 1) % 3]:
            # for every type of agent he could be matched with
            for j in range(3):
                # For each object this 'partner' could have in hand (production, third good)
                for k in [(j+1) % 3, (j-1) % 3]:
                    # Key is composed by good in hand, partner type, good in partner's hand
                    for e in [0, 1]:
                        parameters.append("q{}{}{}{}".format(i, j, k, e))

        search_space = []

        for i in parameters:
            if i != 'temp':
                minimum, maximum = 0., 1.
            else:
                minimum, maximum = 0.01, 1.

            search_space.append(
                op.hp.uniform(i, minimum, maximum)
            )

        return search_space, parameters

    def _create_search_space_for_RLForward2(self, ind):

        parameters = [
            "alpha",
            "temp",
            "gamma"
        ]

        # For each object the agent could have in hand
        for i in [self.data[ind]["subject_good"][0], (self.data[ind]["subject_good"][0] - 1) % 3]:
            # for proposed object
            for j in range(3):
                # Key is composed by good in hand, partner type, good in partner's hand
                for e in [0, 1]:
                    parameters.append("q{}{}{}".format(i, j, e))

        search_space = []

        for i in parameters:
            if i != 'temp':
                minimum, maximum = 0., 1.
            else:
                minimum, maximum = 0.01, 1.

            search_space.append(
                op.hp.uniform(i, minimum, maximum)
            )

        return search_space, parameters

    def _create_search_space_for_RLStrategic(self, ind):

        parameters = [
            "alpha",
            "temp"
        ]

        n_strategies = 4

        for i in range(n_strategies):
            parameters.append(
                "s_{:02d}".format(i)
            )

        search_space = []

        for i in parameters:
            if i != 'temp':
                minimum, maximum = 0., 1.
            else:
                minimum, maximum = 0.01, 1.

            search_space.append(
                op.hp.uniform(i, minimum, maximum)
            )

        return search_space, parameters


class PerformanceComputerWithoutParameters(object):

    def __init__(self, data, subjects_idx=None):

        self.data = data
        if subjects_idx is not None:
            self.subjects_idx = subjects_idx
        else:
            self.subjects_idx = np.arange(len(self.data))

    def run(self, model):

        print("Evaluating performance of {}...".format(model))

        backup = []

        for i in self.subjects_idx:
            p = PerformanceComputer(individual_data=self.data[i], model="NonParametrizedAgent")
            squares_sum, bic_value = p.evaluate((model, ))

            # Put results in a dictionary
            results = dict()
            results["squares_sum"] = squares_sum
            results["bic"] = bic_value

            backup.append(results)

        print('Done!')
        print()

        return backup


class ModelComparison(object):

    def __init__(self, data, model_to_test, subjects_idx=None):

        self.data = data
        self.model_to_test = model_to_test
        if subjects_idx is not None:
            self.subjects_idx = subjects_idx
        else:
            self.subjects_idx = np.arange(len(self.data))

    def run(self):

        general_parameters = {
            "subjects_idx": self.subjects_idx,
            "data": self.data
        }

        optimizer = Optimizer(**general_parameters)
        without_parameters_eval = PerformanceComputerWithoutParameters(**general_parameters)

        results = dict()

        for model in self.model_to_test:

            if model in optimizer.models:
                results[model] = optimizer.run(model)
            else:
                results[model] = without_parameters_eval.run(model)

        self.print_results(results=results)
        self.save(results=results)

    def print_results(self, results):

        for model in self.model_to_test:

            for var in ["squares_sum", "bic"]:

                data = [results[model][i][var] for i in range(len(self.subjects_idx))]

                msg = "{} - {}: {:.2f} +/- {:.2f} [{:.2f}; {:.2f}]".format(
                    model, var,
                    np.mean(data),
                    np.std(data),
                    min(data), max(data)
                )
                print(msg)
            print()

    def save(self, results):

        np.save("../optimization.npy", results)

        with open('../optimization_individual.csv', 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([
                "idx", "model", "squares_sum", "BIC"
            ])

            for i, idx in enumerate(self.subjects_idx):
                for model in self.model_to_test:

                    to_write = [
                        idx,
                        model,
                        results[model][i]["squares_sum"],
                        results[model][i]["bic"]
                    ]
                    if "best" in results[model][i]:
                        for parameter, value in sorted(results[model][i]["best"].items()):
                            to_write += [parameter, value]
                    writer.writerow(to_write)

        with open('../optimization_stats.csv', 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([
                "model",
                "squares_sum_mean", "squares_sum_std", "squares_sum_min", "squares_sum_max",
                "BIC_mean", "BIC_std", "BIC_min", "BIC_max"
            ])
            for model in self.model_to_test:

                to_analyse = ["squares_sum", "bic"]

                to_write = [model]
                for var in to_analyse:

                    data = [results[model][i][var] for i in range(len(self.subjects_idx))]

                    to_write += [
                        np.mean(data), np.std(data),
                        min(data), max(data)
                    ]
                writer.writerow(to_write)


def comparison_multi_models(
        data, model_to_test=("RLForward2", "KW", "Duffy", "TotalGogol", "StupidAgent", "RLStrategic", "RLForward")):

    m = ModelComparison(data=data, model_to_test=model_to_test)
    m.run()


def test_single_non_parametric_model(model, data):

    without_parameters_eval = PerformanceComputerWithoutParameters(data=data)
    results = without_parameters_eval.run(model=model)
    print(results)


def test_single_agent_with_non_parametric_model(model, data, idx):

    p = PerformanceComputer(individual_data=data[idx], model="NonParametrizedAgent")
    squares_sum, bic_value = p.evaluate((model,))

    print("squares_sum", squares_sum)
    print("bic", bic_value)


def test_single_agent_with_parametric_model(model, data, idx):

    optimizer = Optimizer(data=data)
    results = optimizer.run_for_a_single_agent(model=model, idx=idx)
    print(results)


def main():

    data = import_data()
    comparison_multi_models(data=data)
    # test_single_agent_with_non_parametric_model(model="Duffy", data=data, idx=0)


if __name__ == "__main__":
    main()
