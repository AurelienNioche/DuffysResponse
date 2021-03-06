import csv
import itertools as it
import json
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import hyperopt as op
import numpy as np
from tqdm import tqdm

from agent.DuffyAgent import DuffyAgent
from agent.ForwardRL import ForwardRLAgent
from agent.FrequentistAgent import FrequentistAgent
from agent.KwAgent import KwAgent
from agent.RL2Steps import RL2StepsAgent
from agent.StrategicRL import StrategicRLAgent
from agent.stupid_agent import StupidAgent
from agent.stupidy_is_better import TotalGogol
from data_analysis.data_manager import import_data


class PerformanceComputer(object):

    def __init__(self, individual_data, model):

        self.data = individual_data

        self.raw_storing_costs = self.data["storing_costs"]
        self.raw_u = self.data["u"]
        self.beta = self.data["beta"]

        self.prod = self.data["subject_good"][0]
        self.cons = (self.data["subject_good"][0] - 1) % 3

        self.t_max = len(self.data["subject_good"])

        self.model = model

        self.func = {
            "ForwardRL": self.get_FRL_model,
            "RL2Steps": self.get_RL2Steps_model,
            "StrategicRL": self.get_SRL_model,
            "Frequentist": self.get_frequentist_model,
            "NonParametrized": self.get_non_parametrized_model,
        }

        self.non_parametrized_model = {
            "TotalGogol": TotalGogol, "StupidAgent": StupidAgent,
            "Duffy": DuffyAgent, "KW": KwAgent
        }

    def run(self, *args):

        model = self.func[self.model](*args)
        squares_sum = self.compute_sum_errors_squares(model)
        return squares_sum
    
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

        if "NonParametrized" in args:
            degrees_of_freedom = 0
        else:
            degrees_of_freedom = len(args)

        return squares_sum, self.bic_formula(
            squares_sum=squares_sum,
            n_trials=self.t_max, degrees_of_freedom=degrees_of_freedom)

    @staticmethod
    def bic_formula(squares_sum, n_trials, degrees_of_freedom):

        return n_trials * np.log(squares_sum / n_trials) + np.log(n_trials) * degrees_of_freedom

    # ---- GET MODELS --------------- #

    def get_FRL_model(self, *args):

        alpha, temp, gamma = args[0][:3]
        q_values = np.asarray(args[0][3:]).reshape((12, 2))

        model = ForwardRLAgent(
            prod=self.prod,
            cons=self.cons,  # Suppose we are in the KW's Model A
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

    def get_RL2Steps_model(self, *args):

        alpha, temp, gamma = args[0][:3]
        q_values = np.asarray(args[0][3:]).reshape((6, 2))

        model = RL2StepsAgent(
            prod=self.prod,
            cons=self.cons,  # Suppose we are in the KW's Model A
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

        alpha, temp = args[0][:2]
        strategy_values = np.asarray(args[0][2:])

        model = StrategicRLAgent(
            prod=self.prod,
            cons=self.cons,  # Suppose we are in the KW's Model A
            storing_costs=self.raw_storing_costs,
            u=self.raw_u,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "strategy_values": strategy_values,
            }
        )

        return model

    def get_frequentist_model(self, *args):

        encounter_memory_span, acceptance_memory_span, temp = args[0][:3]
        n_exchanges = len(args[0][3:]) // 2
        encounter_probabilities = np.asarray(args[0][3:n_exchanges + 3])
        acceptance_probabilities = np.asarray(args[0][n_exchanges + 3:])
        
        model = FrequentistAgent(
            prod=self.prod,
            cons=self.cons,
            storing_costs=self.raw_storing_costs,
            u=self.raw_u,
            agent_parameters={
                "encounter_memory_span": encounter_memory_span,
                "acceptance_memory_span": acceptance_memory_span,
                "temp": temp,
                "encounter_probabilities": encounter_probabilities,
                "acceptance_probabilities": acceptance_probabilities
            }
        )
        
        return model
        
    def get_non_parametrized_model(self, args):

        model_name = args[0][0]

        model = self.non_parametrized_model[model_name](
                prod=self.prod,
                cons=self.cons,  # Suppose we are in the KW's Model A
                storing_costs=self.raw_storing_costs,
                u=self.raw_u,
                beta=self.beta
            )

        return model


class Optimizer(object):

    def __init__(self, data, subjects_idx=None):

        self.data = data

        if subjects_idx is not None:
            self.subjects_idx = subjects_idx
        else:
            self.subjects_idx = np.arange(len(self.data))

        # ------ Optimization parameters ------- #
        with open("parameters/optimization_parameters.json") as file:
            param = json.load(file)
        self.random_evaluations = param["random_evaluations"]
        self.max_evaluations = param["max_evaluations"]

        self.n_processes = cpu_count()

        self._create_search_space = {
            "ForwardRL": self._create_search_space_for_ForwardRL,
            "RL2Steps": self._create_search_space_for_RL2Steps,
            "StrategicRL": self._create_search_space_for_StrategicRL,
            "Frequentist": self._create_search_space_for_Frequentist,
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

        if "p_bar" in args:
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

    def _create_search_space_for_ForwardRL(self, ind):

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

    def _create_search_space_for_RL2Steps(self, ind):

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
                    parameters.append("({}, {}, {})".format(i, j, e))

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

    def _create_search_space_for_StrategicRL(self, ind):

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

    def _create_search_space_for_Frequentist(self, ind):

        parameters = [
            "encounter_memory_span",
            "acceptance_memory_span",
            "temp"
        ]

        n_goods = 3

        for i in it.permutations(range(n_goods), r=2):
            parameters.append(
                "encounter{}".format(i)
            )

        for i in it.permutations(range(n_goods), r=2):
            parameters.append(
                "acceptance{}".format(i)
            )

        t_max = len(self.data[ind]["subject_good"])

        search_space = []

        for i in parameters:

            if i == 'temp':

                minimum, maximum = 0.01, 1.
                search_space.append(
                    op.hp.uniform(i, minimum, maximum)
                )

            elif i in ["encounter_memory_span", "acceptance_memory_span"]:

                minimum, maximum = 1, t_max
                search_space.append(
                    op.hp.quniform(i, minimum, maximum, 1)
                )

            else:

                minimum, maximum = 0., 1.
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
            p = PerformanceComputer(individual_data=self.data[i], model="NonParametrized")
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

        summary = self.print_results(results=results)
        self.save(results=results, summary=summary)

    def print_results(self, results):

        msg = ""

        for model in self.model_to_test:

            for var in ["squares_sum", "bic"]:

                data = [results[model][i][var] for i in range(len(self.subjects_idx))]

                txt = "{} - {}: {:.2f} +/- {:.2f} [{:.2f}; {:.2f}]".format(
                    model, var,
                    np.mean(data),
                    np.std(data),
                    min(data), max(data)
                )
                msg += txt + "\n"
            msg += "\n"

        return msg

    def save(self, results, summary=None):

        np.save("../optimization.npy", results)

        with open("../summary.txt", "w") as file:
            file.write(summary)

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
        data, model_to_test=(
                "RL2Steps", "Frequentist", "KW", "Duffy", "TotalGogol", "StupidAgent", "StrategicRL", "ForwardRL",
        )):

    m = ModelComparison(data=data, model_to_test=model_to_test)
    m.run()


def test_single_non_parametric_model(model, data):

    without_parameters_eval = PerformanceComputerWithoutParameters(data=data)
    results = without_parameters_eval.run(model=model)
    print(results)


def test_single_agent_with_non_parametric_model(model, data, idx):

    p = PerformanceComputer(individual_data=data[idx], model="NonParametrized")
    squares_sum, bic_value = p.evaluate((model,))

    print("squares_sum", squares_sum)
    print("bic", bic_value)


def test_single_agent_with_parametric_model(model, data, idx):

    optimizer = Optimizer(data=data)
    results = optimizer.run_for_a_single_agent(model=model, idx=idx)
    print(results)


def main():

    data = import_data()
    # comparison_multi_models(data=data)
    # test_single_agent_with_non_parametric_model(model="Duffy", data=data, idx=0)
    # test_single_agent_with_parametric_model(model="Frequentist", data=data, idx=0)
    comparison_multi_models(data=data,
                            model_to_test=(
                                # "RL2Steps", "Frequentist", "KW", "Duffy", "TotalGogol",
                                # "StupidAgent", "StrategicRL", "ForwardRL"
                                "RL2Steps", "Duffy"
    ))


if __name__ == "__main__":
    main()
