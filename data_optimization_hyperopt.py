import numpy as np
import hyperopt as op
import csv
from tqdm import tqdm
from data_manager import import_data
from RLForward import RLForwardAgent
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
            "RLStrategic": self.get_SRL_model,
            "NonParametrizedAgent": self.get_non_parametrized_model,
        }

        self.non_parametrized_model = {
            "TotalGogol": TotalGogol, "StupidAgent": StupidAgent,
            "Duffy": DuffyAgent, "KW": KwAgent
        }

    def run(self, *args):

        model = self.func[self.model](*args)
        neg_ll_sum = self.compute_neg_sum_log_likelihood(model)
        return neg_ll_sum

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

    def compute_neg_sum_log_likelihood(self, model):

        log_likelihood_list = []

        for t in range(self.t_max):

            model.match_departure_good(subject_good=self.data["subject_good"][t])

            likelihood = model.probability_of_responding(
                subject_response=self.data["subject_choice"][t],
                partner_good=self.data["partner_good"][t],
                partner_type=self.data["partner_type"][t],
                proportions=self.data["prop"][t])

            if likelihood > 0:
                perf = np.log(likelihood)
            else:
                perf = np.log(0.001)  # To avoid log(0). We could have a best idea. Maybe.
                # We could interpret this as the probability of making a stupid error
            log_likelihood_list.append(perf)

            model.do_the_encounter(
                partner_choice=self.data["partner_choice"][t],
                partner_type=self.data["partner_type"][t],
                partner_good=self.data["partner_good"][t],
                subject_choice=self.data["subject_choice"][t])

        result = - sum(log_likelihood_list)
        return result

    def evaluate(self, *args):

        neg_ll_sum = self.run(args)
        ll_sum = neg_ll_sum * (- 1)

        if "NonParametrizedAgent" in args:
            degrees_of_freedom = 0
        else:
            degrees_of_freedom = len(args)

        return ll_sum, self.bic_formula(max_log_likelihood=ll_sum,
                                        n_trials=self.t_max, degrees_of_freedom=degrees_of_freedom)

    @staticmethod
    def bic_formula(max_log_likelihood, n_trials, degrees_of_freedom):

        return - 2 * max_log_likelihood + np.log(n_trials) * degrees_of_freedom


class Optimizer(object):

    def __init__(self, subjects_idx, data):

        self.subjects_idx = subjects_idx
        self.data = data
        
        self.random_evaluations = 250
        self.max_evaluations = 100

        self.optimize_model = {
            "RLForward": self.run_forward_RL,
            "RLStrategic": self.run_strategic_RL,
        }

        self.optimize_for_a_single_subject = {
            "RLForward": self.optimize_single_individual_with_RLForward,
            "RLStrategic": self.optimize_single_individual_with_RLStrategic,
        }

    # ----- Let's say 'generic' functions -------- #
        
    def run(self, model):
        
        return self.optimize_model[model]()

    def optimize(self, ind, model, search_space):

        pc = PerformanceComputer(
            individual_data=self.data[ind],
            model=model)

        alg = op.partial(op.tpe.suggest,  # bayesian optimization
                          # tpe.rand, #random optimization
                          n_startup_jobs=self.random_evaluations)
        best = op.fmin(
            fn=pc.run,
            space=search_space,
            algo=alg,
            max_evals=self.max_evaluations)

        return best

    def optimize_for_n_subjects(self, model, parameters):

        backup = dict()
        backup["max_log_likelihood"] = []
        backup["bic"] = []
        backup["best"] = []

        print("Optimizing with {}...".format(model))
        print()

        # --- Optimize for selected individuals

        for i in tqdm(self.subjects_idx):

            best = self.optimize_for_a_single_subject[model](i)

            max_log_likelihood, bic_value = \
                self.evaluate_performance(ind=i, model=model, args=[best[i] for i in parameters])

            # Put results in backup dic

            backup["max_log_likelihood"].append(max_log_likelihood)
            backup["bic"].append(bic_value)
            backup["best"].append(best)

        print()
        print("Optimization done!")
        print()

        return backup

    def evaluate_performance(self, ind, model, args):

        pc = PerformanceComputer(
            individual_data=self.data[ind],
            model=model)
        return pc.evaluate(*args)

    # --------------------- RLForward ------------------------------ #

    def optimize_single_individual_with_RLForward(self, ind):

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

    def run_forward_RL(self):

        model = "RLForward"
        
        # --- Prepare backup

        parameters = ["alpha", "temp", "gamma"]

        n_situations = 12
        for i in range(n_situations):
            for j in range(2):

                parameters.append(
                    "q_{:02d}_{}".format(i, j)
                )

        return self.optimize_for_n_subjects(model=model, parameters=parameters)

    # --------------------- RLStrategic ------------------------------ #

    def optimize_single_individual_with_RLStrategic(self, ind):

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

    def run_strategic_RL(self):

        model = "RLStrategic"

        # --- Prepare backup

        parameters = ["alpha", "temp"]

        n_strategies = 4

        for i in range(n_strategies):
            parameters.append(
                "s_{:02d}".format(i)
            )

        return self.optimize_for_n_subjects(model=model, parameters=parameters)


class PerformanceComputerWithoutParameters(object):

    def __init__(self, data, subjects_idx):

        self.data = data
        self.subjects_idx = subjects_idx

    def run(self, model):

        print("Evaluating performance of {}...".format(model))

        sum_ll_list = []
        bic_list = []

        for i in self.subjects_idx:
            p = PerformanceComputer(individual_data=self.data[i], model="NonParametrizedAgent")
            neg_sum_ll, bic = p.evaluate((model, ))

            sum_ll_list.append(neg_sum_ll * -1)
            bic_list.append(bic)

        results = {"max_log_likelihood": sum_ll_list, "bic": bic_list}

        print('Done!')
        print()

        return results


class ModelComparison(object):

    def __init__(self):

        self.subjects_idx = np.arange(30)

        self.data = import_data()

        self.model_to_test = ["KW", "Duffy", "TotalGogol", "StupidAgent", "RLStrategic", "RLForward"]

    def run(self):

        general_parameters = {
            "subjects_idx": self.subjects_idx,
            "data": self.data
        }

        optimizer = Optimizer(**general_parameters)
        without_parameters_eval = PerformanceComputerWithoutParameters(**general_parameters)

        results = dict()

        for model in self.model_to_test:

            if model in optimizer.optimize_model.keys():
                results[model] = optimizer.run(model)
            else:
                results[model] = without_parameters_eval.run(model)

        for model in sorted(results.keys()):

            to_analyse = ["max_log_likelihood", "bic"]

            for var in to_analyse:

                msg = "{} - {}: {:.2f} +/- {:.2f} [{:.2f}; {:.2f}]".format(
                    model, var, np.mean(results[model][var]), np.std(results[model][var]),
                    min(results[model][var]), max(results[model][var])
                )
                print(msg)

        self.save(results=results)

    def save(self, results):

        with open('../optimization_individual.csv', 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([
                "idx", "model", "LLmax", "BIC"
            ])

            for i, idx in enumerate(self.subjects_idx):
                for model in self.model_to_test:

                    to_write = [
                        idx,
                        model,
                        results[model]["max_log_likelihood"][i],
                        results[model]["bic"][i]
                    ]
                    if "best" in results[model]:
                        for parameter, value in sorted(results[model]["best"][i].items()):
                            to_write += [parameter, value]
                    writer.writerow(to_write)

        with open('../optimization_stats.csv', 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([
                "model",
                "LLmax_mean", "LLmax_std", "LLmax_min", "LLmax_max",
                "BIC_mean", "BIC_std", "BIC_min", "BIC_max"
            ])
            for model in self.model_to_test:

                to_analyse = ["max_log_likelihood", "bic"]

                to_write = [model]
                for var in to_analyse:
                    to_write += [

                        np.mean(results[model][var]), np.std(results[model][var]),
                        min(results[model][var]), max(results[model][var])
                    ]
                writer.writerow(to_write)


def main():
    
    m = ModelComparison()
    m.run()

if __name__ == "__main__":
    main()
