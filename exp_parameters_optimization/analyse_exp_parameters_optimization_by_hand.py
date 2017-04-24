from os import path
import pickle
import numpy as np


class Analyst(object):

    file_name = \
        {
            "all": path.expanduser(
                "~/Desktop/exp_parameters_optimization_by_hand_data.p"),
            "speculative": path.expanduser(
                "~/Desktop/exp_parameters_optimization_by_hand_data_speculative.p"),
            "zero_as_medium": path.expanduser(
                "~/Desktop/exp_parameters_optimization_zero_as_medium.p"),
            "one_as_medium": path.expanduser(
                "~/Desktop/exp_parameters_optimization_one_as_medium.p"),
            "two_as_medium": path.expanduser(
                "~/Desktop/exp_parameters_optimization_two_as_medium.p"),
            "storing_costs": path.expanduser(
                "~/Desktop/exp_parameters_optimization_storing_costs.p"),
            "diff_storing_costs": path.expanduser(
                "~/Desktop/exp_parameters_optimization_diff_storing_costs.p")
        }

    labels = {
        "zero_as_medium": 0,
        "one_as_medium": 1,
        "two_as_medium": 2,
        "diff_storing_costs": 3
    }

    def __init__(self):

        self.data = self.load()

        self.run()

    def load(self):

        data = {}

        for key, value in self.file_name.items():

            if path.exists(value):
                print("'{}' exits, I will load data from it.".format(value))
                with open(value, 'rb') as f:
                    data[key] = pickle.load(f)
            else:
                print("'{}' doesn't exits.".format(value))
                data[key] = None

        return data

    def run(self):

        if self.data["speculative"] is None:
            self.sort_speculative()

        if self.data["two_as_medium"] is None or \
                self.data["storing_costs"] is None or \
                self.data["diff_storing_costs"] is None or \
                self.data["zero_as_medium"] is None or \
                self.data["one_as_medium"] is None:
            self.extract_what_is_relevant()

        self.analyse()

    def sort_speculative(self):

        self.data["speculative"] = dict()

        for key, value in self.data["all"].items():

            if type(value) == str and value == "non-speculative":
                pass
            else:
                self.data["speculative"][key] = value

        with open(self.file_name["speculative"], "wb") as f:
            pickle.dump(self.data["speculative"], f)

        print("N all: {}.".format(len(self.data["all"])))
        print('N speculative: {}.'.format(len(self.data["speculative"])))

    def extract_what_is_relevant(self):

        n = len(self.data["speculative"])
        zero_as_medium = np.zeros(n)
        one_as_medium = np.zeros(n)
        two_as_medium = np.zeros(n)

        diff_storing_costs = np.zeros(n)
        storing_costs = np.zeros(n, dtype=tuple)

        for i, (key, value) in enumerate(sorted(self.data["speculative"].items())):

            storing_costs[i] = key
            diff_storing_costs[i] = value[self.labels["diff_storing_costs"]]
            zero_as_medium[i] = value[self.labels["zero_as_medium"]]
            one_as_medium[i] = value[self.labels["one_as_medium"]]
            two_as_medium[i] = value[self.labels["two_as_medium"]]

        with open(self.file_name["zero_as_medium"], "wb") as f:
            pickle.dump(zero_as_medium, f)

        with open(self.file_name["one_as_medium"], "wb") as f:
            pickle.dump(one_as_medium, f)

        with open(self.file_name["two_as_medium"], "wb") as f:
            pickle.dump(two_as_medium, f)

        with open(self.file_name["diff_storing_costs"], "wb") as f:
            pickle.dump(diff_storing_costs, f)

        with open(self.file_name["storing_costs"], "wb") as f:
            pickle.dump(storing_costs, f)

        self.data["zero_as_medium"] = zero_as_medium
        self.data["one_as_medium"] = one_as_medium
        self.data["two_as_medium"] = two_as_medium
        self.data["storing_costs"] = storing_costs
        self.data["diff_storing_costs"] = diff_storing_costs

    def analyse(self):

        print()
        res = self.function_to_minimize()
        # print(min(res), max(res))

        sorted_indices = np.argsort(res)

        print("Storing costs from the best to the worst for having speculative equilibrium "
              "[corresponding distance between the storing cost of the good 2 and 1]:")

        for i, j in zip(self.data["storing_costs"][sorted_indices], self.data["diff_storing_costs"][sorted_indices]):
            print("{} [{}]".format(i, j))

    def function_to_minimize(self):

        return (1 - self.data["zero_as_medium"]) + self.data["one_as_medium"] + (1 - self.data["two_as_medium"])


def main():

    a = Analyst()

if __name__ == "__main__":

    main()