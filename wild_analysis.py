import numpy as np
from scipy.stats import f_oneway
import pandas as pd
from statsmodels.formula.api import ols
from scipy.stats.stats import pearsonr
from data_manager import import_data
from sort_subjects import *


class Analyst(object):

    def __init__(self):

        self.data = import_data()
        self.op_results = np.load("../optimization.npy")[()]

        self.models = [i for i in sorted(self.op_results.keys())]

        self.n = len(self.op_results[self.models[0]])

        self.speculation_ratio = compute_speculation_ratio(data=self.data)

    def compute_correlation_between_gamma_and_speculation(self):

        gamma_list = self.get_best_parameter_for_every_agent(model="RLForward", parameter="gamma")
        r, p_value = self.compute_correlation(
            gamma_list,
            self.speculation_ratio
        )
        print("Correlation between gamma and speculation: {:.2f} [p={:.3f}]".format(r, p_value))

    def get_best_parameter_for_every_agent(self, model, parameter):

        return np.asarray([self.op_results[model][i]["best"][parameter] for i in range(self.n)])

    def get_best_model_for_each_agent(self):

        best_models = []

        for i in range(self.n):
            model_error = [self.op_results[model][i]["squares_sum"] for model in self.models]
            best_model = self.models[np.argmin(model_error)]
            best_models.append(best_model)

        return np.asarray(best_models)

    def distribution_of_best_models(self):

        best_models = self.get_best_model_for_each_agent()

        r = {}
        for m in self.models:
            r[m] = 0

        for bm in best_models:
            r[bm] += 1

        return r

    @staticmethod
    def compute_correlation(x, y):

        return pearsonr(x, y)

    @staticmethod
    def compute_regression(x, y):

        d = pd.DataFrame({
            "x": x,
            "y": y
        })

        model = ols("y ~ x", d).fit()
        return model.summary()


def main():

    a = Analyst()
    # a.compute_correlation_between_gamma_and_speculation()
    print(a.distribution_of_best_models())
if __name__ == "__main__":

    main()
