from os import path
import numpy as np
import seaborn as sns
from scipy.stats.stats import pearsonr

from data_analysis.data_manager import import_data


def compute_speculation_ratio(data, verbose=False):

    ratio_speculate_list = []

    for i in data:

        t_max = len(i["subject_good"])
        prod_good = i["subject_good"][0]
        cons_good = (i["subject_good"][0] - 1) % 3
        third_good = (i["subject_good"][0] - 2) % 3

        storing_costs = i["storing_costs"]

        speculate = 0
        speculative_proposition = 0
        for t in range(t_max):

            if storing_costs[i["subject_good"][t]] < storing_costs[i["partner_good"][t]] \
                    and i["partner_good"][t] == third_good:

                speculative_proposition += 1
                if i["subject_choice"][t] == 1:
                    speculate += 1

        ratio_speculate = speculate / speculative_proposition
        ratio_speculate_list.append(ratio_speculate)

        if verbose:
            print('Storing costs', storing_costs)
            print("prod: {}, cons: {}, third: {}".format(prod_good, cons_good, third_good))
            print("t_max", t_max)
            print("speculate", speculate)
            print("ratio speculate", ratio_speculate)
            print()

    return np.asarray(ratio_speculate_list)


def compute_consumption_ratio(data):

    ratio_list = []

    for i in data:

        t_max = len(i["subject_good"])
        cons_good = (i["subject_good"][0] - 1) % 3

        consumption_proposition = 0
        consumption = 0
        for t in range(t_max):

            if i["partner_good"][t] == cons_good:

                consumption_proposition += 1
                if i["subject_choice"][t] == 1:
                    consumption += 1
        ratio = consumption / consumption_proposition
        ratio_list.append(ratio)

    return np.asarray(ratio_list)


def compute_pure_consumption_ratio(data):

    ratio_list = []

    for i in data:

        t_max = len(i["subject_good"])
        cons_good = (i["subject_good"][0] - 1) % 3

        consumption = 0
        for t in range(t_max):

            if i["partner_good"][t] == cons_good and i["subject_choice"][t] == 1:
                    consumption += 1
        ratio = consumption / t_max
        ratio_list.append(ratio)

    return np.asarray(ratio_list)


def do_some_stats(array_like, label):

    # Plot distribution
    plot = sns.distplot(array_like, kde=False)
    fig = plot.get_figure()
    fig.savefig(path.expanduser("~/Desktop/{}.pdf".format(label)))
    fig.clear()

    # Do some print
    print("{}".format(label))
    ind_results = "Ind results: "
    for i in array_like:
        ind_results += "{:.2f}; ".format(i)
    ind_results = ind_results[:-2] + "."
    print(ind_results)
    print("Min: {:.2f}, max: {:.2f}, mean: {:.2f}, std: {:.2f}".format(
        min(array_like), max(array_like), np.mean(array_like), np.std(array_like)
    )
    )
    print()


def do_some_correlation_analysis(array_like_0, array_like_1, label):

    cor, p = pearsonr(array_like_0, array_like_1)
    print("{}: {:.2f} [p={:.3f}]".format(label, cor, p))


def main():

    data = import_data()
    sp_ratio_list = compute_speculation_ratio(data=data)
    cons_ratio_list = compute_consumption_ratio(data=data)
    pure_cons_list = compute_pure_consumption_ratio(data=data)
    do_some_stats(sp_ratio_list, "Speculation")
    do_some_stats(cons_ratio_list, "Consumption")
    do_some_stats(pure_cons_list, "PureConsumption")

    do_some_correlation_analysis(sp_ratio_list, cons_ratio_list,
                                 label="Correlation between accepting to speculate and accepting to consume")

    do_some_correlation_analysis(sp_ratio_list, pure_cons_list,
                                 label="Correlation between accepting to speculate and average consumption")

    do_some_correlation_analysis(cons_ratio_list, pure_cons_list,
                                 label="Correlation between accepting to consume and average consumption")

if __name__ == "__main__":

    main()
