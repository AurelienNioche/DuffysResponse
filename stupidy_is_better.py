import numpy as np
from data_manager import import_data


class Agent(object):

    """
    Abstract class for agents
    """
    name = "Stupid agent"

    def __init__(self, prod, cons):

        # Production object (integer in [0, 1, 2])
        self.P = prod

        # Consumption object (integer in [0, 1, 2])
        self.C = cons

        # Object an agent has in hand
        self.in_hand = self.P

    def probability_of_responding(self, subject_response, partner_good):

        if partner_good == self.C:
            if subject_response:
                return 1
            else:
                return 0

        else:
            return 0.5

    def do_the_encounter(self, partner_choice, partner_good, subject_choice):

        if subject_choice and partner_choice:
            self.in_hand = partner_good
            if self.in_hand == self.C:
                self.in_hand = self.P


def main():

    data = import_data()

    max_sum_ll = []

    for i in range(30):

        t_max = len(data[i]["subject_good"])

        a = Agent(
            prod=data[i]["subject_good"][0],
            cons=(data[i]["subject_good"][0] - 1) % 3,  # Suppose we are in the KW's Model A
        )

        log_likelihood_list = []

        for t in range(t_max):

            likelihood = a.probability_of_responding(
                subject_response=data[i]["subject_choice"][t],
                partner_good=data[i]["partner_good"][t]
            )

            if likelihood > 0:
                perf = np.log(likelihood)
            else:
                perf = np.log(0.001)  # To avoid log(0). We could have a best idea. Maybe.
                # We could interpret this as the probability of making a stupid error
            log_likelihood_list.append(perf)

            a.do_the_encounter(partner_choice=data[i]["partner_choice"][t],
                               partner_good=data[i]["partner_good"][t],
                               subject_choice=data[i]["subject_choice"][t])

        max_sum_ll.append(sum(log_likelihood_list))

    print(np.mean(max_sum_ll))
    print(np.mean(- 2 * np.asarray(max_sum_ll)))


if __name__ == "__main__":

    main()