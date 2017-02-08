import numpy as np
from scipy import optimize as op
from data_manager import import_data
from RLForward import RLForwardAgent

class PerformanceComputer(object):

    def __init__(self, subject_idx, storing_costs):

        self.data = import_data()[subject_idx]  # This is for subject one

        self.storing_costs = storing_costs
        
        self.t_max = len(self.data["subject_good"])

    def run(self, *args):

        alpha, temp, gamma = args[0][:3]

        print()
        print(alpha, temp, gamma)

        q_values = args[0][3:].reshape((12, 2))

        print(q_values)

        m = RLForwardAgent(
            prod=self.data["subject_good"][0],
            cons=(self.data["subject_good"][0] - 1) % 3,  # Suppose we are in the KW's Model A
            third=(self.data["subject_good"][0] - 2) % 3,
            agent_type=(self.data["subject_good"][0] - 1) % 3,
            storing_costs=self.storing_costs,
            agent_parameters={
                "alpha": alpha,
                "temp": temp,
                "gamma": gamma,
                "q_values": q_values,
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
        print(result)
        return result


def main():

    storing_costs = [1, 4, 9]

    pc = PerformanceComputer(subject_idx=0, storing_costs=storing_costs)

    first_alpha = 0.5
    alpha_limits = (0., 1.)

    first_temp = 0.5
    temp_limits = (0., np.inf)

    first_gamma = 0.5
    gamma_limits = (0., 1.)

    q_values = np.random.random((12, 2))
    q_values_limits = [(-np.inf, np.inf), ] * q_values.size

    x0 = np.array(
        [first_alpha, first_temp, first_gamma]
        + list(np.ravel(q_values))
    )

    bounds = np.array(
        [alpha_limits, temp_limits, gamma_limits] + q_values_limits
    )

    print(op.minimize(fun=pc.run, x0=x0, method="L-BFGS-B", bounds=bounds))

    # bounds = np.array([
    #     [alpha_limits, temp_limits] + q_values_limits
    # ])))


if __name__ == "__main__":

    main()
