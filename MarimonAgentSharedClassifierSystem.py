import numpy as np
from itertools import product
from graph import represent_results
from Economy import Economy
from stupid_agent import StupidAgent

# --------------------------------------------------------------------------------------------------- #
# -------------------------------- MARIMON AGENT ---------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #


class MarimonAgent(StupidAgent):

    name = "Marimon"

    def __init__(self, prod, cons, exchange_classifier_system,
                 consumption_classifier_system, storing_costs, u, idx):
        
        super().__init__(prod=prod, cons=cons, storing_costs=storing_costs, u=u, idx=idx)

        # Parameters for agent that could be different in nature depending on the agent model in use (Python dictionary)
        self.exchange_classifier_system = exchange_classifier_system
        self.consumption_classifier_system = consumption_classifier_system

        self.previous_object_H = self.P

        self.best_exchange_classifier = None
        self.best_consumption_classifier = None

        self.bid_from_exchange_classifier = None
        self.bid_from_consumption_classifier = None

        self.utility = None

    def are_you_satisfied(self, partner_good, partner_type, proportions):

        # Choose the right classifier

        # Equation 5
        m_index = self.exchange_classifier_system.get_potential_bidders(self.H, partner_good)

        # Equation 6
        self.best_exchange_classifier = self.exchange_classifier_system.get_best_classifier(m_index)

        # Look at the decision

        # Equation 1
        exchange_decision = bool(self.best_exchange_classifier.decision)

        return exchange_decision

    def consume(self):

        # Re-initialize
        self.consumption = 0

        # Equation 7
        m_index = self.consumption_classifier_system.get_potential_bidders(self.H)

        # Equation 8
        new_best_consumption_classifier = self.consumption_classifier_system.get_best_classifier(m_index)

        # If he decides to consume...
        # Equation 3 & 4
        if new_best_consumption_classifier.decision == 1:

            # And the agent has his consumption good
            if self.H == self.C:
                self.consumption = 1

            # If he decided to consume, he produce a new unity of his production good
            self.H = self.P

        self.proceed_to_payments(new_best_consumption_classifier)

        # ----- FOR FUTURE ------- #

        # Compute utility
        self.utility = self.u * self.consumption \
            - self.storing_costs[self.H]

        # Will be the next best consumption classifier to update its weights
        self.best_consumption_classifier = new_best_consumption_classifier

    def proceed_to_payments(self, new_best_consumption_classifier):

        # Is there a winning exchange classifier?
        is_winning_exchange_classifier = \
            self.best_exchange_classifier.decision == 0 \
            or self.exchange is True

        if is_winning_exchange_classifier:

            self.best_exchange_classifier.update_theta_counter()
            exchange_classifier_bid = self.best_exchange_classifier.get_bid()

        else:
            exchange_classifier_bid = 0

        if self.best_consumption_classifier:

            self.best_consumption_classifier.update_strength(
                utility=self.utility,
                exchange_classifier_bid=exchange_classifier_bid
            )

        new_best_consumption_classifier.update_theta_counter()

        if is_winning_exchange_classifier:

            self.best_exchange_classifier.update_strength(new_best_consumption_classifier.get_bid())

# --------------------------------------------------------------------------------------------------- #
# -------------------------------- CLASSIFIER SYSTEM ------------------------------------------------ #
# --------------------------------------------------------------------------------------------------- #


class ClassifierSystem(object):

    def __init__(self):

        self.collection_of_classifiers = list()

        # Encoding of goods
        self.encoding_of_goods = np.array(
            [
                [1, 0, 0],  # Good 0
                [0, 1, 0],  # Good 1
                [0, 0, 1],  # Good 2
                [0, -1, -1],  # Not good 0
                [-1, 0, -1],  # Not good 1
                [-1, -1, 0]   # Not good 2
            ], dtype=int
        )

    def get_best_classifier(self, m_index):

        s = np.asarray([self.collection_of_classifiers[i].strength for i in m_index])

        best_m_idx = np.random.choice(np.where(s == max(s))[0])

        best_classifier_idx = m_index[best_m_idx]

        return self.collection_of_classifiers[best_classifier_idx]


class ExchangeClassifierSystem(ClassifierSystem):

    def __init__(self, b11, b12, initial_strength):

        super().__init__()

        self.b11 = b11
        self.b12 = b12
        self.initial_strength = initial_strength

        self.prepare_classifiers()

    def prepare_classifiers(self):

        idx = 0
        for i, j in product(self.encoding_of_goods, repeat=2):

            for k in [0, 1]:

                self.collection_of_classifiers.append(
                    ExchangeClassifier(
                        own_storage=i,
                        partner_storage=j,
                        decision=k,
                        strength=self.initial_strength,
                        b11=self.b11,
                        b12=self.b12,
                        idx=idx
                    )
                )
                idx += 1

    def get_potential_bidders(self, own_storage, partner_storage):

        # List of indexes of classifiers that match the current situation
        match_index = []
        for i, c in enumerate(self.collection_of_classifiers):

            if c.is_matching(own_storage, partner_storage):
                match_index.append(i)

        return match_index


class ConsumptionClassifierSystem(ClassifierSystem):

    def __init__(self, b21, b22, initial_strength):

        super().__init__()

        self.b21 = b21
        self.b22 = b22
        self.initial_strength = initial_strength

        self.prepare_classifiers()

    def prepare_classifiers(self):

        idx = 0

        for i in self.encoding_of_goods:

            for j in [0, 1]:

                self.collection_of_classifiers.append(
                    ConsumptionClassifier(
                        own_storage=i,
                        decision=j,
                        b21=self.b21,
                        b22=self.b22,
                        strength=self.initial_strength,
                        idx=idx
                    )
                )
                idx += 1

    def get_potential_bidders(self, own_storage):

        match_index = []
        for i, c in enumerate(self.collection_of_classifiers):

            if c.is_matching(own_storage):

                match_index.append(i)

        return match_index


# --------------------------------------------------------------------------------------------------- #
# -------------------------------- CLASSIFIER ------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #

class Classifier(object):

    def __init__(self, strength, decision, idx):

        self.strength = strength
        self.decision = decision

        # Equations 9 and 10
        self.theta_counter = 1

        self.idx = idx

    def update_theta_counter(self):

        self.theta_counter += 1


class ExchangeClassifier(Classifier):

    def __init__(self, own_storage, partner_storage, decision, strength, b11, b12, idx):

        super().__init__(strength=strength, decision=decision, idx=idx)

        self.own_storage = np.asarray(own_storage)
        self.partner_storage = np.asarray(partner_storage)

        self.sigma = 1 / (1 + sum(self.own_storage[:] == -1) + sum(self.partner_storage[:] == -1))

        # Equation 11a
        self.b1 = b11 + b12 * self.sigma

    def get_bid(self):

        # Def of a bid p 138
        return self.b1 * self.strength

    def update_strength(self, consumption_classifier_bid):

        self.strength -= (1 / self.theta_counter) * (
            self.get_bid() + self.strength
            - consumption_classifier_bid
        )

    def is_matching(self, own_storage, partner_storage):

        # Args are integers (0, 1 or 2)
        # self.own_storage is an array ([0, 0, 1] or [-1, -1, 0] and so on)
        cond_own_storage = self.own_storage[own_storage] != 0
        cond_partner_storage = self.partner_storage[partner_storage] != 0
        return cond_own_storage and cond_partner_storage

    def get_info(self):

        return "[Exchange {}] own_storage: {}, partner_storage: {},\n" \
               "decision: {}, strength: {}, bid: {}".format(
                self.idx, self.own_storage, self.partner_storage,
                self.decision, self.strength, self.get_bid()
                )


class ConsumptionClassifier(Classifier):

    def __init__(self, own_storage, strength, b21, b22, decision, idx):

        super().__init__(strength=strength, decision=decision, idx=idx)

        # Object in hand at the end of the turn
        self.own_storage = np.asarray(own_storage)

        sigma = 1 / (1 + sum(self.own_storage[:] == -1))

        # Equation 11b
        self.b2 = b21 + b22 * sigma

    def get_bid(self):

        return self.b2 * self.strength

    def update_strength(self, exchange_classifier_bid, utility):

        # Equation 12
        self.strength -= (1 / (self.theta_counter - 1)) * (
            self.get_bid() + self.strength
            - exchange_classifier_bid - utility
        )

    def is_matching(self, own_storage):

        return self.own_storage[own_storage] != 0

    def get_info(self):

        return "[Consumption {}] own_storage: {}, " \
              "decision: {}, strength: {}, bid: {}".format(
                self.idx, self.own_storage,
                self.decision, self.strength, self.get_bid()
              )

# --------------------------------------------------------------------------------------------------- #
# -------------------------------- CLASSIFIER SYSTEM ------------------------------------------------ #
# --------------------------------------------------------------------------------------------------- #


class MarimonEconomy(Economy):

    def __init__(self, repartition_of_roles, t_max, storing_costs,
                 b11, b12, b21, b22, initial_strength, u):
        
        super().__init__(repartition_of_roles=repartition_of_roles, t_max=t_max, storing_costs=storing_costs,
                         agent_model=MarimonAgent, u=u)

        self.exchange_classifier_systems = []
        self.consumption_classifier_systems = []

        for i in range(3):
            self.exchange_classifier_systems.append(
                ExchangeClassifierSystem(
                    b11=b11,
                    b12=b12,
                    initial_strength=initial_strength)
            )
            self.consumption_classifier_systems.append(
                ConsumptionClassifierSystem(
                    b21=b21,
                    b22=b22,
                    initial_strength=initial_strength)
            )

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.repartition_of_roles):

            i, j = self.roles[agent_type]

            for ind in range(n):
                a = MarimonAgent(
                    prod=i, cons=j,
                    storing_costs=self.storing_costs,
                    idx=agent_idx,
                    exchange_classifier_system=self.exchange_classifier_systems[agent_type],
                    consumption_classifier_system=self.consumption_classifier_systems[agent_type],
                    u=self.u
                )

                agents.append(a)
                agent_idx += 1

        return agents


def main():

    parameters = {
        "t_max": 500,
        "u": 100, "b11": 0.025, "b12": 0.025, "b21": 0.25, "b22": 0.25, "initial_strength": 0,
        "repartition_of_roles": np.array([50, 50, 50]),
        "storing_costs": np.array([0.1, 1., 20.]),
    }

    e = MarimonEconomy(**parameters)
    backup = e.run()

    parameters["agent_parameters"] = {"u": parameters["u"], "b11": parameters["b11"], "b12": parameters["b12"],
                                      "b21": parameters["b21"], "b22": parameters["b22"],
                                      "initial_strength": parameters["initial_strength"]}

    parameters["agent_model"] = type("", (object, ), {"name": "Marimon"})()

    represent_results(backup=backup, parameters=parameters)


if __name__ == "__main__":

    main()