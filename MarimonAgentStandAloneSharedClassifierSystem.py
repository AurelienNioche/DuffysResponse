import numpy as np
from os import path
from itertools import product
from tqdm import tqdm
from pylab import plt
from analysis import represent_results



class ModelA(object):

    roles = np.array([
        [1, 0, 2],
        [2, 1, 0],
        [0, 2, 1]
    ], dtype=int)

    name = "ModelA"

# --------------------------------------------------------------------------------------------------- #
# -------------------------------- MARIMON AGENT ---------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #


class MarimonAgent(object):

    name = "Marimon"

    def __init__(self, prod, cons, third, agent_type, idx, storing_costs, exchange_classifier_system,
                 consumption_classifier_system, u):

        # Production object (integer in [0, 1, 2])
        self.P = prod

        # Consumption object (integer in [0, 1, 2])
        self.C = cons

        # Other object (integer in [0, 1, 2])
        self.T = third

        # Type of agent (integer in [0, 1, 2])
        self.type = agent_type

        # Index of agent (more or less his name ; integer in [0, ..., n] with n : total number of agent)
        self.idx = idx

        # Utility derived from consumption
        self.utility_derived_from_consumption = u

        # Parameters for agent that could be different in nature depending on the agent model in use (Python dictionary)
        self.exchange_classifier_system = exchange_classifier_system
        self.consumption_classifier_system = consumption_classifier_system

        # Storing costs (numpy array of size 3)
        self.storing_costs = storing_costs

        # Keep a trace for time t if the agent consumed or not.
        self.consumption = 0

        # Keep a trace whether the agent proceed to an exchange
        self.exchange = None

        # Object an agent has in hand
        self.in_hand = self.P

        self.previous_object_in_hand = self.P

        self.best_exchange_classifier = None
        self.best_consumption_classifier = None

        self.bid_from_exchange_classifier = None
        self.bid_from_consumption_classifier = None

        self.utility = None

    def wants_to_exchange(self, proposed_object):

        # Choose the right classifier

        # Equation 5
        m_index = self.exchange_classifier_system.get_potential_bidders(self.in_hand, proposed_object)

        # Equation 6
        self.best_exchange_classifier = self.exchange_classifier_system.get_best_classifier(m_index)

        # Look at the decision

        # Equation 1
        exchange_decision = bool(self.best_exchange_classifier.decision)

        return exchange_decision

    def proceed_to_exchange(self, new_object):

        if new_object is not None:

            self.exchange = True
            self.in_hand = new_object

        else:
            self.exchange = False

    def consume(self):

        # Re-initialize
        self.consumption = 0

        # Equation 7
        m_index = self.consumption_classifier_system.get_potential_bidders(self.in_hand)

        # Equation 8
        new_best_consumption_classifier = self.consumption_classifier_system.get_best_classifier(m_index)

        # If he decides to consume...
        # Equation 3 & 4
        if new_best_consumption_classifier.decision == 1:

            # And the agent has his consumption good
            if self.in_hand == self.C:
                self.consumption = 1

            # If he decided to consume, he produce a new unity of his production good
            self.in_hand = self.P

        self.proceed_to_payments(new_best_consumption_classifier)

        # ----- FOR FUTURE ------- #

        # Compute utility
        self.utility = self.utility_derived_from_consumption * self.consumption \
            - self.storing_costs[self.in_hand]

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


class Economy(object):

    def __init__(self, role_repartition, t_max, storing_costs,
                 b11, b12, b21, b22, initial_strength, u,
                 kw_model=ModelA):

        self.t_max = t_max

        self.role_repartition = role_repartition

        self.storing_costs = storing_costs

        self.n_agent = sum(role_repartition)

        self.kw_model = kw_model

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

        self.u = u

        self.agents = None

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.role_repartition):

            i, j, k = self.kw_model.roles[agent_type]

            for ind in range(n):
                a = MarimonAgent(
                    prod=i, cons=j, third=k,
                    storing_costs=self.storing_costs,
                    agent_type=agent_type,
                    idx=agent_idx,
                    exchange_classifier_system=self.exchange_classifier_systems[agent_type],
                    consumption_classifier_system=self.consumption_classifier_systems[agent_type],
                    u=self.u
                )

                agents.append(a)
                agent_idx += 1

        return agents

    def play(self):

        self.agents = self.create_agents()
        _ = self.agents

        # For future back up
        back_up = {
            "exchanges": [],
            "n_exchanges": [],
            "consumption": [],
            "third_good_acceptance": [],
            "proportions": []
        }

        for t in tqdm(range(self.t_max)):

            # Containers for future backup
            exchanges = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
            n_exchange = 0
            consumption = 0
            third_good_acceptance = np.zeros(3)
            proposition_of_third_object = np.zeros(3)

            # ----- COMPUTE PROPORTIONS ----- #
            # Container for proportions of agents having this or that in hand according to their type
            #  - rows: type of agent
            # - columns: type of good

            proportions = np.zeros((3, 3))
            for i in self.agents:
                proportions[i.type, i.in_hand] += 1

            proportions[:] = proportions / (self.n_agent//3)

            # --------------------------------- #

            # ---------- MANAGE EXCHANGES ----- #
            # Take a random order among the indexes of the agents.
            agent_pairs = np.random.choice(self.n_agent, size=(self.n_agent // 2, 2), replace=False)

            for i, j in agent_pairs:

                i_object = _[i].in_hand
                j_object = _[j].in_hand

                # Each agent is "initiator' of an exchange during one period.
                i_agreeing = _[i].wants_to_exchange(j_object)
                j_agreeing = _[j].wants_to_exchange(i_object)

                # ---- STATS ------ #
                # Consider particular case of offering third object
                if j_object == _[i].T and i_object == _[i].P:
                    proposition_of_third_object[_[i].type] += 1

                if i_object == _[j].T and j_object == _[j].P:
                    proposition_of_third_object[_[j].type] += 1
                # ------------ #

                # If both agents agree to exchange...
                if i_agreeing and j_agreeing:

                    # ---- STATS ------ #
                    # Consider particular case of offering third object
                    if j_object == _[i].T and i_object == _[i].P:
                        third_good_acceptance[_[i].type] += 1

                    if i_object == _[j].T and j_object == _[j].P:
                        third_good_acceptance[_[j].type] += 1

                    # ------------ #

                    # ...exchange occurs
                    _[i].proceed_to_exchange(j_object)
                    _[j].proceed_to_exchange(i_object)

                    # ---- STATS ------ #
                    exchange_type = tuple(sorted([i_object, j_object]))
                    if i_object != j_object:
                        exchanges[exchange_type] += 1
                        n_exchange += 1
                        # ----------- #

                else:

                    _[i].proceed_to_exchange(None)
                    _[j].proceed_to_exchange(None)

                # Each agent consumes at the end of encounter and adapt his behavior (or not).
                _[i].consume()
                _[j].consume()

                # Keep a trace from utilities
                consumption += _[i].consumption + _[j].consumption

            # ----- FOR FUTURE BACKUP ----- #
            for key in exchanges.keys():
                # Avoid division by zero
                if n_exchange > 0:
                    exchanges[key] /= n_exchange
                else:
                    exchanges[key] = 0

            for i in range(3):
                # Avoid division by zero
                if proposition_of_third_object[i] > 0:
                    third_good_acceptance[i] = third_good_acceptance[i] / proposition_of_third_object[i]

                else:
                    third_good_acceptance[i] = 0

            consumption /= self.n_agent

            # For back up
            back_up["proportions"].append(proportions.copy())
            back_up["exchanges"].append(exchanges.copy())
            back_up["consumption"].append(consumption)
            back_up["n_exchanges"].append(n_exchange)
            back_up["third_good_acceptance"].append(third_good_acceptance.copy())

            # ----------------------------- #

        return back_up


def plot_proportions(proportions, fig_name):

    # Container for proportions of agents having this or that in hand according to their type
    #  - rows: type of agent
    # - columns: type of good

    fig = plt.figure(figsize=(25, 12))
    fig.patch.set_facecolor('white')

    n_lines = 3
    n_columns = 1

    x = np.arange(len(proportions))

    for agent_type in range(3):

        # First subplot
        ax = plt.subplot(n_lines, n_columns, agent_type + 1)
        ax.set_title("Proportion of agents of type {} having good 1, 2, 3 in hand\n".format(agent_type + 1))

        y0 = []
        y1 = []
        y2 = []
        for proportions_at_t in proportions:
            y0.append(proportions_at_t[agent_type, 0])
            y1.append(proportions_at_t[agent_type, 1])
            y2.append(proportions_at_t[agent_type, 2])

        ax.set_ylim([-0.02, 1.02])

        ax.plot(x, y0, label="Good 1", linewidth=2)
        ax.plot(x, y1, label="Good 2", linewidth=2)
        ax.plot(x, y2, label="Good 3", linewidth=2)
        ax.legend()

    plt.tight_layout()

    plt.savefig(filename=fig_name)


def main():

    parameters = {
        "t_max": 500,
        "u": 100, "b11": 0.025, "b12": 0.025, "b21": 0.25, "b22": 0.25, "initial_strength": 0,
        "role_repartition": np.array([50, 50, 50]),
        "storing_costs": np.array([0.1, 1., 20.]),
        "kw_model": ModelA
    }

    e = Economy(**parameters)
    backup = e.play()

    with open(path.expanduser("~/Desktop/save.txt"), 'w') as f:
        for i in range(3):
            for j in e.exchange_classifier_systems[i].collection_of_classifiers:
                f.write(j.get_info())
                f.write("\n")
            f.write("\n")

    fig_name = path.expanduser("~/Desktop/KW_Marimon_Agents.pdf")
    init_fig_name = fig_name.split(".")[0]
    i = 2
    while path.exists(fig_name):
        fig_name = "{}{}.pdf".format(init_fig_name, i)
        i += 1

    parameters["agent_parameters"] = {"u": parameters["u"], "b11": parameters["b11"], "b12": parameters["b12"],
                                      "b21": parameters["b21"], "b22": parameters["b22"],
                                      "initial_strength": parameters["initial_strength"]}

    parameters["agent_model"] = type("", (object, ), {"name": "Marimon"})()

    represent_results(backup=backup, parameters=parameters, fig_name=fig_name)

    plot_proportions(proportions=backup["proportions"], fig_name=fig_name.split(".")[0] + "_proportions.pdf")


if __name__ == "__main__":

    main()