import numpy as np
from AbstractClasses import Agent


class MarimonAgent(Agent):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.exchange_classifier_system = ExchangeClassifierSystem()
        self.consumption_classifier_system = ConsumptionClassifierSystem()

        self.previous_object_in_hand = self.P

        self.utility_derived_from_consumption = 3

    def are_you_satisfied(self, proposed_object, type_of_other_agent, proportions):

        # Choose the right classifier
        m_index = self.exchange_classifier_system.get_potential_bidders(self.in_hand, proposed_object)
        best_classifier = self.exchange_classifier_system.get_best_classifier(m_index)

        # Look at the decision
        exchange_decision = bool(best_classifier.decision)

        return exchange_decision

    def consume(self):

        # Re-initialize

        # Choose the right classifier
        m_index = self.consumption_classifier_system.get_potential_bidders(self.in_hand)
        best_consumption_classifier = self.consumption_classifier_system.get_best_classifier(m_index)

        # Update theta counters
        # Equation 9
        if not self.exchange_classifier_system.get_decision() and \
                not self.previous_object_in_hand != self.in_hand:
            self.exchange_classifier_system.update_counter_of_winner_classifier()
            indicated_exchange_classifier = \
                self.exchange_classifier_system.get_selected_classifier()
        else:
            indicated_exchange_classifier = None

        self.consumption_classifier_system.update_counter_of_winner_classifier()

        # This is the strange part: agent choose if he consumes or not
        consumption_decision = best_consumption_classifier.decision

        # If he decides to consume...
        if consumption_decision:

            # And the agent has his consumption good
            if self.in_hand == self.C:

                self.consumption = 1

            # If he decided to consume, he produce a new unity of his production good
            self.in_hand = self.P

        # Keep a trace of the previous object in hand for updating the exchange theta counter
        self.previous_object_in_hand = self.in_hand

        # Make the payments
        self.consumption_classifier_system.make_payments(
            utility_derived_from_consumption=self.consumption*self.utility_derived_from_consumption,
            indicated_exchange_classifier=indicated_exchange_classifier

        )


# ---------------------------------- ABSTRACT CLASSES ---------------------------------------------------------- #

class ClassifierSystem(object):

    def __init__(self):

        self.collection_of_classifiers = list()
        self.selected_classifier_index = None
        self.previous_selected_classifier_index = None

        # To convert good described as int into ternary
        self.int_to_ternary = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 2: np.array([0, 0, 1])}

    def get_best_classifier(self, m_index):
        
        # Keep a trace of the previous selected classifier
        self.previous_selected_classifier_index = self.selected_classifier_index

        s = np.asarray([self.collection_of_classifiers[i].strength for i in m_index])
        best_c_index = np.random.choice(np.where(s == max(s)))

        self.selected_classifier_index = m_index[best_c_index]

        return self.collection_of_classifiers[self.selected_classifier_index]

    def compare_classifier_triad_with_int(self, classifier_triad, int_to_compare_with):
        # Code: Meaning
        # 100: good1 (-> 0)
        # 010: good2 (-> 1)
        # 001: good3 (-> 2)
        # 0##: not good1 (-> 0)
        # #0#: not good2 (-> 1)
        # ##0: not good3 (-> 2)

        tern_good = self.int_to_ternary[int_to_compare_with]
        return classifier_triad == tern_good or \
            np.in1d(np.where(classifier_triad == 0)[0], np.where(tern_good == 0)[0]).all()

    def get_selected_classifier(self):
        return self.collection_of_classifiers[self.selected_classifier_index]


# -------------------------------------------------------------------------------------------------------------- #


class ExchangeClassifierSystem(ClassifierSystem):

    def __init__(self):

        super().__init__()
        self.b_11 = np.random.random() / 2
        self.b_12 = np.random.random() / 2

    def get_potential_bidders(self, own_storage, other_storage):

        # INPUT
        # z_{a, t} = (x_{a, t}, x_{rho_t, (a) t})

        # For a given state or "condition" Zat = (XoI., xp,(o),),
        # there will typically be a collection of classifiers within the classifier system
        # whose condition are satisfied

        # List of indexes of classifiers that match the current situation
        match_index = []
        for i, c in enumerate(self.collection_of_classifiers):

            if self.test_compatibility(c, own_storage, other_storage):
                match_index.append(i)
        # OUTPUT
        # M_e(z_{at}) = {e: Z_{at} matches the condition part of classifier e}.

        # The members of M.(zat) form a class of potential "bidders" in an "auction" whose purpose
        # is to determine which classifier makes the decision of agent a at time t.

        return match_index

    def test_compatibility(self, classifier, own_storage, other_storage):

        cond_own_storage = self.compare_classifier_triad_with_int(classifier.own_storage, own_storage)
        cond_other_storage = self.compare_classifier_triad_with_int(classifier.other_storage, other_storage)
        return cond_own_storage * cond_other_storage

    def get_decision(self):

        return self.collection_of_classifiers[self.selected_classifier_index].trading_decision

    def update_counter_of_winner_classifier(self):

        self.collection_of_classifiers[self.selected_classifier_index].update_theta_counter()

    def make_payments(self):
        
        self.collection_of_classifiers[self.selected_classifier_index].update_strength()


class ConsumptionClassifierSystem(ClassifierSystem):

    def __init__(self):

        super().__init__()

    def get_potential_bidders(self, own_storage):

        match_index = []
        for i, c in enumerate(self.collection_of_classifiers):

            if self.compare_classifier_triad_with_int(c.own_storage, own_storage):

                match_index.append(i)

        return match_index

    def update_counter_of_winner_classifier(self):

        # Equation 10
        self.collection_of_classifiers[self.selected_classifier_index].update_theta_counter()

    def make_payments(self, indicated_exchange_classifier, utility_derived_from_consumption):

        if self.previous_selected_classifier_index is not None:
            self.collection_of_classifiers[self.previous_selected_classifier_index].update_strength(
                indicated_exchange_classifier, utility_derived_from_consumption)


# --------------------------------------------------------------------------------------------------- #

class Classifier(object):

    def __init__(self, parent):

        self.parent = parent

        self.strength = 1
        self.previous_strength = 1
        self.theta_counter = 1
        self.previous_theta_counter = 1

        self.decision = np.random.choice([0, 1])

    def update_theta_counter(self):
        self.previous_theta_counter = self.theta_counter
        self.theta_counter += 1


class ExchangeClassifier(Classifier):

    def __init__(self, parent,
                 own_storage=np.random.choice([0, 1, np.nan], size=3, replace=True),
                 partner_storage=np.random.choice([0, 1, np.nan], size=3, replace=True)):

        super().__init__(parent)

        self.own_storage = own_storage
        self.partner_storage = partner_storage

        self.sigma = 1 / (1 + np.sum(self.own_storage == np.nan) + np.sum(self.partner_storage == np.nan))

        # Equation 11a
        self.b1 = self.parent.b11 + self.parent.b12 * self.sigma

    def update_strength(self, indicated_consumption_classifier):

        consumption_classifier_bid = indicated_consumption_classifier.b2 * indicated_consumption_classifier.strength

        self.strength = self.previous_strength - (1 / self.theta_counter) * (
            (1 + self.b1) * self.previous_strength
            - consumption_classifier_bid
        )


class ConsumptionClassifier(Classifier):

    def __init__(self, parent):

        super().__init__(parent)

        # Object in hand at the end of the turn
        self.own_storage = np.random.choice([0, 1, np.nan], size=3, replace=True)

        self.sigma = 1 / (1 + np.sum(self.own_storage == np.nan))

        # Equation 11b
        self.b2 = self.parent.b21 + self.parent.b22 * self.sigma

    def update_strength(self, indicated_exchange_classifier, utility_derived_from_consumption):

        self.previous_strength = self.strength

        if indicated_exchange_classifier:
            exchange_classifier_bid = indicated_exchange_classifier.b1 * indicated_exchange_classifier.strength
        else:
            exchange_classifier_bid = 0

        # Equation 12
        self.strength = self.previous_strength - (1 / indicated_exchange_classifier.previous_theta) * (
            (1 + self.b2) * self.previous_strength
            - exchange_classifier_bid - utility_derived_from_consumption
        )






