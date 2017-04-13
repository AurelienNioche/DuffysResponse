import itertools as it
import numpy as np
from tqdm import tqdm

from environment.get_roles import get_roles


class EconomyWithoutBackUp(object):

    def __init__(self, repartition_of_roles, t_max, agent_model, storing_costs,
                 u=None, beta=None, agent_parameters=None):

        self.t_max = t_max
        self.agent_parameters = agent_parameters
        self.storing_costs = storing_costs
        self.u = u
        self.beta = beta
        self.agent_model = agent_model

        self.n_goods = len(storing_costs)
        self.roles = get_roles(self.n_goods)
        self.repartition_of_roles = np.asarray(repartition_of_roles)
        self.n_agent = sum(self.repartition_of_roles)

        self.agents = None

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.repartition_of_roles):

            i, j = self.roles[agent_type]

            for ind in range(n):
                a = self.agent_model(
                    prod=i, cons=j,
                    storing_costs=self.storing_costs,
                    u=self.u,
                    beta=self.beta,
                    agent_parameters=self.agent_parameters,
                    idx=agent_idx)

                agents.append(a)
                agent_idx += 1

        return agents

    def run(self):

        self.agents = self.create_agents()

        for t in tqdm(range(self.t_max)):
            self.time_step()

    def time_step(self):

        # ---------- MANAGE EXCHANGES ----- #
        # Take a random order among the indexes of the agents.
        agent_pairs = np.random.choice(self.n_agent, size=(self.n_agent // 2, 2), replace=False)

        for i, j in agent_pairs:
            self.make_encounter(i, j)

        # Each agent consumes at the end of each round and adapt his behavior (or not).
        for agent in self.agents:
            agent.consume()

    def make_encounter(self, i, j):

        i_agreeing, j_agreeing = self.seek_agreement(i=i, j=j, proportions=None)  # Will not work for KWAgents
        self.proceed_to_exchange(i=i, j=j, i_agreeing=i_agreeing, j_agreeing=j_agreeing)

    def seek_agreement(self, i, j, proportions):

        i_H, j_H = self.agents[i].H, self.agents[j].H
        i_C, j_C = self.agents[i].C, self.agents[j].C

        # Each agent is "initiator' of an exchange during one period.
        # Remember that consumption good = type of agent
        i_agreeing = self.agents[i].are_you_satisfied(partner_good=j_H, partner_type=j_C,
                                                      proportions=proportions)
        # is his consumption good
        j_agreeing = self.agents[j].are_you_satisfied(partner_good=i_H, partner_type=i_C,
                                                      proportions=proportions)
        return i_agreeing, j_agreeing

    def proceed_to_exchange(self, i, j, i_agreeing, j_agreeing):

        i_H, j_H = self.agents[i].H, self.agents[j].H

        # If both agents agree to exchange...
        if i_agreeing and j_agreeing:

            # ...exchange occurs
            self.agents[i].proceed_to_exchange(j_H)
            self.agents[j].proceed_to_exchange(i_H)

        else:
            self.agents[i].proceed_to_exchange(None)
            self.agents[j].proceed_to_exchange(None)


class Economy(EconomyWithoutBackUp):
    """ Economy class with full backup"""

    def __init__(self, **parameters):

        super().__init__(**parameters)

        # ----- For backup at t ----- #

        self.exchanges = dict()
        for i in it.combinations(range(self.n_goods), r=2):
            self.exchanges[i] = 0
        self.n_exchange = 0
        self.consumption = 0
        self.good_accepted_as_medium = np.zeros(self.n_goods)
        self.proposition_of_medium = np.zeros(self.n_goods)

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        self.proportions = np.zeros((self.n_goods, self.n_goods))

        # ---- For final backup ----- #
        self.back_up = {
            "exchanges": [],
            "n_exchanges": [],
            "consumption": [],
            "good_accepted_as_medium": [],
            "proportions": []
        }

    def run(self):

        super().run()
        return self.back_up

    def time_step(self):

        """
         Overrided method allowing for backup
        :return: None
        """

        self.reinitialize_backup_containers()
        self.compute_proportions()

        super().time_step()

        self.make_a_backup_for_t()

    def make_encounter(self, i, j):

        """
         Overrided method allowing for backup
        :return: None
        """

        i_agreeing, j_agreeing = self.seek_agreement(i=i, j=j, proportions=self.proportions)
        self.make_stats_about_medium_of_exchange(i=i, j=j, i_agreeing=i_agreeing, j_agreeing=j_agreeing)
        self.proceed_to_exchange(i=i, j=j, i_agreeing=i_agreeing, j_agreeing=j_agreeing)
        self.make_stats_about_exchanges(i=i, j=j, i_agreeing=i_agreeing, j_agreeing=j_agreeing)

    def make_stats_about_medium_of_exchange(self, i, j, i_agreeing, j_agreeing):

        i_H, j_H = self.agents[i].H, self.agents[j].H
        i_P, j_P = self.agents[i].P, self.agents[j].P
        i_C, j_C = self.agents[i].C, self.agents[j].C

        # Consider particular case of offering third object
        i_facing_M = j_H != i_C and i_H == i_P
        j_facing_M = i_H != j_C and j_H == j_P

        if i_facing_M:
            self.proposition_of_medium[j_H] += 1  # Consider as key the good that is proposed as a medium of ex
            if i_agreeing:
                self.good_accepted_as_medium[j_H] += 1

        if j_facing_M:
            self.proposition_of_medium[i_H] += 1
            if j_agreeing:
                self.good_accepted_as_medium[i_H] += 1

    def make_stats_about_exchanges(self, i, j, i_agreeing, j_agreeing):

        i_H, j_H = self.agents[i].H, self.agents[j].H

        if i_agreeing and j_agreeing:

            exchange_type = tuple(sorted([i_H, j_H]))
            if i_H != j_H:
                self.exchanges[exchange_type] += 1
                self.n_exchange += 1

    def reinitialize_backup_containers(self):

        # Containers for future backup
        for k in self.exchanges.keys():
            self.exchanges[k] = 0
        self.n_exchange = 0
        self.consumption = 0
        self.good_accepted_as_medium[:] = 0
        self.proposition_of_medium[:] = 0

        self.proportions[:] = 0

    def compute_proportions(self):

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        for i in self.agents:
            self.proportions[i.C, i.H] += 1  # Type of agent is his consumption good

        for i in range(self.n_goods):
            self.proportions[i] = self.proportions[i] / self.repartition_of_roles[i]

    def make_a_backup_for_t(self):

        # Keep a trace from utilities
        self.consumption = sum([a.consumption for a in self.agents]) / self.n_agent

        # ----- FOR FUTURE BACKUP ----- #

        for key in self.exchanges.keys():
            # Avoid division by zero
            if self.n_exchange > 0:
                self.exchanges[key] /= self.n_exchange
            else:
                self.exchanges[key] = 0

        for i in range(self.n_goods):
            # Avoid division by zero
            if self.proposition_of_medium[i] > 0:
                self.good_accepted_as_medium[i] = self.good_accepted_as_medium[i] / self.proposition_of_medium[i]

            else:
                self.good_accepted_as_medium[i] = 0

        assert 0 <= self.good_accepted_as_medium.all() <= 1

        # For back up
        self.back_up["exchanges"].append(self.exchanges.copy())
        self.back_up["consumption"].append(self.consumption)
        self.back_up["n_exchanges"].append(self.n_exchange)
        self.back_up["good_accepted_as_medium"].append(self.good_accepted_as_medium.copy())
        self.back_up["proportions"].append(self.proportions.copy())


def launch(**kwargs):
    
    e = Economy(**kwargs)
    return e.run()
