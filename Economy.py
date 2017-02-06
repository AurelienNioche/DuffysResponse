import numpy as np
from tqdm import tqdm
from RL import RLAgent
from KWModels import ModelA


class Economy(object):

    def __init__(self, role_repartition, t_max, storing_costs,
                 agent_parameters=None, agent_model=RLAgent, kw_model=ModelA):

        self.t_max = t_max

        self.agent_parameters = agent_parameters

        self.role_repartition = role_repartition

        self.storing_costs = storing_costs

        self.n_agent = sum(role_repartition)

        self.kw_model = kw_model

        self.agent_model = agent_model

        self.agents = None

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.role_repartition):

            i, j, k = self.kw_model.roles[agent_type]

            for ind in range(n):
                a = self.agent_model(
                    agent_parameters=self.agent_parameters,
                    prod=i, cons=j, third=k,
                    storing_costs=self.storing_costs,
                    agent_type=agent_type,
                    kw_model=self.kw_model,
                    idx=agent_idx)

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
            "third_good_acceptance": []
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

            proportions[:] = proportions / self.n_agent

            # --------------------------------- #

            # ---------- MANAGE EXCHANGES ----- #
            # Take a random order among the indexes of the agents.
            agent_pairs = np.random.choice(self.n_agent, size=(self.n_agent // 2, 2), replace=False)

            for i, j in agent_pairs:

                i_object = _[i].in_hand
                j_object = _[j].in_hand

                # Each agent is "initiator' of an exchange during one period.
                i_agreeing = _[i].are_you_satisfied(j_object, _[j].type, proportions)
                j_agreeing = _[j].are_you_satisfied(i_object, _[i].type, proportions)

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

            # Each agent consumes at the end of each round and adapt his behavior (or not).
            for agent in self.agents:
                
                agent.consume()

                # Keep a trace from utilities
                consumption += agent.consumption

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
            back_up["exchanges"].append(exchanges.copy())
            back_up["consumption"].append(consumption)
            back_up["n_exchanges"].append(n_exchange)
            back_up["third_good_acceptance"].append(third_good_acceptance.copy())

            # ----------------------------- #

        return back_up


def launch(**kwargs):
    
    e = Economy(**kwargs)
    return e.play()
