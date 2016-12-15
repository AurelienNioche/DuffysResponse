import numpy as np
from tqdm import tqdm
from os import path
from RL import RLAgent
from analysis import represent_results


class ModelA(object):

    roles = np.array([
        [1, 0, 2],
        [2, 1, 0],
        [0, 2, 1]
    ], dtype=int)

    def __str__(self):

        return "Model A"


class ModelB(object):

    roles = np.array([
        [2, 0, 1],
        [0, 1, 2],
        [1, 2, 0]
    ], dtype=int)

    def __str__(self):

        return "Model B"


class Economy(object):

    def __init__(self, role_repartition, t_max, alpha, temp, storing_costs, model):

        self.t_max = t_max

        self.alpha = alpha
        self.temp = temp

        self.role_repartition = role_repartition

        self.storing_costs = storing_costs

        self.n_agent = sum(role_repartition)

        self.model = model

        self.agents = None

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.role_repartition):

            i, j, k = self.model.roles[agent_type]

            for ind in range(n):
                a = RLAgent(
                    alpha=self.alpha,
                    temp=self.temp,
                    prod=i, cons=j, third=k,
                    storing_costs=self.storing_costs,
                    agent_type=agent_type,
                    model=self.model,
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
            # --------------------------------- #

            # ---------- MANAGE EXCHANGES ----- #
            # Take a random order among the indexes of the agents.
            agent_pairs = np.random.choice(self.n_agent, size=(self.n_agent // 2, 2), replace=False)

            for i, j in agent_pairs:

                i_object = _[i].in_hand
                j_object = _[j].in_hand

                # Each agent is "initiator' of an exchange during one period.
                i_agreeing = _[i].are_you_satisfied(j_object, proportions)
                j_agreeing = _[j].are_you_satisfied(i_object, proportions)

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
                    if j_object == _[i].T:
                        third_good_acceptance[_[i].type] += 1

                    if i_object == _[j].T:
                        third_good_acceptance[_[j].type] += 1
                    # ------------ #

                    # ...exchange occurs
                    _[i].in_hand = j_object
                    _[j].in_hand = i_object

                    # ---- STATS ------ #
                    exchange_type = tuple(sorted([i_object, j_object]))
                    exchanges[exchange_type] += 1
                    n_exchange += 1
                    # ----------- #

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


def main():

    parameters = {
        "t_max": 500,
        "alpha": 0.1,
        "temp": 0.01,
        "role_repartition": np.array([500, 500, 500]),
        "storing_costs": np.array([0.1, 0.25, 0.265]),
        "model": ModelA()
    }

    backup = \
        launch(
            t_max=parameters["t_max"], alpha=parameters["alpha"], temp=parameters["temp"],
            role_repartition=parameters["role_repartition"],
            storing_costs=parameters["storing_costs"],
            model=parameters["model"]
        )

    represent_results(backup=backup, parameters=parameters, fig_name=path.expanduser("~/Desktop/KW.pdf"))


if __name__ == "__main__":

    main()
