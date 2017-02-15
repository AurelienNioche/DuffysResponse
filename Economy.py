import numpy as np
from tqdm import tqdm
from KWModels import ModelA


class Economy(object):

    def __init__(self, role_repartition, t_max, storing_costs, agent_model, u=100, beta=0.9,
                 agent_parameters=None, kw_model=ModelA):

        self.t_max = t_max

        self.agent_parameters = agent_parameters

        self.role_repartition = np.asarray(role_repartition)

        self.storing_costs = storing_costs
        self.u = u
        self.beta = beta

        self.n_agent = sum(role_repartition)

        self.kw_model = kw_model

        self.agent_model = agent_model

        self.agents = None
        
        # ----- For backup at t ----- #

        self.exchanges = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
        self.n_exchange = 0
        self.consumption = 0
        self.third_good_acceptance = np.zeros(3)
        self.proposition_of_third_object = np.zeros(3)

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        self.proportions = np.zeros((3, 3))
        
        # ---- For final backup ----- #

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.role_repartition):

            i, j, k = self.kw_model.roles[agent_type]

            for ind in range(n):
                a = self.agent_model(
                    agent_parameters=self.agent_parameters,
                    prod=i, cons=j, third=k, u=self.u, beta=self.beta,
                    storing_costs=self.storing_costs,
                    idx=agent_idx)

                agents.append(a)
                agent_idx += 1

        return agents

    def run(self):

        self.agents = self.create_agents()
        return self.play()
    
    def play(self):

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
            for k in self.exchanges.keys():
                self.exchanges[k] = 0
            self.n_exchange = 0
            self.consumption = 0
            self.third_good_acceptance[:] = 0
            self.proposition_of_third_object[:] = 0

            # ----- COMPUTE PROPORTIONS ----- #
            # Container for proportions of agents having this or that in hand according to their type
            #  - rows: type of agent
            # - columns: type of good

            self.proportions[:] = 0
            for i in self.agents:
                self.proportions[i.C, i.in_hand] += 1  # Type of agent is his consumption good

            self.proportions[:] = self.proportions / (self.n_agent // 3)

            # --------------------------------- #

            # ---------- MANAGE EXCHANGES ----- #
            # Take a random order among the indexes of the agents.
            agent_pairs = np.random.choice(self.n_agent, size=(self.n_agent // 2, 2), replace=False)

            for i, j in agent_pairs:

                self.make_encounter(i, j)

            # Each agent consumes at the end of each round and adapt his behavior (or not).
            for agent in self.agents:
                agent.consume()

                # Keep a trace from utilities
                self.consumption += agent.consumption

            # ----- FOR FUTURE BACKUP ----- #
            for key in self.exchanges.keys():
                # Avoid division by zero
                if self.n_exchange > 0:
                    self.exchanges[key] /= self.n_exchange
                else:
                    self.exchanges[key] = 0

            for i in range(3):
                # Avoid division by zero
                if self.proposition_of_third_object[i] > 0:
                    self.third_good_acceptance[i] = self.third_good_acceptance[i] / self.proposition_of_third_object[i]

                else:
                    self.third_good_acceptance[i] = 0

            self.consumption /= self.n_agent

            # For back up
            back_up["exchanges"].append(self.exchanges.copy())
            back_up["consumption"].append(self.consumption)
            back_up["n_exchanges"].append(self.n_exchange)
            back_up["third_good_acceptance"].append(self.third_good_acceptance.copy())
            back_up["proportions"].append(self.proportions.copy())
            # ----------------------------- #

        return back_up
    
    def make_encounter(self, i, j):

        i_H, j_H = self.agents[i].in_hand, self.agents[j].in_hand        
        i_P, j_P = self.agents[i].P, self.agents[j].P
        i_T, j_T = self.agents[i].T, self.agents[j].T
        i_C, j_C = self.agents[i].C, self.agents[j].C

        # Each agent is "initiator' of an exchange during one period.
        # Remember that consumption good = type of agent 
        i_agreeing = self.agents[i].are_you_satisfied(j_H, j_C, self.proportions) 
        # is his consumption good
        j_agreeing = self.agents[j].are_you_satisfied(i_H, i_C, self.proportions)

        # Consider particular case of offering third object
        i_facing_T = j_H == i_T and i_H == i_P
        j_facing_T = i_H == j_T and j_H == j_P
        
        # ---- STATS ------ #
      
        if i_facing_T:
            self.proposition_of_third_object[i_C] += 1
            if i_agreeing:
                self.third_good_acceptance[i_C] += 1 

        if j_facing_T:
            self.proposition_of_third_object[j_C] += 1
            if j_agreeing:
                self.third_good_acceptance[j_C] += 1
                
        # ------------ #

        # If both agents agree to exchange...
        if i_agreeing and j_agreeing:

            # ...exchange occurs
            self.agents[i].proceed_to_exchange(j_H)
            self.agents[j].proceed_to_exchange(i_H)

            # ---- STATS ------ #
            exchange_type = tuple(sorted([i_H, j_H]))
            if i_H != j_H:
                self.exchanges[exchange_type] += 1
                self.n_exchange += 1

            # ---------------- #

        else:

            self.agents[i].proceed_to_exchange(None)
            self.agents[j].proceed_to_exchange(None)


def launch(**kwargs):
    
    e = Economy(**kwargs)
    return e.run()
