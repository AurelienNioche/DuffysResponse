import numpy as np
from module.useful_functions import softmax


class Player:

    def __init__(self, consumption_benefit, agent_type):

        self.possible_goods = ['cyan', 'yellow', 'magenta']
        self.possible_types = ['cyan', 'yellow', 'magenta']
        self.consumption_benefit = consumption_benefit

        self.type = agent_type

        self.already_produced = 0

        self.consumption_good = self.possible_goods[agent_type]
        self.production_good = self.possible_goods[(agent_type - 1) % 3]

        self.my_good = self.production_good

        # .... and plenty of other attributes

    def should_I_exchange(self, proposed_good, proportion=0):

        proposed_good_is_of_my_type = proposed_good == self.type
        if proposed_good_is_of_my_type:

            return "ouaiiis"

        else:
            return "va te..."


if __name__ == '__main__':

    p = Player(consumption_benefit=100, agent_type=0)
    result = p.should_I_exchange(proposed_good="magenta")
    print(result)