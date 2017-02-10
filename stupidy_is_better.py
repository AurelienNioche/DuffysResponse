from AbstractAgent import Agent


class StupidAgent(Agent):

    name = "Stupid agent"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)


class TotalGogol(Agent):

    ''' Encore pire'''

    name = "TotalGogol"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):
        return 0.5