from agent.stupid_agent import StupidAgent


class TotalGogol(StupidAgent):

    ''' Encore pire'''

    name = "TotalGogol"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def probability_of_responding(self, subject_response, partner_good, partner_type, proportions):
        return 0.5