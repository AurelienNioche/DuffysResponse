import numpy as np
from data_manager import import_data


class StupidAgent(object):

    name = "Stupid agent"

    def __init__(self, **kwargs):

        # Production object (integer in [0, 1, 2])
        self.P = kwargs["prod"]

        # Consumption object (integer in [0, 1, 2])
        self.C = kwargs["cons"]

        # Object an agent has in hand
        self.in_hand = self.P

    def probability_of_responding(self, subject_response, partner_good):

        if partner_good == self.C:
            if subject_response:
                return 1
            else:
                return 0

        else:
            return 0.5

    def do_the_encounter(self, partner_choice, partner_good, subject_choice):

        if subject_choice and partner_choice:
            self.in_hand = partner_good
            if self.in_hand == self.C:
                self.in_hand = self.P


class TotalGogol(StupidAgent):

    ''' Encore pire'''

    name = "TotalGogol"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def probability_of_responding(self, subject_response, partner_good):
        return 0.5


def main():

    pass


if __name__ == "__main__":

    main()