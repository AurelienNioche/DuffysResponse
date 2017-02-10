import numpy as np


def save(backup):

    for i, strategies in enumerate(backup["last_strategies"]):

        print("Agent {}".format(i))
        for key, value in strategies.items():
            print("{} {:.2f}, {:.2f}".format(key, value[0], value[1]))

        print()

    print(backup)