def compute_equilibrium(storing_costs, beta, u):
    if (storing_costs[2] - storing_costs[1]) < (2 ** 0.5 - 1) * (beta / 3) * u:
        return "speculative"

    elif (storing_costs[2] - storing_costs[1]) > 0.5 * (beta / 3) * u:

        return "fundamental"

    else:
        return "no equilibrium"
