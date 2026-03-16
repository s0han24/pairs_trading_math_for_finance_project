import numpy as np


def generate_positions(z, entry=2, exit=0.5):

    position = np.zeros(len(z))

    for t in range(1, len(z)):

        if z.iloc[t] > entry:
            position[t] = -1

        elif z.iloc[t] < -entry:
            position[t] = 1

        elif abs(z.iloc[t]) < exit:
            position[t] = 0

        else:
            position[t] = position[t - 1]

    return position
