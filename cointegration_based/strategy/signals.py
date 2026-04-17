import numpy as np


def generate_positions(z, entry=2, exit=0.5, stop_loss=5):

    position = np.zeros(len(z))
    
    z = z.ffill().fillna(0)  # Forward-fill NaNs, then fill any leading NaNs with 0
    z = z.to_numpy()  # Convert to numpy array for faster processing

    for t in range(1, len(z)):

        if position[t - 1] == 0:
            if z[t] > entry:
                position[t] = -1  # Short
            elif z[t] < -entry:
                position[t] = 1   # Long
            else:
                position[t] = 0   # No position
                
        elif position[t - 1] == -1 and (z[t] < exit or z[t] > stop_loss):
            position[t] = 0  # Exit
        elif position[t - 1] == 1 and (z[t] > -exit or z[t] < -stop_loss):
            position[t] = 0  # Exit
        else:
            position[t] = position[t - 1]

    return position
