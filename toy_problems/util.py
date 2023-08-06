import numpy as np

def encode_state(state, row=12, col=9):
    one_hot = np.zeros(row*col)
    one_hot[col * state[0] + state[1]] = 1
    return one_hot
