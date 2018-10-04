import numpy as np
def linear_remapping(cardinality, vector):
    pow_vec = cardinality ** np.linspace(vector.size-1,0,vector.size)
    return np.dot(vector, pow_vec).astype(int)

def reverse_mapping(cardinality , length, value):
    out = np.empty(length).astype(int)
    for i, pot in enumerate(cardinality ** np.linspace(length-1,0,length)):
        digit = np.floor(value/pot)
        out[i] = digit
        value -= digit*pot
    return out